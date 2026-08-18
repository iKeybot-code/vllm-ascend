# vLLM Ascend Prefix Cache 完整逻辑详解

> 生成日期：2026-08-19
> 代码版本：`2d75463 [Enhancement][MRV2] Add potential max tokens setting in NPUModelRunner`
> 项目仓库：vLLM Ascend (插件架构，通过 entry_points 注入 vLLM 主项目)

---

## 整体架构流程

```
HTTP 请求 (API Server)
    ↓
AsyncLLMEngine.add_request()  →  Request 对象 + block_hashes 生成
    ↓  (asyncio 队列 / EngineCore 进程间通信)
EngineCore 主循环 → scheduler.schedule()
    ↓
┌─────────────────────────────────────────────────────────────┐
│                  Scheduler 调度核心                          │
│  1. running 队列处理 (decode 阶段)                            │
│  2. waiting 队列处理 (新请求 / prefill 阶段)                   │
│     → kv_cache_manager.get_computed_blocks()                │
│       → coordinator.find_longest_cache_hit()                │
│         → block_pool.get_cached_block(block_hash)           │
│  3. allocate_slots() 分配 KV blocks                          │
│  4. SchedulerOutput 输出 (NewRequestData/CachedRequestData) │
└─────────────────────────────────────────────────────────────┘
    ↓
Worker.execute_model() → ModelRunner.execute_model()
    ↓
┌─────────────────────────────────────────────────────────────┐
│               ModelRunner 执行核心                           │
│  prepare_inputs() → input_batch (slot_mappings 计算)         │
│     → AscendBlockTables.compute_slot_mappings()             │
│        Triton Kernel: block_table → position → slot_id      │
│  model.forward() → attention 计算                            │
│     → PagedAttention: slot_mappings 索引 KV cache           │
│     → reshape_and_cache: 新 token KV 写入 KV cache           │
│  Sampler 采样 → 新 token id                                  │
└─────────────────────────────────────────────────────────────┘
    ↓
Scheduler.update_from_output()
    → kv_cache_manager.cache_blocks()  ← **写入 prefix cache**
    → request.num_computed_tokens 更新
    → 完成检测 (EOS / max_tokens)
    ↓
EngineCoreOutput → API Server → HTTP 响应 (streaming/non-streaming)
```

---

## 第1层：API Server 接收 HTTP 请求 → 进入引擎

### 关键入口追踪

根据 `profiling_config.py` 标注的追踪点：

**OpenAI Chat Completion 入口**：
```
vllm.entrypoints.openai.chat_completion.serving:OpenAIServingChat.create_chat_completion
vllm.entrypoints.openai.completion.serving:OpenAIServingCompletion.create_completion
```

**请求生命周期入口**：
```
vllm.v1.engine.async_llm:AsyncLLM.add_request            # v1 引擎
vllm.engine.async_llm_engine:AsyncLLMEngine.add_request    # v0 引擎
```

### 请求流程简述

1. FastAPI/Quart HTTP Server 接收 `POST /v1/chat/completions`
2. `create_chat_completion` 解析 JSON body：`model`, `messages`, `max_tokens`, `stream` 等
3. Tokenizer 将 messages 编码为 `prompt_token_ids`
4. 调用 `LLM.generate()` / `AsyncLLMEngine.add_request()`，把请求封装为 `Request` 对象

**代码参考**：
- 入口符号定义：[profiling_config.py#L38-L153](file:///workspace/vllm_ascend/profiling_config.py#L38-L153)
- 测试夹具 API Server：[conftest.py#L267-L391](file:///workspace/tests/e2e/conftest.py#L267-L391)
- 插件注册入口：[`__init__.py` register()](file:///workspace/vllm_ascend/__init__.py#L73-L116)

---

## 第2层：Request 对象与 Block Hash 计算（Prefix Cache 的基础）

### Request 对象核心属性

每个请求都会计算一条 **block hash 链**，这是 prefix cache 匹配的关键依据。

**测试用例参考**：[test_compressed_prefix_cache.py#L34-L43](file:///workspace/tests/ut/test_compressed_prefix_cache.py#L34-L43)

```python
def _make_request(request_id: str, token_ids: list[int], hash_block_size: int) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        # ★ 关键：每个请求关联一个 block hasher，按 hash_block_size 切片算哈希
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )
```

### 链式 Block Hash 工作原理

- `hash_block_size`：计算 block hash 的粒度（如 128 tokens）
- 对于 prompt tokens `[t0, t1, t2, ..., tN]`，按 `hash_block_size` 分块：
  - Block 0: `[t0..t127]` → `hash0 = sha256(t0..t127)`
  - Block 1: `[t128..t255]` → `hash1 = sha256(hash0 + t128..t255)` **（链式哈希！）**
  - ... 形成一条 hash 链：`[hash0, hash1, hash2, ...]`
- **设计原因**：prefix 命中判断只需顺序比较，一旦某个 hash 不匹配，后续肯定不匹配，可直接 break

---

## 第3层：Scheduler 调度核心（最复杂部分）

### 核心调度类：RecomputeScheduler

**核心文件**：[recompute_scheduler.py](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L95-L939)

### schedule() 主流程详解

**入口**：[recompute_scheduler.py#L182-L939](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L182-L939)

```python
def schedule(self, throttle_prefills: bool = False) -> RecomputeSchedulerOutput:
    self.current_step += 1
```

#### 设计思想注释 [L184-L193]

> There's no "decoding phase" nor "prefill phase" in the scheduler.
> Each request just has the num_computed_tokens and num_tokens_with_spec.
> At each step, the scheduler tries to assign tokens to the requests
> so that each request's num_computed_tokens can catch up its
> num_tokens_with_spec. This is general enough to cover
> chunked prefills, prefix caching, speculative decoding,
> and the "jump decoding" optimization in the future.

翻译：调度器不分 "prefill 阶段" 和 "decode 阶段"。每个请求只有 `num_computed_tokens`（已算完的 token 数）和 `num_tokens_with_spec`（需要算到的 token 数，含投机 draft）。每步尽量让前者追上后者，这样天然覆盖分块 prefill、prefix cache、投机解码、以及未来的 jump decoding。

---

#### Part A：Running 请求处理（Decode / 继续 Prefill）

**位置**：[recompute_scheduler.py#L228-L460](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L228-L460)

```python
# 遍历 running 队列已有请求，尝试分配 token 预算
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # 跳过已达 max_tokens 或 max_model_len 的请求
    if (request.num_output_placeholders > 0
            and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            or request.num_computed_tokens >= self.max_model_len):
        req_index += 1
        continue

    # PP+async：强制同请求 decode 间隔 pp_size 步，匹配 broadcast ring
    if self.current_step < request.next_decode_eligible_step:
        req_index += 1
        continue

    # ★ 本次要调度的新 token 数 = 需要算到的 - 已算过的
    num_new_tokens = (
        request.num_tokens_with_spec + request.num_output_placeholders
        - request.num_computed_tokens
    )
    # 受长 prefill 阈值、全局 token_budget、max_model_len 限制
    if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = self.scheduler_config.long_prefill_token_threshold
    num_new_tokens = min(num_new_tokens, token_budget)
    num_new_tokens = min(
        num_new_tokens,
        self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step,
    )
```

##### 关键：allocate_slots 分配 KV Block Slots [L316-L326]

```python
# Schedule newly needed KV blocks for the request.
with record_function_or_nullcontext("schedule: allocate_slots"):
    while True:
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )

        if new_blocks is not None:
            break  # 分配成功

        # 分配失败 → 抢占策略
        transfer_config = self.vllm_config.kv_transfer_config
        # ★ PD 分离场景：preemption 走 recompute_offload（丢到 PD proxy 重算）
        if transfer_config is not None and not transfer_config.is_kv_producer:
            recomputed_req = self.running.pop()
            recomputed_block_ids = self.kv_cache_manager.get_block_ids(...)
            preempt_hook = getattr(self.connector, "update_state_before_preempt", None)
            offloaded = bool(preempt_hook(recomputed_req, ...)) if preempt_hook else False
            if offloaded:
                # 记录到 preempted_reqs，worker kv_consumer 会处理
                preempted_req_data.append(PreemptedRequestData(...))
                self._preempt_request(recomputed_req, scheduled_timestamp)
                preempted_reqs.append(recomputed_req)
            else:
                # offload 失败 → finish_recomputed：返回客户端重算
                self._finish_recomputed_request(recomputed_req, recomputed_reqs)
            if recomputed_req == request:
                break
        else:
            # 普通场景：按优先级 / FCFS 抢占
            if self.policy == SchedulingPolicy.PRIORITY:
                preempted_req = max(self.running,
                    key=lambda r: (r.priority, r.arrival_time))
                self.running.remove(preempted_req)
                # 回滚此 step 已分配给它的 token / blocks
                token_budget += num_scheduled_tokens.pop(preempted_req_id)
                req_to_new_blocks.pop(preempted_req_id)
                ...
            else:
                preempted_req = self.running.pop()

            self._preempt_request(preempted_req, scheduled_timestamp)
            preempted_reqs.append(preempted_req)
            if preempted_req == request:
                break  # 当前请求就是被抢占的那个 → 跳出
```

**抢占 (Preemption) 的含义**：将 running 请求的 KV blocks 全部释放，请求丢回 waiting 队列，下次被接纳时重新从 prefill 阶段开始（但仍可享受 prefix cache 命中，所以实际上只需要重算最后的非前缀部分）。

---

#### Part B：Waiting 请求处理 + Prefix Cache 匹配

**位置**：[recompute_scheduler.py#L471-L829](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L471-L829)

**这是 prefix cache 核心逻辑所在！**

```python
# Next, schedule the WAITING requests.
if not preempted_reqs and not recomputed_reqs and self._pause_state == PauseState.UNPAUSED:
    step_skipped_waiting = create_request_queue(self.policy)

    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        # 最大并发请求数限制
        num_running = len(self.running) + self.num_waiting_for_streaming_input
        if num_running >= self.max_num_running_reqs:
            break

        request_queue = self._select_waiting_queue_for_scheduling()
        request = request_queue.peek_request()
        request_id = request.request_id
```

##### 处理阻塞中的等待状态 [L488-L499]

```python
# try to promote blocked statuses while traversing skipped queue.
if self._is_blocked_waiting_status(request.status) and not self._try_promote_blocked_waiting_request(request):
    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        logger.debug("[RecomputeScheduler] %s is still in WAITING_FOR_REMOTE_KVS state.", request_id)
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue  # 仍然阻塞，移到 skipped，本轮跳过
```

##### ★ Step 1：Prefix Cache 查找 [L524-L580]

```python
# Get already-cached tokens.
if request.num_computed_tokens == 0:  # 全新请求，还没算过任何 token
    did_prefix_cache_lookup = True

    if self.connector is not None:
        # ★ KV Connector 场景（PD 分离 / 远端 KV）：走增强路径
        (new_computed_blocks,
         num_new_local_computed_tokens,
         request.shared_prefix_boundary,
         hit_diverged,
         ) = self._get_computed_blocks_for_connector(request)
        partial_group_hit_selected = hit_diverged
    else:
        # ★ 常规路径：本地 KV cache manager 查找已缓存 blocks
        computed_result = self.kv_cache_manager.get_computed_blocks(request)
        (new_computed_blocks,
         num_new_local_computed_tokens,
         request.shared_prefix_boundary,
         ) = cast(tuple[KVCacheBlocks, int, int], computed_result)
```

**`_get_computed_blocks_for_connector` 增强查找**：[L98-L134](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L98-L134)

```python
def _get_computed_blocks_for_connector(self, request: Request) -> tuple[KVCacheBlocks, int, int, bool]:
    kv_cache_manager = self.kv_cache_manager
    coordinator = kv_cache_manager.coordinator
    if (request.kv_transfer_params
            and request.kv_transfer_params.get("do_remote_prefill")
            and isinstance(coordinator, HybridKVCacheCoordinator)
            and getattr(coordinator, "full_attention_group_id", None) is not None):
        prefix_cache_lookup_enabled = kv_cache_manager.enable_caching and not request.skip_reading_prefix_cache
        if not prefix_cache_lookup_enabled:
            return kv_cache_manager.empty_kv_cache_blocks, 0, 0, False

        fa_group_id = coordinator.full_attention_group_id
        # ★ 按每个 group 独立查 cache hit 长度
        computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, request.num_tokens - 1,
        )
        # ★ 如果其他 group hit 都不超过 full_attention group：可接受
        if not any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):
            num_local = per_group_hits[fa_group_id]
            hit_diverged = min(per_group_hits) < num_local
            if hit_diverged:
                # ★ Partial-group cache hit：各 group hit 不一致
                blocks = kv_cache_manager.create_kv_cache_blocks(computed)
                logger.info("[RecomputeScheduler] Partial-group cache candidate ...")
                return blocks, num_local, 0, True  # 标记需要进一步结合远端 connector 判断

    # 常规回退：统一 get_computed_blocks
    blocks, num_local, shared_prefix_boundary = kv_cache_manager.get_computed_blocks(request)
    return blocks, num_local, shared_prefix_boundary, False
```

##### Step 2：外部 KV 缓存（KVConnector）查找 [L543-L577]

```python
# Get externally-cached tokens if using a KVConnector.
if self.connector is not None:
    # ★ 问远端 connector：此请求有多少 token 可以从远端加载
    ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )

    if ext_tokens is None:
        # Connector 无法确定 → 本轮不可调度，移到 skipped
        request_queue.pop_request()
        step_skipped_waiting.prepend_request(request)
        continue
    num_external_computed_tokens = ext_tokens

    # ★ Partial-group + ext_tokens=0 → 回退到普通 unified 查找
    if hit_diverged and num_external_computed_tokens == 0:
        (new_computed_blocks,
         num_new_local_computed_tokens,
         request.shared_prefix_boundary,
         ) = self.kv_cache_manager.get_computed_blocks(request)
        partial_group_hit_selected = False
```

##### Step 3：计算总 computed tokens，决定实际需计算 tokens [L579-L654]

```python
# Total computed tokens (local + external).
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
assert num_computed_tokens <= request.num_tokens

# ...

if load_kv_async:
    # ★ 异步加载远端 KV：本步不做 forward，num_new_tokens=0
    assert num_external_computed_tokens > 0
    num_new_tokens = 0
elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
    # DP 多机 prefill 负载均衡：非 cadence step 推迟 prefill
    break
else:
    # ★ 实际需要计算的 token 数 = 总 token 数 - 已缓存 token 数
    num_new_tokens = request.num_tokens - num_computed_tokens

    # ★ 投机解码 decode pad：统一 spec_tokens 尺寸，保持 CUDAGraph/ACLGraph
    if ((self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
            and self.num_sampled_tokens_per_step > 0
            and num_new_tokens == 1
            and (scheduled_running_reqs and not prefill_scheduled)):
        num_new_tokens = 1 + self.num_spec_tokens

    # 长 prefill 阈值截断 → chunked prefill
    threshold = self.scheduler_config.long_prefill_token_threshold
    if 0 < threshold < num_new_tokens:
        num_new_tokens = threshold

    num_new_tokens = min(num_new_tokens, token_budget)
```

**举个直观例子**：
- 总 prompt 长度：1000 tokens
- 本地 prefix cache 命中：600 tokens
- 远端 connector 命中：100 tokens
- **实际需要本轮 prefill 计算的只有：300 tokens！**

##### Step 4：allocate_slots 最终分配 blocks [L704-L716]

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,                              # 需要计算的新 token 数
    num_new_computed_tokens=num_new_local_computed_tokens,  # 本地已命中数
    new_computed_blocks=new_computed_blocks,      # ★ 命中的 cached blocks 列表
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,  # 远端命中数
    delay_cache_blocks=load_kv_async,
    num_encoder_tokens=num_encoder_tokens,
    full_sequence_must_fit=self.scheduler_reserve_full_isl,
    reserved_blocks=reserved_blocks,
    has_scheduled_reqs=bool(self.running),
)

if new_blocks is None:
    # block 不够 → 这个请求不能加入，下轮再见
    if request.has_encoder_inputs:
        self.encoder_cache_manager.free(request)
    break
```

##### Step 5：异步 KV 加载处理 [L759-L787]

```python
request = request_queue.pop_request()
if load_kv_async:
    # ★ 异步队列机制：请求状态变为 WAITING_FOR_REMOTE_KVS
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    step_skipped_waiting.prepend_request(request)

    # 先写入预期的 num_computed_tokens（实际加载完再修正）
    request.num_computed_tokens = num_computed_tokens
    self._inflight_prefills.add(request)

    if self.needs_kv_cache_zeroing:
        # 这部分 block 是 connector 接收用，无需清零
        self._skip_zero_block_ids.update(
            self.kv_cache_manager.get_zeroing_block_ids_in_range(...)
        )
    continue  # 不加入 running，等 KV 传输完成后再下一轮
```

##### Step 6：正常加入 running 队列 [L789-L826]

```python
self.running.append(request)
request.status = RequestStatus.RUNNING

# ★ 关键：更新 num_computed_tokens。下轮调度只会从这之后计算
request.num_computed_tokens = num_computed_tokens

# 只有还有剩余 prefill 工作的请求才算 inflight prefill
if num_computed_tokens + num_new_tokens < request.num_tokens:
    self._inflight_prefills.add(request)
```

---

## 第4层：KVCacheCoordinator / Manager 深度解析

### 整体结构

```
KVCacheManager (vllm.v1.core.kv_cache_manager)
    └── coordinator: AscendHybridKVCacheCoordinator  (patched)
            ├── block_pool: BlockPool  (所有物理 blocks + hash→block 全局索引)
            └── single_type_managers: tuple
                    ├── CompressAttentionManager  (MLA with compress_ratio)
                    ├── FullAttentionManager
                    └── SlidingWindowManager
```

### Patch 注入点：替换 Coordinator 构造

**文件**：[patch_kv_cache_coordinator.py#L390-L467](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L390-L467)

```python
def get_kv_cache_coordinator(kv_cache_config, max_model_len, ...):
    ...
    if _is_deepseek_v4_kv_cache_config(kv_cache_config):
        # ★ DeepSeek-V4 等混合模型：用 Ascend 专属 Hybrid Coordinator
        return AscendHybridKVCacheCoordinator(
            kv_cache_config, max_model_len, use_eagle, enable_caching,
            ...
        )
    ...

# ★ Monkey patch：替换上游 vllm 的函数引用
vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator

# 同时修补 engine/core.py 里的直接 import 引用
_kv_cache_manager = sys.modules.get("vllm.v1.core.kv_cache_manager")
if _kv_cache_manager is not None:
    _kv_cache_manager.get_kv_cache_coordinator = get_kv_cache_coordinator
```

---

### Coordinator 初始化

**文件**：[patch_kv_cache_coordinator.py#L67-L177](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L67-L177)

```python
class AscendHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    def __init__(self, kv_cache_config, max_model_len, use_eagle, enable_caching,
                 enable_kv_cache_events, dcp_world_size, pcp_world_size,
                 hash_block_size, eagle_attn_layer_names=None,
                 metrics_collector=None, max_in_flight_tokens=None,
                 max_num_batched_tokens=None, scheduler_block_size=None):
        del pcp_world_size  # Ascend 平台不支持 PCP
        self.dcp_world_size = dcp_world_size
        self.scheduler_block_size = scheduler_block_size

        # 1. 创建全局 BlockPool：所有 group 共享同一个 block hash 索引
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,  # 哈希计算粒度
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )

        # 2. 为每个 KV cache group 独立创建 manager（共享 block_pool）
        extra_mgr_kwargs = {"scheduler_block_size": scheduler_block_size}
        extra_mgr_kwargs["needs_kv_cache_zeroing"] = kv_cache_config.needs_kv_cache_zeroing
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec,
                block_pool=self.block_pool,  # ★ 共享 pool
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                max_in_flight_tokens=token_budget,
                max_model_len=max_model_len,
                **extra_mgr_kwargs,
            )
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
        )

        # 3. 验证 hash / block size 对齐
        self.hash_block_size = hash_block_size
        if enable_caching:
            assert all(
                self._get_effective_block_size(g.kv_cache_spec) % hash_block_size == 0
                for g in kv_cache_config.kv_cache_groups
            )
```

**Block Size 对齐计算** `_get_effective_block_size` [L185-L195](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L185-L195)：

```python
def _get_effective_block_size(self, kv_cache_spec: KVCacheSpec) -> int:
    block_size = kv_cache_spec.block_size
    if isinstance(kv_cache_spec, MambaSpec) and self.enable_caching:
        return block_size
    if self.dcp_world_size > 1:
        block_size *= self.dcp_world_size          # ★ DCP 场景：× context parallel 数
    if hasattr(kv_cache_spec, "compress_ratio"):
        compress_ratio = kv_cache_spec.compress_ratio or 1
        block_size *= compress_ratio                # ★ MLA 压缩：× compress_ratio
    return block_size
```

---

### ★ find_longest_cache_hit：Prefix Cache 匹配算法（Hybrid Coordinator 层）

**文件**：[patch_kv_cache_coordinator.py#L266-L387](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L266-L387)

```python
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    """
    不动点迭代算法：
    每个 attention type 要么接受当前候选长度，要么缩短它。
    只要任何 type 缩短了长度，就重新从头检查所有 type。
    单调下降 + 下界 0 → 必然收敛。
    """

    hit_length = max_cache_hit_length
    longest_hit_length = 0
    hit_blocks_by_group = [None] * num_groups
    hit_length_by_group = [0] * num_groups

    is_simple_hybrid = (len(self.attention_groups) == 2
                        and isinstance(self.attention_groups[0].spec, FullAttentionSpec))

    eagle_verified: set[int] = set()

    while True:
        curr_hit_length = hit_length
        for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(self.attention_groups):
            group_block_size = self._get_effective_block_size(spec)

            if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                # ★ Full attention 是向下闭包的：第一次查完后，后续迭代只需截断
                curr_hit_length = min(curr_hit_length, hit_length_by_group[first_group_id])
                continue

            drop_eagle_block = use_eagle and idx not in eagle_verified

            _max_length = curr_hit_length
            if drop_eagle_block and not isinstance(spec, MambaSpec):
                # ★ EAGLE 需要多匹配一个 block 然后 pop 掉最后一个
                eagle_margin = group_block_size  # 或精细 hash 的 hash_block_size
                _max_length = min(curr_hit_length + eagle_margin, max_cache_hit_length)

            # ★ 调用具体 manager 的 find_longest_cache_hit
            hit_result = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=_max_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=drop_eagle_block,
                alignment_tokens=self._cache_hit_alignment_tokens,
                dcp_world_size=self.dcp_world_size,
                pcp_world_size=1,
            )
            hit_blocks, _new_hit_length = hit_result
            if drop_eagle_block:
                eagle_verified.add(idx)
            elif _new_hit_length < curr_hit_length:
                eagle_verified.clear()  # 长度变了 → eagle drop 验证作废
            curr_hit_length = _new_hit_length

            # 记录 per-group 结果
            for group_id, blocks in zip(group_ids, hit_blocks):
                hit_blocks_by_group[group_id] = blocks
                hit_length_by_group[group_id] = _new_hit_length

            longest_hit_length = max(longest_hit_length, curr_hit_length)

        if curr_hit_length >= hit_length:   # 收敛
            break
        hit_length = curr_hit_length
        if is_simple_hybrid:
            break  # 简单双组 hybrid：一次迭代就够

    # 最后：按最终 hit_length 截断所有 FullAttention group
    for group in self.attention_groups:
        if not isinstance(group.spec, FullAttentionSpec):
            continue
        num_blocks = cdiv(hit_length, self._get_effective_block_size(group.spec))
        for group_id in group.group_ids:
            if (blks := hit_blocks_by_group[group_id]) is not None:
                del blks[num_blocks:]
                hit_length_by_group[group_id] = hit_length

    cache_hit_blocks = tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group)
    return cache_hit_blocks, hit_length, longest_hit_length - hit_length
```

---

### ★ CompressAttentionManager.find_longest_cache_hit：压缩 MLA 的匹配逻辑

**文件**：[single_type_kv_cache_manager.py#L220-L270](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L220-L270)

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                           block_pool, kv_cache_spec, alignment_tokens,
                           dcp_world_size=1, pcp_world_size=1,
                           drop_eagle_block=False):
    del pcp_world_size
    eagle_drop = drop_eagle_block

    computed_blocks: tuple[list[KVCacheBlock], ...] = tuple([] for _ in range(len(kv_cache_group_ids)))
    block_size = kv_cache_spec.block_size
    if dcp_world_size > 1:
        block_size *= dcp_world_size
    # ★ 逻辑块大小 = 物理块大小 × compress_ratio
    #   例：compress_ratio=4, block_size=128 → logical_block_size=512
    logical_block_size = block_size * kv_cache_spec.compress_ratio
    hash_block_size = block_pool.hash_block_size

    # ★ BlockHashListWithBlockSize：把物理级 (hash_block_size) 的 hashes
    #   聚合封装为按 logical_block_size 的迭代器
    logical_block_hashes = BlockHashListWithBlockSize(
        block_hashes, hash_block_size, logical_block_size
    )
    max_num_blocks = max_length // logical_block_size

    # ★ 链式查找
    for block_hash in itertools.islice(logical_block_hashes, max_num_blocks):
        # block_hashes 是链。一旦一个 hash 不在 cached_block_hash_to_id 中，
        # 后面的 hash 必然也没算过 → 直接 break
        if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)  # 命中！加入结果
        else:
            break  # ★ 链式哈希特性保证

    # ★ EAGLE/MTP 投机解码：最后一个 matched block 丢弃
    #   (因为 draft token 的 K/V 会被验证/覆盖，不保证稳定)
    if eagle_drop and computed_blocks[0]:
        for computed in computed_blocks:
            computed.pop()

    # ★ Alignment 对齐：确保命中长度是 alignment_tokens 的整数倍
    #   这是 hybrid coordinator 跨组协调的关键（不同组 block size 必须对齐）
    while (logical_block_size != alignment_tokens
           and len(computed_blocks[0]) * logical_block_size % alignment_tokens != 0):
        for computed in computed_blocks:
            computed.pop()

    hit_length = len(computed_blocks[0]) * logical_block_size
    return computed_blocks, hit_length
```

### 为什么压缩 cache 要用逻辑块？

**测试验证**：[test_compressed_prefix_cache.py#L73-L112](file:///workspace/tests/ut/test_compressed_prefix_cache.py#L73-L112)

```python
def test_compressed_prefix_cache_uses_logical_block_hash():
    block_size = 128
    compress_ratio = 4
    logical_block_size = block_size * compress_ratio   # = 512

    # Request B 在第 1 个物理块后改了 1 个 token (第 135 号)
    # 位置 135 ∈ 物理块 1 (128~255)，但仍属于 逻辑块 0 (0~511)
    request_b_tokens = request_a_tokens.copy()
    request_b_tokens[block_size + 7] = 999_999

    # ★ 逻辑块哈希结果不同 → 请求 B 逻辑块 0 命中 = 0
    #   这是正确的：因为压缩 MLA 的 KV 以逻辑块为单位存储，不能部分共享
    hit_result = CompressAttentionManager.find_longest_cache_hit(
        block_hashes=request_b.block_hashes, max_length=logical_block_size, ...
    )
    hit_blocks = hit_result[0][0]
    assert hit_blocks == []   # ✅ 正确：没有误命中
```

---

### allocate_new_computed_blocks：把命中的 cached blocks 分配给请求

**文件**：[single_type_kv_cache_manager.py#L65-L134](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L65-L134)

```python
def allocate_new_computed_blocks(self, request_id, new_computed_blocks,
                                  num_local_computed_tokens, num_external_computed_tokens):
    # Fast-path：running 中的请求不会再有 prefix cache hit
    if request_id in self.num_cached_block:
        assert len(new_computed_blocks) == 0
        return

    req_blocks = self.req_to_blocks[request_id]
    # 压缩 MLA：token 数也要除以 compress_ratio
    num_total_computed_tokens = (num_local_computed_tokens + num_external_computed_tokens) // self.compress_ratio

    # ★ Sliding Window：前面不可达的 blocks 用 null_block 占位
    num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
    num_skipped_blocks = num_skipped_tokens // self.block_size
    if num_skipped_blocks > 0:
        new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
        num_external_computed_tokens = min(
            num_total_computed_tokens - num_skipped_tokens,
            num_external_computed_tokens,
        )

    # ★ Touch：刷新 LRU 时间戳，防止这些 blocks 被 evict
    if self.enable_caching:
        self.block_pool.touch(new_computed_blocks)

    # null 占位 blocks + 命中的 computed blocks
    req_blocks.extend([self._null_block] * num_skipped_blocks)
    req_blocks.extend(new_computed_blocks)

    # ★ 关键：记录这些 blocks 已在 cache 中，cache_blocks() 不会重复写 hash
    self.num_cached_block[request_id] = len(req_blocks)

    # ★ 远端 KV：为外部 computed tokens 分配本地物理块
    if num_external_computed_tokens > 0:
        allocated_blocks = self.block_pool.get_new_blocks(
            cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        req_blocks.extend(allocated_blocks)
```

---

### cache_blocks：模型计算后，将 blocks 写入 Prefix Cache 全局索引

**文件**：[single_type_kv_cache_manager.py#L183-L217](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L183-L217)

```python
def cache_blocks(self, request, num_tokens, retention_interval=None, *, alignment_tokens=None):
    num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
    # ★ 压缩 MLA：完整逻辑块数 = tokens // (block_size × compress_ratio)
    #   只缓存完整的逻辑块！未满的不写入 pool 共享索引
    num_full_blocks = num_tokens // (self.block_size * self.compress_ratio)

    if num_cached_blocks >= num_full_blocks:
        return   # 之前已缓存 → 跳过

    # ★ 把 blocks 注册到 BlockPool 的全局 hash 索引
    #   之后其他请求通过 find_longest_cache_hit() 就能查到这些 block
    self.block_pool.cache_full_blocks(
        request=request,
        blocks=self.req_to_blocks[request.request_id],
        num_cached_blocks=num_cached_blocks,
        num_full_blocks=num_full_blocks,
        block_size=self.block_size * self.compress_ratio,  # ★ 用逻辑块大小
        kv_cache_group_id=self.kv_cache_group_id,
    )
    self.num_cached_block[request.request_id] = num_full_blocks
```

**写入时机**：在 `Scheduler.update_from_output()` 中调用，即模型 forward 完成、KV 已写入物理 cache 之后。

---

## 第5层：BlockTable → Slot Mapping（KV Cache 物理寻址）

### AscendBlockTables.compute_slot_mappings

**文件**：[worker/v2/block_table.py#L65-L93](file:///workspace/vllm_ascend/worker/v2/block_table.py#L65-L93)

```python
def compute_slot_mappings(self, idx_mapping, query_start_loc, positions,
                          num_tokens_padded, out=None):
    # 每个 token 的 (req_idx, position) → KV cache 中的 slot_id
    # 使用 Triton kernel 在 NPU 上并行计算
    _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
        slot_mappings.shape[1],
        idx_mapping, query_start_loc, positions,
        self.block_table_ptrs, self.block_table_strides,
        self.block_sizes_tensor,
        slot_mappings, slot_mappings.stride(0),
        self.cp_rank,
        CP_SIZE=self.cp_size,
        CP_INTERLEAVE=self.cp_interleave,
        PAD_ID=PAD_SLOT_ID,
        TRITON_BLOCK_SIZE=1024,
        TOTAL_BLOCK_SIZE=4096,
    )
    return slot_mappings[:, :num_tokens_padded]
```

### Triton Kernel 逐行解释

**文件**：[worker/v2/block_table.py#L96-L166](file:///workspace/vllm_ascend/worker/v2/block_table.py#L96-L166)

```python
@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens, idx_mapping, query_start_loc, pos,
    block_table_ptrs, block_table_strides, block_sizes,
    slot_mappings_ptr, slot_mappings_stride,
    cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_ID,
    TRITON_BLOCK_SIZE, TOTAL_BLOCK_SIZE,
):
    group_id = tl.program_id(0)       # 第几个 KV cache group (MLA/SWA/Full)
    batch_idx = tl.program_id(1)      # 第几个 request

    # ★ 加载指针：该 group 的 block_table 基地址
    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)

    # 当前 request 在 req_states 中的索引
    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)

    # 对 request 中的每个 token 计算 slot_id
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)
        positions = positions.to(tl.int32)

        # ★ Step 1：position → 第几个逻辑 block + block 内 offset
        #   block_indices = position // (block_size * CP_SIZE)
        #   block_offsets = position  % (block_size * CP_SIZE)  改用 sub+mul
        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions - (block_size * CP_SIZE) * block_indices

        # ★ Step 2：从 block_table 读取物理 block_number
        #   block_table[req_state_idx, block_indices] → 物理块号
        #   Ascend NPU 优化：整块加载后 tl.gather（避免非连续标量退化）
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride
                          + tl.arange(0, TOTAL_BLOCK_SIZE)
        )
        block_numbers = block_numbers.to(tl.float32)
        block_numbers = tl.gather(block_numbers, block_indices, 0)

        # ★ Step 3：slot_id = block_number * block_size + block_offset
        if CP_SIZE == 1:
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context Parallel：只保留 cp_rank 负责的 token，其他 PAD
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
```

**slot_mappings 的两大用途**：
1. **Attention 计算**：`K_tensor[slot_mapping[i]]` 读取第 i 个 token 的 K cache
2. **KV 写入**：`reshape_and_cache` kernel 将新计算的 K/V 写入 `slot_mapping[i]` 指定的物理地址

---

## 第6层：异步队列机制详解

### 异步队列层级总览

```
┌──────────────────────────────────────────────────────────────────┐
│ API Server 进程 (asyncio 事件循环)                                │
│                                                                  │
│  ┌─ asyncio.Queue (per-request streaming)                        │
│  │   异步生成器：_stream_request_chat 每次拿到新 token 就 SSE 推送 │
│  │                                                               │
│  └─► add_request() → 写入 EngineCore 进程的 input_queue          │
│            (multiprocessing.Queue / ZMQ / IPC pipe)              │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼ 跨进程通信
┌──────────────────────────────────────────────────────────────────┐
│ EngineCore 进程 (独占 GPU/NPU，常驻循环)                          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Scheduler 内部队列（纯 Python list / deque）                │  │
│  │                                                            │  │
│  │  waiting            ← 新请求 + 被抢占后重入的请求          │  │
│  │  skipped_waiting    ← 本轮无法调度、下次优先重试的请求      │  │
│  │  running            ← 正在进行 prefill/decode 的请求       │  │
│  │  _inflight_prefills ← 还在 chunked prefill 中的请求集      │  │
│  │                                                            │  │
│  │  WAITING_FOR_REMOTE_KVS 特殊状态：                         │  │
│  │    connector 异步接收远端 KV，完成后自动解除阻塞            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  EngineCore.output_queue (multiprocessing)                       │
│    → API Server 消费：EngineCoreOutputs                          │
│      → 每个 client_index 对应一个 HTTP 连接的输出流              │
└──────────────────────────────────────────────────────────────────┘
```

### Waiting 请求阻塞状态流转

**文件**：[recompute_scheduler.py#L488-L499](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L488-L499)

```python
# try to promote blocked statuses while traversing skipped queue.
if self._is_blocked_waiting_status(request.status) and not self._try_promote_blocked_waiting_request(request):
    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        logger.debug("[RecomputeScheduler] %s is still in WAITING_FOR_REMOTE_KVS state.", request_id)
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

可能的阻塞 status：
- `WAITING_FOR_REMOTE_KVS`：远端 KV 传输中
- `WAITING_FOR_STREAMING_REQ`：多轮对话等待用户输入
- `WAITING_FOR_MM_ENCODER`：多模态编码器预取中

### KV 异步加载完成回调：`_update_waiting_for_remote_kv`

**文件**：[recompute_scheduler.py#L136-L161](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L136-L161)

```python
def _update_waiting_for_remote_kv(self, request: Request) -> None:
    """
    KV Connector 异步接收结束后调用。
    finished_recving_kv_req_ids / failed_recving_kv_req_ids 由
    worker 端 connector 在 update_from_output 的上一步报告。
    """
    assert self.connector is not None

    if request.request_id in self.failed_recving_kv_req_ids:
        # ★ 传输失败：只缓存已经接收成功的部分
        if request.num_computed_tokens:
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            ...
        else:
            self.kv_cache_manager.free(request)
        self.failed_recving_kv_req_ids.remove(request.request_id)
    else:
        # ★ 传输成功：修正 num_computed_tokens 并写入 cache
        num_computed_tokens = min(request.num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1   # 保留最后 1 个 token 给本地 prefill 预测
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
        request.num_computed_tokens = num_computed_tokens

    self.finished_recving_kv_req_ids.remove(request.request_id)
    # → 下一轮 schedule() 时 _try_promote_blocked_waiting_request 会解除阻塞
```

---

## 第7层：Prefill / Decode 计算 + KV 写入

### ModelRunner.execute_model() 流程

**文件**：[worker/v2/model_runner.py#L212-L250](file:///workspace/vllm_ascend/worker/v2/model_runner.py#L212-L250)（V2） / `worker/model_runner_v1.py`（V1）

```python
@torch.inference_mode()
def execute_model(self, scheduler_output, intermediate_tensors=None,
                  dummy_run=False, skip_attn_for_dummy_run=False, is_profile=False):
    with flashcomm_dispatch_wrapper(self.vllm_config):
        output = super().execute_model(scheduler_output, ...)
```

上游 GPUModelRunner 逻辑（Ascend 继承后扩展）：

```python
def execute_model(scheduler_output):
    # ── Step 1: prepare_inputs() 构造 input_batch ──
    #   a. 把 new_request_data 的 token_ids / positions 异步拷到 device (H2D)
    #   b. 更新 req_states：num_tokens, num_computed_tokens, block_table
    #   c. ★ compute_slot_mappings()：生成 [num_groups, num_tokens] 的 slot_ids
    #   d. 构造 AttentionMetadata：
    #        - prefill：query_start_loc, seq_lens (变长 dense attention)
    #        - decode：block_table + slot_mapping (paged attention)
    #   e. 构造 SamplingMetadata：temperature, top_p, logit_bias 等

    # ── Step 2: 可能的 CUDAGraph/ACLGraph 捕获 + dispatch ──
    #   uniform decode batch → 走 graph replay（超快）
    #   非 uniform / prefill 混合 → 走 eager mode

    # ── Step 3: set_ascend_forward_context ──
    #   (Ascend 特有) MoE 通信方式选择、MC2 tokens capacity、cos/sin 缓存设置、
    #   EPLB expert 加载、结构化输出语法 mask 等

    # ── Step 4: model.forward() 逐层计算 ──
    #   每个 Attention 层行为：
    #     Prefill:
    #       Q = X @ Wq, K = X @ Wk, V = X @ Wv  (num_new_tokens × d_k)
    #       Attn = Q @ K.T / sqrt(d_k) + AttnMask  (dense)
    #       O = Attn @ V
    #       ★ KV 写入：通过 slot_mapping[i] → 写 KV_tensor[slot, ...]
    #
    #     Decode:
    #       Q_new = X_new @ Wq  (1 token per req)
    #       ★ 读历史 K/V：KV_tensor[slot_mapping[:-1]] → K_hist, V_hist
    #       Attn = Q_new @ K_hist.T / sqrt(d_k)  (GEMV / FlashAttention)
    #       O_new = Attn @ V_hist
    #       ★ 写新 token K/V：KV_tensor[slot_mapping[-1]] ← K_new, V_new
    #
    #   MLP / MoE：正常前向 + All-to-All 路由通信（MoE）

    # ── Step 5: Sampler 采样 ──
    #   logits → 温度缩放 / top-p / top-k → multinomial → new_token_ids
    #   spec decode：rejection sampling 验证 draft tokens

    return ModelRunnerOutput(
        sampled_token_ids=...,
        logprobs=...,
        prompt_logprobs_dict=...,
        routed_experts=...,          # MoE routing 结果 (D2H)
        kv_connector_output=...,     # KV connector 传输结果
        ...
    )
```

### Prefill vs Decode 差异对比表

| 维度 | Prefill 阶段 | Decode 阶段 |
|------|-------------|-------------|
| Q 长度 | `num_new_tokens`（可达数千） | 每个请求 1 token |
| KV 来源 | 本次计算产生的 K/V | 从 KV cache **读**所有历史 + 本次的 1 个 |
| Attention 模式 | Dense Masked Attention | Paged FlashAttention (GEMV 风格) |
| slot_mapping 作用 | **写入** 新 K/V 到对应 slot | **读取** 历史 K/V + 写入新 1 token 的 K/V |
| 计算瓶颈 | 算力（O(n²) 注意力） | 显存带宽（KV cache 随机读） |
| Prefix Cache 影响 | **大幅减少 Q/K/V 计算量** | 无直接影响（decode 都是 1 token） |

---

## 第8层：Scheduler.update_from_output → 请求返回

**核心函数**：[recompute_scheduler.py#L967+](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L967-L999)

```python
def update_from_output(self, scheduler_output, model_runner_output):
    sampled_token_ids = model_runner_output.sampled_token_ids
    logprobs = model_runner_output.logprobs
    ...
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

    # ── recompute 请求：先把重算通知发出去 ──
    self._add_recomputed_outputs(scheduler_output, outputs)

    # ── 处理 KV connector 失败的 block：调整 num_computed_tokens ──
    if kv_connector_output and kv_connector_output.invalid_block_ids:
        failed_kv_load_req_ids = self._handle_invalid_blocks(
            kv_connector_output.invalid_block_ids, num_scheduled_tokens
        )

    # ── 每个请求：更新状态、追加 token、写 cache、判断 finish ──
    for req in scheduled_requests:
        request_id = req.request_id
        new_token_count = num_scheduled_tokens.get(request_id, 0)
        if new_token_count == 0:
            continue

        # 1. D2H：sampled_token_ids 搬运到 CPU
        sample_slice = sampled_token_ids[start:end]
        new_token_ids = sample_slice.to(torch.int32).tolist()

        # 2. 追加到 request.output_token_ids（保留完整输出历史）
        output_token_ids.extend(new_token_ids)

        # 3. 处理 spec decode：discard 被拒的 draft tokens
        if request.num_output_placeholders > 0:
            ...  # 截断 placeholder，保留实际被接受的 token

        # 4. ★ 更新 num_computed_tokens
        request.num_computed_tokens += num_new_accept_tokens

        # 5. ★★★ 写入 Prefix Cache ★★★
        #    把当前请求填满的 blocks 注册到 BlockPool 全局索引
        #    → 后续请求就能命中这段前缀！
        if request.status == RequestStatus.RUNNING and kv_cache_write_eligible:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens,
                alignment_tokens=alignment,
            )

        # 6. 判断是否 finish
        stop_reason = None
        finish_reason = None
        if eos_token_id in new_token_ids or len(pattern_matchers) > 0 matched:
            finish_reason = FinishReason.STOP
            stop_reason = "eos" / "stop"
        elif len(prompt) + len(output) >= max_tokens:
            finish_reason = FinishReason.LENGTH
        elif request.num_computed_tokens >= self.max_model_len:
            finish_reason = FinishReason.LENGTH

        if finish_reason is not None:
            # ★ 最终完成：生成 EngineCoreOutput 推给客户端
            outputs[request.client_index].append(EngineCoreOutput(
                request_id=request_id,
                new_token_ids=new_token_ids,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                ...
            ))
            # ★ 释放 KV blocks → 归还 BlockPool
            self.finish_requests(request_id, RequestStatus.FINISHED_STOPPED)
        else:
            # 流式模式下：每步都推增量 token
            if streaming:
                outputs[request.client_index].append(EngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=delta_token_ids,
                    finish_reason=None,
                    ...
                ))

    return outputs
```

### EngineCoreOutput → API Server → 客户端

```
EngineCore 进程
  │ output_queue.put(outputs)
  ▼
API Server 进程
  │ async 消费者从 input_queue / output_queue 读写
  │
  ├─ Non-streaming:
  │     攒到 finish_reason != None，组装 HTTP JSON 响应返回
  │
  └─ Streaming (SSE / text/event-stream):
        每个增量 EngineCoreOutput → 通过 async generator 推送：
          data: {"choices":[{"delta":{"content":"你"}, ...}]}
          data: {"choices":[{"delta":{"content":"好"}, ...}]}
          ...
          data: [DONE]
```

---

## 关键文件 / 函数 / 代码行索引表

| 层级 | 文件 | 关键函数 / 代码段 | 行号 |
|------|------|-------------------|------|
| **Plugin Entry** | [`__init__.py`](file:///workspace/vllm_ascend/__init__.py) | `register()`, `register_connector()`, `_ensure_global_patch()` | [L73](file:///workspace/vllm_ascend/__init__.py#L73-L116) |
| **Platform Patch** | [`patch_kv_cache_coordinator.py`](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py) | `AscendHybridKVCacheCoordinator.__init__` | [L67](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L67-L177) |
| | | `_get_effective_block_size()` | [L185](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L185-L195) |
| | | `find_longest_cache_hit()` (hybrid 不动点迭代) | [L266](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L266-L387) |
| | | `get_kv_cache_coordinator()` 工厂 + patch | [L390](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L390-L467) |
| **Platform Patch** | [`patch_kv_cache_utils.py`](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py) | `_ascend_resolve_kv_cache_block_sizes()` | [L23](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L23-L56) |
| **Scheduler** | [`recompute_scheduler.py`](file:///workspace/vllm_ascend/core/recompute_scheduler.py) | `RecomputeSchedulerConfig.initialize_from_config()` | [L54](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L54-L73) |
| | | `_get_computed_blocks_for_connector()` | [L98](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L98-L134) |
| | | `schedule()` 主流程 | [L182](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L182-L939) |
| | | running 队列 allocate_slots + 抢占循环 | [L315](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L315-L417) |
| | | waiting 队列 prefix cache lookup | [L524](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L524-L580) |
| | | waiting 队列 allocate_slots + 异步 KV 加载分支 | [L704](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L704-L787) |
| | | `_update_waiting_for_remote_kv()` 回调 | [L136](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L136-L161) |
| | | `update_from_output()` + cache_blocks 写入点 | [L967](file:///workspace/vllm_ascend/core/recompute_scheduler.py#L967-L1019) |
| **KV Cache Manager** | [`single_type_kv_cache_manager.py`](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py) | `CompressAttentionManager` 类定义 | [L30](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L30-L35) |
| | | `get_num_blocks_to_allocate()` | [L36](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L36-L63) |
| | | `allocate_new_computed_blocks()` | [L65](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L65-L134) |
| | | `allocate_new_blocks()` | [L135](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L135-L161) |
| | | `cache_blocks()` 写入 BlockPool | [L183](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L183-L217) |
| | | `find_longest_cache_hit()` （压缩版链式查找） | [L220](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L220-L270) |
| | | `get_manager_for_kv_cache_spec()` 工厂 | [L273](file:///workspace/vllm_ascend/core/single_type_kv_cache_manager.py#L273-L326) |
| **KV Cache Spec** | [`kv_cache_interface.py`](file:///workspace/vllm_ascend/core/kv_cache_interface.py) | `AscendMLAAttentionSpec` (压缩 MLA) | [L18](file:///workspace/vllm_ascend/core/kv_cache_interface.py#L18-L93) |
| | | `AscendSFAIndexerCacheSpec` (SFA indexer) | [L96](file:///workspace/vllm_ascend/core/kv_cache_interface.py#L96-L158) |
| | | `register_ascend_kv_cache_specs()` | [L213](file:///workspace/vllm_ascend/core/kv_cache_interface.py#L213-L228) |
| **BlockTable** | [`worker/v2/block_table.py`](file:///workspace/vllm_ascend/worker/v2/block_table.py) | `AscendBlockTables.compute_slot_mappings()` | [L65](file:///workspace/vllm_ascend/worker/v2/block_table.py#L65-L93) |
| | | Triton `_compute_slot_mappings_kernel` | [L96](file:///workspace/vllm_ascend/worker/v2/block_table.py#L96-L166) |
| **ModelRunner** | [`worker/v2/model_runner.py`](file:///workspace/vllm_ascend/worker/v2/model_runner.py) | `NPUModelRunner.__init__()` | [L120](file:///workspace/vllm_ascend/worker/v2/model_runner.py#L120-L197) |
| | | `execute_model()` (包含 flashcomm 包装) | [L212](file:///workspace/vllm_ascend/worker/v2/model_runner.py#L212-L250) |
| **Tests 参考** | [`test_compressed_prefix_cache.py`](file:///workspace/tests/ut/test_compressed_prefix_cache.py) | 逻辑块哈希正确性验证 | [L73](file:///workspace/tests/ut/test_compressed_prefix_cache.py#L73-L112) |
| | | 相同逻辑块完整命中验证 | [L115](file:///workspace/tests/ut/test_compressed_prefix_cache.py#L115-L141) |
| | | Hybrid coordinator 拒绝部分逻辑块命中 | [L143](file:///workspace/tests/ut/test_compressed_prefix_cache.py#L143-L210) |

---

## 总结：Prefix Cache 全流程一图流

```
 新请求到达
    │
    ▼
 生成 Request.block_hashes
   (链式 SHA256, 按 hash_block_size 分块)
    │
    ▼
 Scheduler.schedule()
   处理 waiting 队列
    │
    ├─ request.num_computed_tokens == 0 ?
    │     │
    │     ▼ YES
    │   coordinator.find_longest_cache_hit()
    │     │
    │     ├─► 按 attention group 迭代：
    │     │     manager.find_longest_cache_hit()
    │     │       │
    │     │       ├─ 构造 logical_block_hashes (×compress_ratio)
    │     │       ├─ 遍历 block_hash 链
    │     │       │     block_pool.get_cached_block(hash)
    │     │       │     命中 → append，继续
    │     │       │     未命中 → break ★ 链式哈希保证
    │     │       ├─ EAGLE drop last block
    │     │       └─ alignment_tokens 截断对齐
    │     │
    │     └─► 所有 group hit 取最小值 → 最终 hit_length
    │
    ├─ num_new_tokens = total - local_hit - external_hit
    │
    ├─ allocate_slots()
    │     ├─ allocate_new_computed_blocks (touch 命中块防 evict)
    │     └─ block_pool.get_new_blocks (新物理块)
    │
    ├─ 异步 KV 加载?
    │     YES → status=WAITING_FOR_REMOTE_KVS
    │           connector recv 完成 → _update_waiting_for_remote_kv
    │           cache_blocks() 写成功接收的部分
    │           解除阻塞，等待下轮 schedule
    │     NO  → 加入 running 队列
    │
    ▼
 request.num_computed_tokens 被更新为已算的总数
    │
    ▼
 Worker / ModelRunner.execute_model()
    │
    ├─► prepare_inputs()
    │     └─► compute_slot_mappings()
    │          position → block_index → block_number → slot_id
    │
    ├─► model.forward()
    │     ├─ Prefill: QK^T → O → 新 K/V 写入 slot_mapping[i]
    │     └─ Decode:  读 KV cache → QK^T → O → 写新 token 的 K/V
    │
    └─► Sampler → new_token_ids
    │
    ▼
 Scheduler.update_from_output()
    │
    ├─ request.num_computed_tokens += scheduled_tokens
    │
    ├─ kv_cache_manager.cache_blocks()
    │     ▼
    │   block_pool.cache_full_blocks()
    │     → 注册 block_hash → KVCacheBlock 到全局索引
    │     → ✅ 后续新请求即可命中这段 prefix！
    │
    ├─ append new_token_ids 到输出
    │
    └─ EOS / max_tokens?
          YES → finish_requests() → free blocks → 推最终响应
          NO  → 留在 running 队列，下轮 decode 1 token
                 流式 → 推增量 EngineCoreOutput
```

**Prefix Cache 的核心本质**：通过**链式 block hash** 作为 key，以 BlockPool 为全局共享索引，把不同请求的**相同前缀**映射到同一份物理 KV blocks，从而跳过重复的 prefill 计算。Ascend 在标准 vLLM 基础上额外做了：
1. **混合 KV group 协调器**（DeepSeek-V4 多规格 MLA/SWA 组间不动点迭代对齐）
2. **压缩 MLA 逻辑块哈希**（×compress_ratio 避免物理块级别的误命中）
3. **PD 分离 connector 增强路径**（本地 + 远端双重 prefix cache 查找）
4. **NPU Triton slot_mapping kernel**（针对 NPU 的非连续访存 + int32 兼容优化）

至此，从 HTTP 请求入口到最终响应返回的完整 Prefix Cache 特性逻辑已全部覆盖。
