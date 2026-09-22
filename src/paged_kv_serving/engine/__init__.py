"""A from-scratch CPU inference engine: paged KV cache, continuous batching, admission control.

Reading order for the primary pattern:

- :mod:`blocks` -- the reference-counted block pool and copy-on-write;
- :mod:`prefix_cache` -- hash-chained reuse of full blocks, scoped per key;
- :mod:`kvcache` -- block tables, prefix reuse and eviction per sequence;
- :mod:`attention` -- attention over blocks with an online softmax;
- :mod:`runner` -- ragged batched forward passes against the pool;
- :mod:`scheduler` -- the three-phase token budget: decode, chunked prefill, admit;
- :mod:`admission` -- the EDF feasibility test behind every early 503;
- :mod:`engine` -- one ``step()`` tying them together.

:mod:`static` is the padded, fixed-batch baseline the benchmarks compare against, and
:mod:`equivalence` is the cache-free oracle both are checked against.
"""

from __future__ import annotations

__all__ = [
    "BlockManager",
    "Engine",
    "EngineConfig",
    "KVCacheManager",
    "ModelConfig",
    "PrefixCache",
    "Request",
    "SamplingParams",
    "Scheduler",
    "SloClass",
    "StaticBatchingEngine",
]

from paged_kv_serving.engine.blocks import BlockManager
from paged_kv_serving.engine.config import EngineConfig, ModelConfig, SloClass
from paged_kv_serving.engine.engine import Engine
from paged_kv_serving.engine.kvcache import KVCacheManager
from paged_kv_serving.engine.prefix_cache import PrefixCache
from paged_kv_serving.engine.scheduler import Scheduler
from paged_kv_serving.engine.sequence import Request, SamplingParams
from paged_kv_serving.engine.static import StaticBatchingEngine
