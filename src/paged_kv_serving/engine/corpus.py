"""A small original corpus, used only to train the demo tokenizer's merges.

Nothing is downloaded and nothing is copied from another project: these are a few paragraphs
written for this repository about the mechanics it implements. They give the byte-level BPE
trainer enough repeated substrings to learn useful merges, and they double as the prompt material
for the demo and the benchmarks, where a shared system prompt is what the prefix cache reuses.
"""

from __future__ import annotations

__all__ = ["DEMO_CORPUS", "SHARED_SYSTEM_PROMPT", "USER_TURNS"]

SHARED_SYSTEM_PROMPT = (
    "You are a serving engine assistant. Answer questions about batching, paged key value "
    "caches, scheduling and admission control. Keep answers short, concrete and honest about "
    "what the engine measures rather than what it hopes for. When a number is quoted, say which "
    "benchmark produced it and on which machine it ran.\n\n"
)

USER_TURNS: tuple[str, ...] = (
    "Why does static batching waste compute on padding?",
    "What does a block table buy over a contiguous key value cache?",
    "When is a sequence preempted and what happens to the tokens it already produced?",
    "How does chunked prefill stop a long prompt from blocking decode?",
    "Which metric tells me the key value pool is the bottleneck?",
    "What does a prefix cache hit actually save?",
    "Why reject a request early instead of queueing it?",
    "How is time to first token different from time per output token?",
    "What makes copy on write correct for shared prompt blocks?",
    "Why is the reference implementation slower than the paged one?",
)

DEMO_CORPUS: tuple[str, ...] = (
    SHARED_SYSTEM_PROMPT,
    *USER_TURNS,
    "Continuous batching re-decides the batch every iteration, so a finished sequence leaves "
    "immediately and a new one joins the next step without waiting for its neighbours.",
    "A paged key value cache stores a sequence in fixed size blocks listed by a block table, so "
    "memory never has to be contiguous and the only waste is the tail of the last block.",
    "Reference counted blocks let several sequences share the same prompt. A write to a shared "
    "block copies it first, so sharing is safe and the copy costs one block rather than a prompt.",
    "The scheduler fills a token budget with decode tokens first and then with prefill chunks, so "
    "a long prompt is spread over iterations instead of blocking every sequence that is streaming.",
    "Preemption frees the blocks of the lowest priority sequence and recomputes it later. The "
    "tokens it already produced are kept, so preemption costs work but never costs output.",
    "Admission control predicts whether a request can still finish before its deadline and returns "
    "a status of five hundred and three with a retry after header when it cannot.",
    "Time to first token measures queueing and prefill. Time per output token measures the steady "
    "state of decode. A server can be fast at one and slow at the other, so both are reported.",
    "An open loop load generator decides arrival times in advance and does not wait for responses, "
    "which is what keeps a slow server from quietly lowering its own offered load.",
    "Goodput counts only the requests that finished inside their deadline. Throughput that misses "
    "every deadline is not throughput anybody can use.",
    "The prefix cache keys each full block by a hash of the block and of every block in front of "
    "it, so a match implies the whole history matches and not merely the last few tokens.",
)
