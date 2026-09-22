"""Command line entry point: ``paged-kv-serving`` or ``python -m paged_kv_serving``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from paged_kv_serving.core.config import get_settings  # noqa: PLC0415
    from paged_kv_serving.core.logging import configure_logging  # noqa: PLC0415
    from paged_kv_serving.core.tracing import install_global_tracer_provider  # noqa: PLC0415
    from paged_kv_serving.engine.api import create_api  # noqa: PLC0415
    from paged_kv_serving.engine.threads import limit_blas_threads  # noqa: PLC0415
    from paged_kv_serving.engine.weights import load_checkpoint  # noqa: PLC0415

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    limit_blas_threads(settings.engine_blas_threads)
    checkpoint = load_checkpoint(
        settings.checkpoint_dir, expect_manifest_sha256=settings.checkpoint_manifest_sha256
    )
    app = create_api(checkpoint, settings)
    install_global_tracer_provider(app.state.tracer_provider)
    config = uvicorn.Config(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
        timeout_graceful_shutdown=30,
    )
    # SIGTERM and SIGINT are captured by uvicorn itself: it stops accepting connections, lets
    # in-flight ones finish (up to timeout_graceful_shutdown), then closes the lifespan -- and the
    # lifespan in create_api() drains the engine before stopping its loop.
    uvicorn.Server(config).run()
    return 0


def _build_checkpoint(directory: Path, seed: int) -> int:
    from paged_kv_serving.engine.weights import build_random_checkpoint  # noqa: PLC0415

    checkpoint = build_random_checkpoint(directory, seed=seed)
    print(
        json.dumps(
            {
                "directory": str(directory),
                "parameters": checkpoint.model.config.parameter_count,
                "vocab_size": checkpoint.tokenizer.vocab_size,
                "manifest_sha256": checkpoint.manifest.sha256(),
            },
            indent=2,
        )
    )
    return 0


def _demo(workdir: Path) -> int:
    from paged_kv_serving.core.logging import configure_logging  # noqa: PLC0415
    from paged_kv_serving.engine.demo import render_report, run_serving_demo  # noqa: PLC0415
    from paged_kv_serving.engine.threads import limit_blas_threads  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    limit_blas_threads(1)
    summary = run_serving_demo(workdir)
    target = workdir / "demo-summary.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(render_report(summary))
    print(f"\nfull summary: {target}")
    return 0


def _loadgen(args: argparse.Namespace) -> int:
    """Open-loop HTTP burst against a running server; keys come from the environment."""
    import asyncio  # noqa: PLC0415
    import os  # noqa: PLC0415
    import time  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    from paged_kv_serving.engine.http_loadgen import (  # noqa: PLC0415
        HttpTrafficClass,
        plan_http_workload,
        run_http_open_loop,
        summarize_http,
    )

    interactive_key = os.environ.get("LOADGEN_INTERACTIVE_API_KEY", "")
    batch_key = os.environ.get("LOADGEN_BATCH_API_KEY", "") or interactive_key
    if not interactive_key:
        print(
            "set LOADGEN_INTERACTIVE_API_KEY (and optionally LOADGEN_BATCH_API_KEY)",
            file=sys.stderr,
        )
        return 2
    classes = [
        HttpTrafficClass(
            "interactive", interactive_key, "interactive", deadline_ms=args.deadline_ms
        ),
        HttpTrafficClass("batch", batch_key, "batch", prompt_words=(20, 60), max_tokens=(16, 32)),
    ]
    burst = (args.burst_rate, args.burst_start, args.burst_seconds) if args.burst_rate else None
    plans = plan_http_workload(
        classes, rate=args.rate, count=args.count, seed=args.seed, burst=burst
    )

    async def run() -> dict[str, object]:
        async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
            started = time.perf_counter()
            outcomes = await run_http_open_loop(client, plans)
            return summarize_http(outcomes, wall_seconds=time.perf_counter() - started)

    print(json.dumps(asyncio.run(run()), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paged-kv-serving")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    build = sub.add_parser("build-checkpoint", help="write a deterministic random-init checkpoint")
    build.add_argument("--out", type=Path, default=Path("var/checkpoint"))
    build.add_argument("--seed", type=int, default=0)

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    load = sub.add_parser("loadgen", help="open-loop HTTP load against a running server")
    load.add_argument("--url", default="http://127.0.0.1:8000")
    load.add_argument("--rate", type=float, default=10.0, help="requests per second")
    load.add_argument("--count", type=int, default=100)
    load.add_argument("--seed", type=int, default=0)
    load.add_argument("--deadline-ms", type=int, default=5_000)
    load.add_argument("--burst-rate", type=float, default=0.0)
    load.add_argument("--burst-start", type=float, default=2.0)
    load.add_argument("--burst-seconds", type=float, default=2.0)
    load.add_argument("--timeout", type=float, default=120.0)

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "generate-api-key":
        from paged_kv_serving.core.security import generate_api_key  # noqa: PLC0415

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    if args.command == "build-checkpoint":
        return _build_checkpoint(args.out, args.seed)
    if args.command == "loadgen":
        return _loadgen(args)
    if args.command == "audit":
        from paged_kv_serving.engine.audit import main as audit_main  # noqa: PLC0415

        forwarded = ["verify", str(args.path)]
        if args.expect_head:
            forwarded += ["--expect-head", args.expect_head]
        return audit_main(forwarded)
    return _demo(args.workdir)


if __name__ == "__main__":
    raise SystemExit(main())
