"""Run the real ASGI server on a loopback port in a background thread.

httpx's in-process ASGI transport buffers a whole response before handing it back, so a client
using it can never hang up *mid-stream* -- by the time it sees the first SSE line the request has
already finished. Cancellation on disconnect is one of this repository's claims, so the demo and
the cancellation tests run uvicorn for real on ``127.0.0.1`` with an ephemeral port instead. The
socket never leaves the machine, so this stays offline.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI

__all__ = ["serve_in_thread"]


@contextmanager
def serve_in_thread(
    app: FastAPI, *, host: str = "127.0.0.1", startup_timeout: float = 20.0
) -> Iterator[str]:
    """Serve ``app`` (lifespan included) until the block exits. Yields the base URL."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    port = int(sock.getsockname()[1])
    config = uvicorn.Config(
        app,
        log_config=None,
        access_log=False,
        server_header=False,
        lifespan="on",
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, name="uvicorn-loopback", daemon=True
    )
    thread.start()
    deadline = time.monotonic() + startup_timeout
    try:
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("the loopback server did not start")
            time.sleep(0.01)
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=startup_timeout)
        sock.close()
