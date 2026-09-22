"""A CPU inference engine for a small transformer, written to be measured rather than configured.

Sub-packages:

- ``engine``: the paged KV cache, the continuous-batching scheduler, admission control, the
  numpy transformer they run, the static-batching baseline and the HTTP API.
- ``ml``: the request-telemetry data contract, run manifests, a content-addressed registry,
  PSI and KS drift monitors and pickle-free serialization.
- ``core``: settings, logging with PII redaction, tracing, metrics and the hardened app factory.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
