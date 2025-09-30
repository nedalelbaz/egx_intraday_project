# Release Notes

## v1.0.0 – Initial Release

* FastAPI webhook hardened for Telegram path token and secret header.
* Strict environment validation for PAPER mode deployments.
* Structured JSON logging with secret redaction.
* In-memory per-IP rate limiting using token bucket algorithm.
* Risk manager enforcing daily loss cap and 2% reserve settlement with append-only ledgers.
* Dockerfile for non-root execution with `/healthz` HEALTHCHECK.
* Comprehensive documentation and pytest suite (environment, security, risk controls).
