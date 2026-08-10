# Contributing to memoryflow-lab

Thanks for helping improve the simulator. Keep changes focused, reproducible, and easy to review.

## Development workflow

1. Create a branch from `main`.
2. Install the Python 3.11+ development environment with `make install`.
3. Add or update tests and evidence for behavior or model changes.
4. Run `make verify` before opening a pull request.
5. Describe the change, validation evidence, and any modeling limitations in the pull request.

Use concise conventional commit subjects such as `fix:`, `test:`, `docs:`, `deps:`, or `ci:`.

## Quality and security

- Keep generated evidence reproducible from committed inputs.
- Document changes to equations, assumptions, APIs, schemas, or operational commands.
- Never commit credentials, tokens, private data, local caches, virtual environments, or raw production data.
- Report security-sensitive findings through the private channel described in `SECURITY.md`.
