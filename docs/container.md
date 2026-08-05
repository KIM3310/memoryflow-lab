# Container Runtime Contract

MemoryFlow's container is a reproducible review surface for the analytical model. It is not a claim that the simulator is a production service or that its synthetic hardware profile represents a vendor product.

## Build boundary

- The Python 3.11 Alpine base is pinned by multi-platform digest.
- The builder stage creates a Python wheel from the checked-in package.
- The runtime stage contains only the installed wheel, runtime dependencies, and generated static dashboard.
- Package installers and build tooling are removed from the runtime image after installation.
- Tests, scenarios, evidence sources, repository history, and local virtual environments are excluded from the image context.
- OCI labels identify the source repository and MIT license.

## Runtime boundary

- Runs as unprivileged UID/GID `10001`.
- Supports a read-only root filesystem with a small temporary `/tmp` mount.
- Requires no Linux capabilities and sets `no-new-privileges`.
- Listens on container port `8000`; Compose binds it to loopback only.
- Uses `/health` for an application-level Docker health check.
- Persists no user or simulation payloads.

## Verification

`make docker-verify` proves more than image construction:

1. Build the multi-stage image.
2. Start it with the hardened runtime flags.
3. Wait for the live health endpoint.
4. Confirm the process UID is `10001`.
5. Confirm the dashboard is served.
6. Submit the checked-in 7B tiered-memory scenario.
7. Require a feasible result, 64 decode steps, and the expected transfer bottleneck.
8. Require Docker's image health status to become `healthy`.

The local Apple Silicon run validates `linux/arm64`. GitHub Actions repeats the same contract on an `amd64` Linux runner.

Vulnerability findings are time-dependent and are not stored as a permanent repository claim.
Re-scan the built image when making a release decision.
