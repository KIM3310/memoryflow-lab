# Validation Strategy

## Claims the test suite protects

1. GQA KV sizing uses `kv_heads`, not `attention_heads`.
2. Weight and KV precision change capacity in the expected direction.
3. HBM-only fails clearly when weights plus KV exceed capacity.
4. Tiering can restore capacity when remote capacity is sufficient.
5. More remote bandwidth or overlap cannot increase modeled tiering latency.
6. A larger HBM window reduces remote traffic but consumes more HBM.
7. Near-memory reduction lowers transferred cold-KV bytes by the configured ratio.
8. Slow near-memory compute can erase the transfer benefit and change the winner.
9. The same scenario produces byte-for-byte stable JSON evidence.
10. Bundled systems are labeled synthetic to block accidental product claims.

## Verification layers

| Layer | Check |
|---|---|
| Domain | invalid dimensions, precisions, head relationships, and policy values fail fast |
| Equations | closed-form weight and KV examples |
| Properties | monotonic bandwidth, overlap, and window behavior |
| Failure states | HBM, weight, and remote-capacity rejection |
| Interfaces | CLI summaries, detailed traces, API validation, dashboard delivery |
| Evidence | generated JSON and Markdown must match committed artifacts |

## Remaining validation gap

No public result is calibrated against hardware counters or a cycle-accurate simulator. A production-oriented next phase would replay real framework traces and compare the first-order model against a trusted simulator such as Ramulator 2.0 or measured server data, with all calibration inputs versioned.
