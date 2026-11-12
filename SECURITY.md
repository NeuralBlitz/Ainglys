# Security policy — NeuralBlitz/Ainglys

Ainglys is a research-oriented Python/ML repository undergoing active security hardening. Passing automated tests does not constitute approval for an internet-facing, multi-tenant, or regulated production deployment.

## Supported versions

| Version | Security maintenance |
| --- | --- |
| Current `main` (0.1.x research development) | Best-effort issue investigation and updates |
| Earlier experimental snapshots, stale branches or unpublished prototypes | Not guaranteed |

There is no committed security-response service-level agreement or guaranteed patch timeline.

## Reporting a vulnerability

If GitHub private vulnerability reporting is enabled on this repository, use its **Security → Report a vulnerability** path. Otherwise, contact the repository maintainers through an appropriate private channel. Do not put exploit details, unredacted secrets, personal information, live API keys, or private credentials in a public issue or pull request.

For non-sensitive improvements, create a normal GitHub issue. Include the affected commit, affected file/path, severity rationale, reproducible symptoms and a safe minimal reproduction. Avoid testing against systems you do not own or have permission to test.

## Active security boundaries

- API operations require `X-API-Key` and use bounded inputs.
- Optional `AI_HUB_API_KEYS_JSON` supports scoped, hashed API client tokens. Legacy single-key behavior is retained for compatibility.
- API request and training quotas are **process-local**, not distributed or globally enforceable.
- The minimal Compose deployment binds published service ports to loopback by default.
- Protected model loading uses PyTorch `weights_only=True` for state-dict loads; other file/data loaders still require review.

See [scoped API credentials](docs/SCOPED_API_KEYS.md), the [Ainglys audit](https://app.notion.com/p/41525ca447c181c39a20e61bd536bd67), and [security issue #3](https://github.com/NeuralBlitz/Ainglys/issues/3).

## Security engineering and dependency updates

The repository uses the main GitHub Actions test matrix, a separate CodeQL Python security analysis workflow, and scheduled Dependabot checks for uv, ACCA pip dependencies, GitHub Actions and Docker.

Static analysis provides findings to triage, **not a guarantee of vulnerability absence**. Do not bypass failing tests or security scans merely to produce a green badge. Record false-positive rationale and mitigation evidence when resolving a finding.

Outstanding controls include repository branch protection/required checks, production-grade identity and revocation, centralized per-user quotas, resource limitations, monitored releases, and independently reproducible model evaluation.
