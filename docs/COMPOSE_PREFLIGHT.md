# Core Compose deployment preflight

Ainglys supports a **minimal local deployment** through `docker-compose.yml`. The optional `ui` profile runs the Streamlit interface alongside the authenticated API. The legacy multi-service file, `docker-compose.full.experimental.yml`, has **not** been validated as a deployable production stack.

## Run the supported preflight

From the root of the Ainglys repository, with Docker Compose v2 installed:

```bash
# Supply a high-entropy key from a secret manager; do not commit it to Git.
export AI_HUB_API_KEY='<at least 32 random characters>'

python scripts/verify_compose.py
python scripts/verify_compose.py --ui

docker compose config --quiet
docker compose --profile ui config --quiet
```

The script runs `docker compose config --format json` and checks the rendered configuration. When `--ui` is supplied, it also requires a dedicated Streamlit `/_stcore/health` probe on port **8501** rather than inheriting the API image's healthcheck on port 8000. It fails when a requested service is missing, the build target changes from `production`, the API key is absent/short, extra services appear, a published port is not bound to localhost, unapproved host bind mounts appear, capabilities are not dropped, `no-new-privileges` or init/restart configuration is missing, API health checks are disabled, model storage is not a named volume, or the Streamlit UI no longer uses the internal API service with healthy startup ordering.

Expected local endpoints: `127.0.0.1:8000` (API) and, when using `--profile ui`, `127.0.0.1:8501` (UI). The script never prints rendered configuration or Compose stderr because they can contain secrets.

This check is included in the repository's GitHub Actions integrity job and is unit-tested on Python 3.11 and 3.12. The Docker smoke job also starts both API and optional Streamlit services, checks each service's actual HTTP health endpoint, and verifies scoped API authorization.

## Start the core deployment

```bash
docker compose up --build -d api
curl --fail http://127.0.0.1:8000/health

# Optional local Streamlit UI:
docker compose --profile ui up --build -d

# Inspect service health and logs:
docker compose ps
docker compose logs --tail 100 api

# Shutdown without removing named model data:
docker compose --profile ui down
```

The API requires an `X-API-Key` header for non-public endpoints. Configure `AI_HUB_API_KEYS_JSON` through `compose.scoped-keys.yml` only when opting into separately provisioned scoped credentials. See [Scoped API Keys](SCOPED_API_KEYS.md).

## Experimental full stack

```bash
python scripts/verify_compose.py --experimental-assets
```

The experimental inventory **always exits nonzero and cannot certify deployment**; it reports currently missing legacy assets such as `Dockerfile.ray`, Redis and Nginx configs, monitoring provisioning and init scripts. The experimental file also needs security review of public port exposure, secret handling, service entrypoints, pinned images, TLS termination, and resource isolation before it can be used. Use the tested core deployment instead.

## What the preflight does not prove

- It validates configuration, not a running cluster. Real Docker API build/authentication smoke tests remain separate CI jobs.
- It does not establish TLS on the public internet, multiuser authentication, centralized quotas or fully reproducible scientific results.
- It does not enforce branch rules or detect all application vulnerabilities; configure GitHub required checks separately.
- Docker engine privilege settings, host filesystem security, upstream images, secrets management, and production deployment architecture require independent verification.

Track unfinished infrastructure in [GitHub issue #4](https://github.com/NeuralBlitz/Ainglys/issues/4) and repository governance in [issue #5](https://github.com/NeuralBlitz/Ainglys/issues/5).
