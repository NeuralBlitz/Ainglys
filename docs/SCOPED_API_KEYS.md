# Scoped API credentials (opt-in)

Ainglys supports a small, environment-configured API-key registry. Its purpose is to separate client permissions in a **single-process research deployment** while preserving the existing one-key configuration. It is not a replacement for an external identity provider, managed secret storage, distributed quotas, security monitoring, or per-user login.

## Modes

**Legacy mode (default):** Leave `AI_HUB_API_KEYS_JSON` **unset** and set a 32+-character `AI_HUB_API_KEY`. That one key retains all existing API privileges, and the original shared request budget remains in effect. Do not enable this mode for public multi-user services.

**Scoped keyring mode (opt in):** Set `AI_HUB_API_KEYS_JSON` to a JSON array of 1–32 credential records, each with exactly three fields: `id`, `sha256`, and `scopes`. In this mode, the server **does not accept the legacy master key** unless it is separately registered by digest; malformed/empty registries produce HTTP 503 without fallback.

Each credential must have:

- `id`: unique public client identifier, 1–64 ASCII letters/digits/hyphens/underscores, starting with a letter or digit.
- `sha256`: lowercase hex SHA-256 digest of an independently generated random secret of at least 32 characters. **Generate high-entropy secrets**; length alone does not confer entropy.
- `scopes`: a nonempty list of unique scopes from `read`, `predict`, `train`, `models:write`.

Tokens are transmitted in the existing `X-API-Key` header. They must be treated as passwords, used only with TLS for remote services, and kept out of Git history, logs, tickets, and documentation examples. The server compares the hashed token in constant time against the complete configured registry.

## Permission matrix

| Scope | Allowed operations |
| --- | --- |
| `read` | `GET /status`, `GET /models` |
| `predict` | `POST /predict` |
| `train` | `POST /train`, `POST /train/supervised` |
| `models:write` | `POST /save`, `POST /load/{model_name}`, `DELETE /models/{model_name}` |

Access without the required scope returns **403**. Missing/invalid credentials return **401**. Invalid registry configuration returns **503**. Protected endpoints not in the scope policy are denied by default until their access requirements are explicitly defined.

Quota windows are isolated by credential: by default 600 requests per minute and 60 training operations per minute per configured key. A key that exhausts a budget receives **429** with `Retry-After`. Limits are **per running process**; restarting workers or spreading calls across replicas bypasses this constraint. Use a centralized gateway/storage-backed limiter for any production or multiworker deployment.

## Provisioning example

Generate separate random client tokens using a password manager/secret service. The following **local** example prints a token and its registry entry to your terminal; store the token in a secret manager and avoid copying it to source control.

```python
import hashlib
import json
import secrets

token = secrets.token_urlsafe(48)
record = {
    "id": "research-trainer",
    "sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    "scopes": ["read", "predict", "train"],
}
print("Client token (secret; store securely):", token)
print("Registry record (no plaintext token):", json.dumps(record))
```

Add additional independently generated records for read-only clients, model administrators, or other applications. Configure the API process with the **JSON array** of all records in `AI_HUB_API_KEYS_JSON`, not with their plaintext tokens.

To connect a CLI or Streamlit client, provide that client's **plaintext secret** as its own `AI_HUB_API_KEY` environment variable. The CLI/Streamlit process sends this as `X-API-Key`. Server-side `AI_HUB_API_KEY` is ignored when the registry is set.

For the minimal Compose deployment, opt into the registry explicitly:

```bash
# These values come from a secure secret source, not a committed .env file.
export AI_HUB_API_KEY='<one enrolled client token>'
export AI_HUB_API_KEYS_JSON='<complete JSON array containing this client fingerprint>'
docker compose -f docker-compose.yml -f compose.scoped-keys.yml config --quiet
docker compose -f docker-compose.yml -f compose.scoped-keys.yml up --build -d
```

`AI_HUB_API_KEY` is still required by the base Compose manifest because the optional UI uses it as a client credential. In keyring mode, the API server ignores it except when its digest appears in the registry.

## Rotation and revocation limitations

Removing a credential record or changing its scopes takes effect **on the next request after the running API process receives the new environment configuration**. Updating an environment variable in a host shell does **not** automatically modify an already running container/process. In ordinary Compose deployment, change the secret source and **recreate the affected API containers**; verify the old token now returns 401 and the new token has only intended scopes. This keyring has no dedicated remote revocation service or live configuration reload.

Separate credentials allow more targeted revocation and process-local quotas, but they do **not** imply user identity verification. A stolen token retains whatever scopes it possesses. The JSON registry is capped at 32 entries and 32 KiB to bound parsing and process memory. Up to 32 active per-key quota windows can coexist; rapid credential rotation may temporarily exhaust that process-local pool (returning 429).

## Outstanding security work

- Add an external identity provider and persistent hashed credential store, audited lifecycle and scope administration.
- Centralized, durable per-principal rate limits with coordinated multiworker semantics.
- Resource reservations/quotas for training jobs, authorization logs, detection, alerts and timeouts.
- Repository branch protection, required checks and vulnerability monitoring.

Tracked in [issue #3](https://github.com/NeuralBlitz/Ainglys/issues/3) and [issue #5](https://github.com/NeuralBlitz/Ainglys/issues/5).
