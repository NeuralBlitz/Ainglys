"""Default ASGI development entrypoint for hosted environments.

Set AI_HUB_API_KEY (at least 32 characters) before using authenticated routes.
"""
import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "ai_hub.api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
