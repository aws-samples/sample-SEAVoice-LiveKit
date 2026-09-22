"""Simple web server that serves the frontend and provides LiveKit tokens.

Run alongside the agent in dev mode:
    uv run python web/server.py

Then open http://localhost:8080 in a browser. Binds to 127.0.0.1 by default;
set WEB_HOST/WEB_PORT to override.
"""

import os
import sys
from pathlib import Path

import uvicorn
import yaml
from dotenv import load_dotenv
from livekit import api
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

load_dotenv(".env.local")

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

WEB_DIR = Path(__file__).parent
REPO_ROOT = WEB_DIR.parent

# Agent name comes from the same config the agent runs with (--config, same default)
_config_path = REPO_ROOT / "configs" / "telco_th.yaml"
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        _config_path = Path(sys.argv[i + 1])
        if not _config_path.is_absolute():
            _config_path = REPO_ROOT / _config_path
        break

with _config_path.open("r", encoding="utf-8") as f:
    AGENT_NAME = yaml.safe_load(f).get("agent_name", "my-agent")


async def index(request):
    return FileResponse(WEB_DIR / "index.html")


async def get_token(request):
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("web-user")
        .with_name("Web User")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="voice-agent",
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    jwt = token.to_jwt()

    # Auto-dispatch agent to the room
    lk_api = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    try:
        await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room="voice-agent", agent_name=AGENT_NAME
            )
        )
    except Exception:
        pass  # dispatch may already exist
    finally:
        await lk_api.aclose()

    return JSONResponse({"token": jwt, "url": LIVEKIT_URL, "agent_name": AGENT_NAME})


app = Starlette(
    routes=[
        Route("/", index),
        Route("/token", get_token),
    ]
)

if __name__ == "__main__":
    # Bind to loopback by default: this endpoint mints LiveKit tokens, so it
    # should not be reachable from the network. Mic access needs localhost
    # anyway, and remote use goes through an SSH tunnel (which targets
    # 127.0.0.1 on the remote). Set WEB_HOST to override.
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8080"))
    print(f"Serving web UI at http://{host}:{port}")
    print(f"LiveKit URL: {LIVEKIT_URL}")
    print(f"Agent name: {AGENT_NAME} (from {_config_path.name})")
    uvicorn.run(app, host=host, port=port)
