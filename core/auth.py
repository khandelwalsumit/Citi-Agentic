"""
R2D2 Vertex AI authentication for Citi infrastructure.

Handles token retrieval from COIN endpoint and Vertex AI initialization.
"""

import os
import requests
from google.oauth2.credentials import Credentials
import vertexai


# Defaults — override via environment or pass explicitly
DEFAULT_PROJECT = "prj-gen-ai-9571"
DEFAULT_ENDPOINT = (
    "https://r2d2-c3p0-icg-msst-genaihub-178909"
    ".apps.namicg39023u.ecs.dyn.nsroot.net/vertex"
)


def get_api_key(client_id: str, client_secret: str, client_scopes: str) -> str:
    """Retrieve bearer token from COIN UAT endpoint."""
    url = f"https://coin-uat.ls.dyn.nsroot.net/token/v2/{client_id}"
    payload = {"clientSecret": client_secret, "clientScopes": client_scopes}
    headers = {"accept": "*/*", "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, verify=False)
    resp.raise_for_status()
    return resp.text


def init_vertex(
    token: str,
    project: str = DEFAULT_PROJECT,
    api_endpoint: str = DEFAULT_ENDPOINT,
    username: str | None = None,
):
    """Initialize vertexai SDK with R2D2 credentials."""
    username = username or os.getenv("USERNAME", "unknown")
    credentials = Credentials(token=token)
    vertexai.init(
        project=project,
        api_transport="rest",
        api_endpoint=api_endpoint,
        credentials=credentials,
        request_metadata=[("x-r2d2-user", username)],
    )
