"""
Configuration — loaded from .env file.

Copy .env.example to .env and fill in your R2D2 credentials.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# R2D2 Vertex AI
R2D2_PROJECT = os.getenv("R2D2_PROJECT", "prj-gen-ai-9571")
R2D2_ENDPOINT = os.getenv(
    "R2D2_ENDPOINT",
    "https://r2d2-c3p0-icg-msst-genaihub-178909.apps.namicg39023u.ecs.dyn.nsroot.net/vertex",
)
R2D2_USERNAME = os.getenv("USERNAME", "unknown")

# COIN Auth
COIN_CLIENT_ID = os.getenv("COIN_CLIENT_ID", "")
COIN_CLIENT_SECRET = os.getenv("COIN_CLIENT_SECRET", "")
COIN_CLIENT_SCOPES = os.getenv("COIN_CLIENT_SCOPES", "")

# CA Bundle for Citi internal SSL
CA_BUNDLE = os.getenv("REQUESTS_CA_BUNDLE", "utils/CitiInternalCAChain_PROD.pem")

# Default model
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

# Agents directory
AGENTS_DIR = os.getenv("AGENTS_DIR", "agents")
