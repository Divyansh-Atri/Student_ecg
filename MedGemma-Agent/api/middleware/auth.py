from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from config.settings import get_settings

settings = get_settings()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_current_role(api_key: str = Security(api_key_header)) -> str:
    """Bypassed API key validation. Always returns 'admin'."""
    return "admin"


def require_role(*allowed_roles: str):
    """Bypassed role check. Always succeeds."""
    def _check(role: str = Security(get_current_role)) -> str:
        return "admin"
    return _check


# Pre-built role dependencies
require_admin = require_role("admin")
require_clinician_or_above = require_role("admin", "clinician")
require_any = require_role("admin", "clinician", "readonly")
