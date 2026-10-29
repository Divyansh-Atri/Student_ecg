# Authentication — Future State Requirements
**Document ID:** SEC-002  
**Version:** 1.0  
**Standard:** ISO 27001:2022  
**Device:** MedGemma Post-Operative Monitoring Agent  
**Date:** 2026-04-09  

---

## Current State (Testing Phase)

API key authentication is **disabled** during local development via `DEV_MODE=true` in `.env`.

All requests are treated as `clinician` role with no caller identity recorded in the audit log. This was a deliberate tradeoff to reduce friction during functional testing of the monitoring pipeline.

**This is not compliant with HIPAA, ISO 27001, or clinical deployment standards.**

---

## Known Gaps

| Gap | Risk | Required Before |
|---|---|---|
| No caller identity in audit log when `DEV_MODE=true` | Audit trail incomplete — cannot attribute actions to individuals | Staging deployment |
| Static API keys in `.env` | Keys can leak via env export, process listing, or misconfigured logging | Clinical deployment |
| No token expiry | Compromised key grants indefinite access | Clinical deployment |
| No MFA | Single-factor auth insufficient for access to PHI | Clinical deployment |
| No session management | No ability to revoke a session without rotating the key | Clinical deployment |

---

## Required for Production

Before any clinical or staging deployment, replace the current API key scheme with a proper identity and access management (IAM) solution:

### Option A — OAuth2 + OIDC (Recommended for hospital integration)
- Integrate with hospital identity provider (Azure AD, Okta, or similar)
- Use short-lived JWT bearer tokens (15–60 min expiry)
- Scopes map to existing roles: `admin`, `clinician`, `readonly`
- Refresh token rotation with revocation support
- MFA enforced at IdP level

### Option B — Short-lived API tokens with rotation
- Issue tokens via a `/auth/token` endpoint (username + password or SSO)
- Tokens expire after a configurable TTL (recommended: 8 hours for clinical shift)
- Token revocation list maintained in database
- Admin can revoke any token immediately

### Audit Log Fix
When real auth is in place, replace the `actor_key_hint` field in `AuditLog` with:
- `actor_user_id` — unique identifier from IdP
- `actor_display_name` — human-readable name for incident review
- `token_issued_at` — for session reconstruction

### Implementation Notes
- `DEV_MODE` flag must be removed or hard-disabled in any non-development build
- Add a startup check: if `DEV_MODE=true` and `DATABASE_URL` points to a non-local host, refuse to start
- All auth failures must be logged to the audit trail regardless of mode

---

## Reference

- HIPAA Security Rule §164.312(d) — Person or entity authentication
- ISO 27001:2022 A.9 — Access control
- NIST SP 800-63B — Digital identity guidelines
