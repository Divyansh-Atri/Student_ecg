# Security Incident Response Procedure
**Document ID:** SEC-001  
**Version:** 1.0  
**Standard:** ISO 27001:2022  
**Device:** MedGemma Post-Operative Monitoring Agent  
**Date:** 2026-04-09  

---

## 1. Incident Classification

| Severity | Description | Examples |
|---|---|---|
| P1 — Critical | PHI breach or patient safety impact | Unauthorized PHI access, system producing incorrect alerts unchecked |
| P2 — High | Security control failure | API key compromise, audit log tampering detected |
| P3 — Medium | Service degradation | Ollama unavailability, database corruption |
| P4 — Low | Anomaly with no immediate impact | Unusual API access pattern, failed auth attempts |

---

## 2. Response Timeline

| Severity | Detect → Contain | Contain → Investigate | Notify |
|---|---|---|---|
| P1 | Immediately | Within 4 hours | Within 24 hours (internal) / 60 days (HIPAA breach notification if applicable) |
| P2 | Within 1 hour | Within 24 hours | Within 48 hours (internal) |
| P3 | Within 4 hours | Within 72 hours | Next business day |
| P4 | Within 24 hours | Within 1 week | Optional |

---

## 3. Response Procedures

### P1 — PHI Breach

1. **Contain:** Immediately revoke affected API keys (update `.env`, restart server)
2. **Preserve:** Copy audit logs before any system changes — `GET /api/v1/audit` (admin key)
3. **Verify chain:** Run `compliance/audit_logger.py:verify_chain()` to confirm log integrity
4. **Assess scope:** Identify which patient records were accessed and by whom (audit log actor field)
5. **Notify:** Follow HIPAA Breach Notification Rule — individuals within 60 days if >500 records, annual report if <500
6. **Remediate:** Rotate all API keys, review access control configuration, patch vulnerability

### P2 — API Key Compromise

1. **Revoke:** Remove compromised key from `.env` immediately, restart server
2. **Audit:** Query audit log for all actions taken with the compromised key
3. **Assess:** Determine what data was accessed or modified
4. **Replace:** Issue new key to legitimate user via secure channel
5. **Document:** Record incident, actions taken, and timeline

### P3 — Audit Log Tampering Detected

1. **Isolate:** Take system offline until investigation complete
2. **Preserve:** Backup current database before any writes
3. **Investigate:** Run `verify_chain()` to identify the first tampered entry
4. **Restore:** Restore from last known-good backup if available
5. **Report:** Document findings and notify relevant stakeholders

---

## 4. API Key Rotation

Rotate API keys regularly (recommended: every 90 days) or immediately after any suspected compromise:

```bash
# Generate new key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Update .env with new key
# Restart server
./stop.sh && ./start.sh
```

---

## 5. Contact

In a clinical deployment, this section should list:
- Security Officer contact
- Clinical Informatics lead
- HIPAA Privacy Officer
- Legal counsel
