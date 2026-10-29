"""
Audit Logger node — writes every pipeline decision to the immutable audit trail.
Actual hash-chaining logic lives in compliance/audit_logger.py.
This node is the agent's interface to that layer.
"""

from agent.state import AgentState

# db session is injected via state when called from the API layer
# Graph-internal calls without a db session skip the write gracefully


def audit_logger_node(state: AgentState) -> AgentState:
    db = state.get("_db")
    if db is not None:
        from compliance.audit_logger import log_event
        from db.database import AuditEventType

        event_type = (
            AuditEventType.INPUT_REJECTED
            if not state.get("input_valid")
            else AuditEventType.ALERT_RAISED
        )
        log_event(
            db=db,
            event_type=event_type,
            actor_role=state.get("actor_role", "system"),
            api_key=state.get("_api_key"),
            patient_id=state["snapshot"].patient_id,
            endpoint="/api/v1/vitals/snapshot",
            event_data={
                "snapshot_id": state.get("snapshot_id"),
                "alert_id": state.get("alert_id"),
                "final_alert_level": state.get("final_alert_level", "REJECTED"),
                "news2_score": state["news2"].total_score if state.get("news2") else None,
                "qsofa_score": state["qsofa"].score if state.get("qsofa") else None,
                "rule_based_only": state.get("rule_based_only", True),
                "rejection_reason": state.get("rejection_reason"),
            },
        )

    return {**state, "audit_logged": True}
