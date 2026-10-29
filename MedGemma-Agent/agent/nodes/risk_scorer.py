from agent.state import AgentState
from guardrails.clinical_rules import (
    calculate_news2,
    calculate_qsofa,
    determine_alert_level,
    determine_recommended_action,
)


def risk_scorer_node(state: AgentState) -> AgentState:
    snapshot = state["snapshot"]

    # Re-use pre-computed scores from input_validator if available
    news2 = state.get("news2") or calculate_news2(
        snapshot.systolic_bp, snapshot.diastolic_bp,
        snapshot.heart_rate, snapshot.spo2,
    )
    qsofa = state.get("qsofa") or calculate_qsofa(
        snapshot.systolic_bp, snapshot.heart_rate, snapshot.spo2,
    )

    alert_level = determine_alert_level(news2, qsofa)
    recommended_action = determine_recommended_action(alert_level)

    return {
        **state,
        "news2": news2,
        "qsofa": qsofa,
        "rule_alert_level": alert_level,
        "rule_recommended_action": recommended_action,
    }
