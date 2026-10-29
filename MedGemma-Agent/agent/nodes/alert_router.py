"""
Alert Router node — merges rule-based and LLM assessments into the final alert.

Decision logic:
- If LLM output is valid and not contradicting: use rule level (safety-first)
  but include LLM reasoning for context.
- If LLM is unavailable or invalid: fall back to rule-based level only.
- CRITICAL from rules is never downgraded by LLM.
"""

from agent.state import AgentState
from vitals.schemas import AlertLevel
from guardrails.clinical_rules import determine_recommended_action


def alert_router_node(state: AgentState) -> AgentState:
    rule_level = state["rule_alert_level"]
    llm_valid = state.get("llm_output_valid", False)
    rule_based_only = not llm_valid

    # Safety: rule-based CRITICAL is never downgraded
    if rule_level == AlertLevel.CRITICAL:
        final_level = AlertLevel.CRITICAL
    elif llm_valid and state.get("llm_raw_output"):
        try:
            llm_level = AlertLevel(state["llm_raw_output"].get("risk_level", rule_level.value))
            # Take the more conservative (higher) of the two
            level_order = [AlertLevel.NORMAL, AlertLevel.MEDIUM, AlertLevel.HIGH, AlertLevel.CRITICAL]
            final_level = level_order[max(
                level_order.index(rule_level),
                level_order.index(llm_level),
            )]
        except (ValueError, AttributeError):
            final_level = rule_level
    else:
        final_level = rule_level

    return {
        **state,
        "final_alert_level": final_level,
        "final_recommended_action": determine_recommended_action(final_level),
        "rule_based_only": rule_based_only,
    }
