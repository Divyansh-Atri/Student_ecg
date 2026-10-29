from agent.state import AgentState
from guardrails.output_guardrails import validate_llm_output
from config.settings import get_settings

settings = get_settings()


def output_validator_node(state: AgentState) -> AgentState:
    if not state.get("llm_available") or state.get("llm_raw_output") is None:
        return {
            **state,
            "llm_output_valid": False,
            "sanitized_reasoning": None,
            "llm_confidence": None,
            "flag_for_human_review": False,
            "output_rejection_reason": state.get("llm_error", "LLM unavailable"),
            "contributing_factors": None,
            "llm_limitations": None,
            "sepsis_risk_flag": state["qsofa"].high_risk,
        }

    result = validate_llm_output(
        llm_output=state["llm_raw_output"],
        rule_alert_level=state["rule_alert_level"],
        snapshot=state["snapshot"],
        min_confidence=settings.LLM_CONFIDENCE_THRESHOLD,
    )

    raw = state["llm_raw_output"]
    return {
        **state,
        "llm_output_valid": result.valid,
        "sanitized_reasoning": result.sanitized_reasoning,
        "llm_confidence": result.confidence,
        "flag_for_human_review": result.flag_for_human_review,
        "output_rejection_reason": result.rejection_reason,
        "contributing_factors": raw.get("contributing_factors") if result.valid else None,
        "llm_limitations": raw.get("limitations") if result.valid else None,
        "sepsis_risk_flag": bool(raw.get("sepsis_risk_flag", state["qsofa"].high_risk)),
    }
