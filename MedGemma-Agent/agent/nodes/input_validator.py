from agent.state import AgentState
from guardrails.input_guardrails import validate_input


def input_validator_node(state: AgentState) -> AgentState:
    result = validate_input(state["snapshot"])
    return {
        **state,
        "input_valid": result.valid,
        "rejection_reason": result.rejection_reason,
        "immediate_critical": result.immediate_critical,
        "news2": result.news2_preview,
        "qsofa": result.qsofa_preview,
    }
