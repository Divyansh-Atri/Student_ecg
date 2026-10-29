"""
LangGraph agent graph — wires all 7 nodes into the monitoring pipeline.

Flow:
  input_validator
       ├── (invalid) → audit_logger  [rejection path]
       ├── (immediate_critical) → risk_scorer → alert_router → audit_logger
       └── (normal) → ehr_retriever → risk_scorer → llm_analyzer
                                                         → output_validator
                                                         → alert_router
                                                         → audit_logger
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.input_validator import input_validator_node
from agent.nodes.ehr_retriever import ehr_retriever_node
from agent.nodes.risk_scorer import risk_scorer_node
from agent.nodes.llm_analyzer import llm_analyzer_node
from agent.nodes.output_validator import output_validator_node
from agent.nodes.alert_router import alert_router_node
from agent.nodes.audit_logger import audit_logger_node
from vitals.schemas import AlertLevel


def _route_after_input_validator(state: AgentState) -> str:
    if not state["input_valid"]:
        return "audit_logger"          # rejected — log and stop
    if state["immediate_critical"]:
        return "risk_scorer"           # bypass LLM for immediate CRITICAL
    return "ehr_retriever"             # normal path


def _route_after_risk_scorer(state: AgentState) -> str:
    if state.get("immediate_critical"):
        return "alert_router"          # skip LLM entirely
    return "llm_analyzer"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("input_validator", input_validator_node)
    graph.add_node("ehr_retriever", ehr_retriever_node)
    graph.add_node("risk_scorer", risk_scorer_node)
    graph.add_node("llm_analyzer", llm_analyzer_node)
    graph.add_node("output_validator", output_validator_node)
    graph.add_node("alert_router", alert_router_node)
    graph.add_node("audit_logger", audit_logger_node)

    graph.set_entry_point("input_validator")

    graph.add_conditional_edges(
        "input_validator",
        _route_after_input_validator,
        {
            "audit_logger": "audit_logger",
            "risk_scorer": "risk_scorer",
            "ehr_retriever": "ehr_retriever",
        },
    )
    graph.add_edge("ehr_retriever", "risk_scorer")
    graph.add_conditional_edges(
        "risk_scorer",
        _route_after_risk_scorer,
        {
            "alert_router": "alert_router",
            "llm_analyzer": "llm_analyzer",
        },
    )
    graph.add_edge("llm_analyzer", "output_validator")
    graph.add_edge("output_validator", "alert_router")
    graph.add_edge("alert_router", "audit_logger")
    graph.add_edge("audit_logger", END)

    return graph.compile()


# Module-level compiled graph — imported by the API layer
monitoring_agent = build_graph()
