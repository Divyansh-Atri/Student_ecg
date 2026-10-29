from agent.state import AgentState
from rag.retriever import retrieve_patient_context


def ehr_retriever_node(state: AgentState) -> AgentState:
    context = retrieve_patient_context(state["snapshot"].patient_id)
    return {
        **state,
        "ehr_context": context,
        "ehr_available": "No EHR data available" not in context,
    }
