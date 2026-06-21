from langgraph.graph import StateGraph, START, END
from backend.graph.state import GraphState
from backend.graph.nodes.retrieve import retrieve_node
from backend.graph.nodes.rerank import rerank_node
from backend.graph.nodes.expand import expand_node
from backend.graph.nodes.grade import grade_node
from backend.graph.nodes.generate import generate_node
from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
from backend.graph.nodes.ask_corrective import ask_corrective_node
from backend.graph.nodes.classify import classify_node
from backend.graph.edges import route_after_grade, route_after_corrective, route_after_classify


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("expand", expand_node)
    builder.add_node("grade", grade_node)
    builder.add_node("chat", generate_node)
    builder.add_node("clarify", generate_node)
    builder.add_node("general", generate_node)
    builder.add_node("generate_deep_dive", generate_node)
    builder.add_node("generate_ask", generate_node)
    builder.add_node("deep_dive_insufficient", deep_dive_insufficient_node)
    builder.add_node("ask_corrective", ask_corrective_node)

    # Conversational turns (greetings, small talk) skip retrieval entirely.
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {"chat": "chat", "clarify": "clarify", "general": "general", "retrieve": "retrieve"},
    )
    builder.add_edge("chat", END)
    builder.add_edge("clarify", END)
    builder.add_edge("general", END)
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "expand")
    builder.add_edge("expand", "grade")

    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate_deep_dive": "generate_deep_dive",
            "deep_dive_insufficient": "deep_dive_insufficient",
            "generate_ask": "generate_ask",
            "ask_corrective": "ask_corrective",
        },
    )

    builder.add_edge("generate_deep_dive", END)
    builder.add_edge("deep_dive_insufficient", END)
    builder.add_edge("generate_ask", END)

    builder.add_conditional_edges(
        "ask_corrective",
        route_after_corrective,
        {
            "retrieve": "retrieve",
            "generate_ask": "generate_ask",
        },
    )

    return builder.compile()
