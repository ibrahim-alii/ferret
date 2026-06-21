import os


def route_after_classify(state: dict) -> str:
    intent = state.get("intent")
    if intent == "chat":
        return "chat"
    if intent == "clarify":
        return "clarify"
    if intent == "general":
        return "general"
    return "retrieve"


def route_after_grade(state: dict) -> str:
    mode = state["mode"]
    grade_result = state.get("grade_result", {}) or {}
    sufficient = grade_result.get("sufficient", False)

    if mode == "deep_dive":
        return "generate_deep_dive" if sufficient else "deep_dive_insufficient"
    else:  # ask
        return "generate_ask" if sufficient else "ask_corrective"


def route_after_corrective(state: dict) -> str:
    max_retries = int(os.environ.get("CRAG_MAX_RETRIES", "2"))
    retry_count = state.get("retry_count", 0)
    if retry_count < max_retries:
        return "retrieve"
    return "generate_ask"
