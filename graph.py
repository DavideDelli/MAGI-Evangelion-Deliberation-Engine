from langgraph.graph import StateGraph, START, END
from schemas import MAGIState
from config import MELCHIOR_CONFIGS, BALTHASAR_CONFIGS, CASPER_CONFIGS, DESC_MELCHIOR, DESC_BALTHASAR, DESC_CASPER
from agents import ask_agent_with_fallback
from utils import extract_vote, save_log

async def melchior_node(state: MAGIState):
    print("🧠 Melchior is deliberating...")
    response, elapsed, model_used = await ask_agent_with_fallback(
        DESC_MELCHIOR, state["dilemma"], MELCHIOR_CONFIGS, "Melchior"
    )
    return {"melchior_response": response, "melchior_elapsed": elapsed, "melchior_model_used": model_used}

async def balthasar_node(state: MAGIState):
    print("🤱 Balthasar is deliberating...")
    response, elapsed, model_used = await ask_agent_with_fallback(
        DESC_BALTHASAR, state["dilemma"], BALTHASAR_CONFIGS, "Balthasar"
    )
    return {"balthasar_response": response, "balthasar_elapsed": elapsed, "balthasar_model_used": model_used}

async def casper_node(state: MAGIState):
    print("💃 Casper is deliberating...")
    response, elapsed, model_used = await ask_agent_with_fallback(
        DESC_CASPER, state["dilemma"], CASPER_CONFIGS, "Casper"
    )
    return {"casper_response": response, "casper_elapsed": elapsed, "casper_model_used": model_used}

def arbitration_node(state: MAGIState):
    print("\n⚖️  Counting votes...")
    votes = {
        "melchior":  extract_vote(state["melchior_response"]),
        "balthasar": extract_vote(state["balthasar_response"]),
        "casper":    extract_vote(state["casper_response"]),
    }
    missing_votes = [name for name, v in votes.items() if v == "?"]
    if missing_votes:
        print(f"  ⚠️  Vote not detected from: {', '.join(missing_votes)}")

    votes_yes = sum(1 for v in votes.values() if v == "SI")
    votes_no  = sum(1 for v in votes.values() if v == "NO")

    if votes_yes > votes_no:
        decision = f"APPROVED ({votes_yes} to {votes_no})"
    elif votes_no > votes_yes:
        decision = f"REJECTED ({votes_no} to {votes_yes})"
    else:
        decision = f"TIE (YES={votes_yes}, NO={votes_no}, ?={len(missing_votes)})"

    print(f"  🧠 Melchior  ({state.get('melchior_model_used', '?')}): {votes['melchior']}")
    print(f"  🤱 Balthasar ({state.get('balthasar_model_used', '?')}): {votes['balthasar']}")
    print(f"  💃 Casper    ({state.get('casper_model_used', '?')}): {votes['casper']}")
    print(f"  → Decision: {decision}")

    return {"final_decision": decision}

def logging_node(state: MAGIState):
    save_log(state)
    return {}

def build_graph():
    builder = StateGraph(MAGIState)

    builder.add_node("melchior", melchior_node)
    builder.add_node("balthasar", balthasar_node)
    builder.add_node("casper", casper_node)
    builder.add_node("arbitration", arbitration_node)
    builder.add_node("logging", logging_node)

    # Parallel Execution
    builder.add_edge(START, "melchior")
    builder.add_edge(START, "balthasar")
    builder.add_edge(START, "casper")

    # Sync
    builder.add_edge("melchior", "arbitration")
    builder.add_edge("balthasar", "arbitration")
    builder.add_edge("casper", "arbitration")

    # Output
    builder.add_edge("arbitration", "logging")
    builder.add_edge("logging", END)

    return builder.compile()

# Istanzia il grafo pronto all'uso
magi_system = build_graph()