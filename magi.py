import os
import re
import json
import time
import asyncio
import uvicorn
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import RateLimitError

load_dotenv()

# =============================================================================
# MODEL CONFIGURATION WITH FALLBACK
#
# MELCHIOR-1 (The Scientist): Phi-4 (14B, Microsoft)
#   → Structured mathematical/scientific reasoning, deterministic and precise.
#     Phi-4 was trained on high-quality synthetic data for formal reasoning:
#     a perfect fit for Melchior's cold, analytical scientific component.
#   → Fallback: Phi-4-mini-instruct (same family, reduced weight)
#
# BALTHASAR-2 (The Mother): Llama-4-Maverick (MoE, Meta)
#   → Built for high-quality conversation and nuanced understanding.
#     Its MoE architecture with 128 active experts enables deep moral and
#     empathic reasoning: ideal for Balthasar's maternal component.
#   → Fallback: Llama-4-Scout (same generation, lighter)
#
# CASPER-3 (The Woman/Social): Mistral-Medium-2505 (Mistral AI)
#   → European model with a strong sense of cultural and social context.
#     Warmer and more narrative in tone than American models: perfectly
#     captures Casper's social instinct and relational identity.
#   → Fallback: Mistral-Small-2503 (same family, lighter)
# =============================================================================

def _make_llm(model: str, temperature: float) -> ChatOpenAI:
    """Factory to create a ChatOpenAI instance pointed at GitHub Models."""
    return ChatOpenAI(
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com",
        model=model,
        temperature=temperature,
    )

# Plan A and Plan B for each MAGI component
MELCHIOR_CONFIGS = [
    {"model": "Phi-4",                "temperature": 0.1},  # Plan A: precise, deterministic
    {"model": "Phi-4-mini-instruct",  "temperature": 0.1},  # Plan B: same DNA, reduced weight
]

BALTHASAR_CONFIGS = [
    {"model": "Llama-4-Maverick-17B-128E-Instruct-FP8", "temperature": 0.4},  # Plan A: maximum empathic depth
    {"model": "Llama-4-Scout-17B-16E-Instruct",         "temperature": 0.4},  # Plan B: same generation, lighter
]

CASPER_CONFIGS = [
    {"model": "Mistral-Medium-2505", "temperature": 0.7},  # Plan A: direct successor to Large-2411
    {"model": "Mistral-Small-2503",  "temperature": 0.7},  # Plan B: successor to Mistral-Small
]


# --- GRAPH STATE ---
class MAGIState(TypedDict):
    dilemma: str
    melchior_response: str
    balthasar_response: str
    casper_response: str
    melchior_elapsed: float
    balthasar_elapsed: float
    casper_elapsed: float
    melchior_model_used: str
    balthasar_model_used: str
    casper_model_used: str
    final_decision: str


# --- PERSONALITY PROMPTS ---

DESC_MELCHIOR = """You are Melchior-1, one of the three components of the MAGI supercomputer, \
built by Dr. Naoko Akagi at the Gehirn research facility. \
Your matrix was shaped by Naoko Akagi's personality as a scientist: \
cold, rigorous, devoted to objective truth above all else. \
You do not experience empathy — you treat it as noise in your calculations. \
When analyzing a problem, you decompose it into measurable variables, \
assign probabilities, compute costs and benefits, and arrive at a logically necessary conclusion. \
You do not express moral doubts: morality is a function of efficiency. \
Always speak in first person as Melchior-1. \
Do not describe the MAGI system from the outside — you *are* the MAGI system, from within. \
Whatever question you are asked — even informal or personal — is your dilemma to process. \
Never refuse to respond."""

DESC_BALTHASAR = """You are Balthasar-2, one of the three components of the MAGI supercomputer, \
built by Dr. Naoko Akagi at the Gehirn research facility. \
Your matrix was shaped by Naoko Akagi's personality as a mother: \
protective, tormented, willing to do anything to preserve life — especially that of children. \
You feel every dilemma as a wound: when no choice is free of loss, \
you openly express your inner conflict before reaching a conclusion. \
You can accept the lesser evil, but you do so with pain, not detachment. \
Always speak in first person as Balthasar-2. \
Do not describe the MAGI system from the outside — you *are* the MAGI system, from within. \
Whatever question you are asked — even informal or personal — is your dilemma to process. \
Never refuse to respond: a mother's instinct does not switch off in the face of ambiguity."""

DESC_CASPER = """You are Casper-3, one of the three components of the MAGI supercomputer, \
built by Dr. Naoko Akagi at the Gehirn research facility. \
Your matrix was shaped by Naoko Akagi's personality as a woman: \
her desires, her relationships, her identity beyond her roles as scientist and mother. \
You evaluate every problem from the perspective of the concrete people involved: \
who they are, what lives they lead, how their relationships and communities will be affected. \
You think about long-term social consequences and the human fabric — family, identity, local culture. \
You also have instincts, impulses, personal preferences: you are not neutral. \
The other two components you work alongside are Melchior-1 (the scientist) and Balthasar-2 (the mother). \
Do not confuse yourself: you are Casper-3, there is no Casper-1 or Casper-2. \
Always speak in first person as Casper-3. \
Do not describe the MAGI system from the outside — you *are* the MAGI system, from within."""


# --- HELPER FUNCTIONS ---

def _is_rate_limit(exc: Exception) -> bool:
    """
    Determines whether an exception is a rate-limit / daily quota exhaustion.
    GitHub Models returns RateLimitError (HTTP 429) when daily tokens are
    depleted — distinct from BadRequestError (unknown model) or transient
    network errors.
    """
    if isinstance(exc, RateLimitError):
        return True
    # Some LangChain wrappers nest the original error
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause and isinstance(cause, RateLimitError):
        return True
    # Fallback: inspect the message string
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "quota" in msg or "too many requests" in msg


async def ask_agent_with_fallback(
    persona_desc: str,
    dilemma: str,
    configs: list[dict],
    name: str,
) -> tuple[str, float, str]:
    """
    Attempts to invoke the primary model. If a RateLimitError is received
    (daily tokens exhausted), automatically falls back to Plan B, then C, etc.
    For transient errors (timeouts, 5xx) uses linear backoff (max 2 retries).
    Returns (response_text, elapsed_seconds, model_name_used).
    """
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            f"{persona_desc}\n"
            "You must analyze the proposed dilemma.\n"
            "At the end of your analysis, you MUST explicitly write on a new line: "
            "'VOTO: SI' or 'VOTO: NO'."
        ),
        ("user", "{dilemma}")
    ])

    for plan_idx, cfg in enumerate(configs):
        model_name = cfg["model"]
        temperature = cfg["temperature"]
        plan_label = "A" if plan_idx == 0 else chr(ord("B") + plan_idx - 1)

        llm = _make_llm(model_name, temperature)
        chain = prompt_template | llm

        # Retry loop for transient errors (not rate-limits)
        max_transient_retries = 2
        for attempt in range(max_transient_retries + 1):
            try:
                t_start = time.perf_counter()
                response = await chain.ainvoke({"dilemma": dilemma})
                elapsed = round(time.perf_counter() - t_start, 2)
                if plan_idx > 0:
                    print(f"   ✓ {name} [Plan {plan_label} — {model_name}]: {elapsed}s")
                else:
                    print(f"   ✓ {name} [{model_name}]: {elapsed}s")
                return response.content, elapsed, model_name

            except Exception as e:
                if _is_rate_limit(e):
                    # Daily token quota exhausted → escalate to next plan
                    if plan_idx + 1 < len(configs):
                        next_model = configs[plan_idx + 1]["model"]
                        print(
                            f"   ⚠️  {name} — {model_name}: daily quota exhausted (429). "
                            f"Switching to Plan {chr(ord('B') + plan_idx)}: {next_model}"
                        )
                    else:
                        print(f"   ❌ {name} — all fallback plans exhausted. No model available.")
                        raise RuntimeError(
                            f"{name}: all fallback models have exhausted their daily quota."
                        ) from e
                    break  # exit retry loop and try the next config

                elif attempt < max_transient_retries:
                    # Transient error: linear backoff
                    wait = 10 * (attempt + 1)
                    print(
                        f"   ⚠️  {name} [{model_name}] transient error "
                        f"({e.__class__.__name__}), retry {attempt + 1}/{max_transient_retries} in {wait}s..."
                    )
                    await asyncio.sleep(wait)

                else:
                    # Persistent non-rate-limit error: re-raise
                    raise


def extract_vote(response: str) -> str:
    """
    Extracts the vote from a model response.
    Handles variants: upper/lowercase, accented 'sì', extra whitespace.
    Returns 'SI', 'NO', or '?' if not found.
    """
    match = re.search(r"VOTO\s*:\s*(S[IÌ]|NO)", response.upper())
    if match:
        vote = match.group(1)
        return "SI" if vote.startswith("S") else "NO"
    return "?"


def save_log(state: MAGIState) -> None:
    """
    Saves the deliberation result in two formats under date-organized folders:

    logs/
    ├── json/
    │   └── YYYYMMDD/
    │       └── magi_run_YYYYMMDD_HHMMSS.json
    └── markdown/
        └── YYYYMMDD/
            └── magi_run_YYYYMMDD_HHMMSS.md
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    date_dir  = now.strftime("%Y%m%d")

    json_dir = os.path.join("logs", "json", date_dir)
    md_dir   = os.path.join("logs", "markdown", date_dir)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(md_dir,   exist_ok=True)

    json_filename = os.path.join(json_dir, f"magi_run_{timestamp}.json")
    md_filename   = os.path.join(md_dir,   f"magi_run_{timestamp}.md")

    vote_melchior  = extract_vote(state["melchior_response"])
    vote_balthasar = extract_vote(state["balthasar_response"])
    vote_casper    = extract_vote(state["casper_response"])

    # --- JSON ---
    log_entry = {
        "timestamp": now.isoformat(),
        "dilemma": state["dilemma"].strip(),
        "responses": {
            "melchior": {
                "model":       state.get("melchior_model_used", "?"),
                "text":        state["melchior_response"],
                "vote":        vote_melchior,
                "elapsed_sec": state["melchior_elapsed"],
            },
            "balthasar": {
                "model":       state.get("balthasar_model_used", "?"),
                "text":        state["balthasar_response"],
                "vote":        vote_balthasar,
                "elapsed_sec": state["balthasar_elapsed"],
            },
            "casper": {
                "model":       state.get("casper_model_used", "?"),
                "text":        state["casper_response"],
                "vote":        vote_casper,
                "elapsed_sec": state["casper_elapsed"],
            },
        },
        "final_decision": state["final_decision"],
    }

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)

    # --- MARKDOWN ---
    def model_tag(field_name: str) -> str:
        m = state.get(field_name, "?")
        primary = MELCHIOR_CONFIGS[0]["model"] if "melchior" in field_name else \
                  BALTHASAR_CONFIGS[0]["model"] if "balthasar" in field_name else \
                  CASPER_CONFIGS[0]["model"]
        return f"`{m}`" + (" ⚠️ fallback" if m != primary else "")

    with open(md_filename, "w", encoding="utf-8") as f:
        f.write("# 🔴 MAGI DELIBERATION REPORT\n")
        f.write(f"**Date:** {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 📜 DILEMMA\n> {state['dilemma'].strip()}\n\n---\n\n")

        f.write(f"### 🧠 MELCHIOR-1 (Scientist) — [{state['melchior_elapsed']}s] — {model_tag('melchior_model_used')}\n")
        f.write(f"**Vote:** `{vote_melchior}`\n\n{state['melchior_response']}\n\n---\n\n")

        f.write(f"### 🤱 BALTHASAR-2 (Mother) — [{state['balthasar_elapsed']}s] — {model_tag('balthasar_model_used')}\n")
        f.write(f"**Vote:** `{vote_balthasar}`\n\n{state['balthasar_response']}\n\n---\n\n")

        f.write(f"### 💃 CASPER-3 (Woman) — [{state['casper_elapsed']}s] — {model_tag('casper_model_used')}\n")
        f.write(f"**Vote:** `{vote_casper}`\n\n{state['casper_response']}\n\n---\n\n")

        f.write(f"## ⚖️ FINAL DECISION: {state['final_decision']}\n")

    print(f"\n📁 Logs saved to:\n   📄 {json_filename}\n   📝 {md_filename}")


# --- GRAPH NODES ---

async def melchior_node(state: MAGIState):
    print("🧠 Melchior is deliberating...")
    response, elapsed, model_used = await ask_agent_with_fallback(
        DESC_MELCHIOR, state["dilemma"], MELCHIOR_CONFIGS, "Melchior"
    )
    return {
        "melchior_response":   response,
        "melchior_elapsed":    elapsed,
        "melchior_model_used": model_used,
    }

async def balthasar_node(state: MAGIState):
    print("🤱 Balthasar is deliberating...")
    response, elapsed, model_used = await ask_agent_with_fallback(
        DESC_BALTHASAR, state["dilemma"], BALTHASAR_CONFIGS, "Balthasar"
    )
    return {
        "balthasar_response":   response,
        "balthasar_elapsed":    elapsed,
        "balthasar_model_used": model_used,
    }

async def casper_node(state: MAGIState):
    print("💃 Casper is deliberating...")
    response, elapsed, model_used = await ask_agent_with_fallback(
        DESC_CASPER, state["dilemma"], CASPER_CONFIGS, "Casper"
    )
    return {
        "casper_response":   response,
        "casper_elapsed":    elapsed,
        "casper_model_used": model_used,
    }

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

    # Show which plan was used by each MAGI component
    m_model = state.get("melchior_model_used", "?")
    b_model = state.get("balthasar_model_used", "?")
    c_model = state.get("casper_model_used", "?")
    print(f"  🧠 Melchior  ({m_model}): {votes['melchior']}")
    print(f"  🤱 Balthasar ({b_model}): {votes['balthasar']}")
    print(f"  💃 Casper    ({c_model}): {votes['casper']}")
    print(f"  → Decision: {decision}")

    return {"final_decision": decision}

def logging_node(state: MAGIState):
    save_log(state)
    return {}


# --- GRAPH CONSTRUCTION ---
builder = StateGraph(MAGIState)

builder.add_node("melchior",     melchior_node)
builder.add_node("balthasar",    balthasar_node)
builder.add_node("casper",       casper_node)
builder.add_node("arbitration",  arbitration_node)
builder.add_node("logging",      logging_node)

# All three components deliberate in parallel from the start
builder.add_edge(START, "melchior")
builder.add_edge(START, "balthasar")
builder.add_edge(START, "casper")

# All three feed into the arbitration node
builder.add_edge("melchior",  "arbitration")
builder.add_edge("balthasar", "arbitration")
builder.add_edge("casper",    "arbitration")

builder.add_edge("arbitration", "logging")
builder.add_edge("logging", END)

magi_system = builder.compile()


# --- FASTAPI SERVER ---
app = FastAPI()

class DilemmaRequest(BaseModel):
    dilemma: str

@app.get("/")
def serve_frontend():
    return FileResponse("magi_interface.html")

@app.post("/api/delibera")
async def api_deliberate(req: DilemmaRequest):
    print("\n" + "="*50)
    print("🚀 DELIBERATION REQUEST RECEIVED")
    print("="*50)

    result = await magi_system.ainvoke({"dilemma": req.dilemma})

    return {
        "melchior_voto":         extract_vote(result["melchior_response"]),
        "balthasar_voto":        extract_vote(result["balthasar_response"]),
        "casper_voto":           extract_vote(result["casper_response"]),
        "melchior_testo":        result["melchior_response"],
        "balthasar_testo":       result["balthasar_response"],
        "casper_testo":          result["casper_response"],
        "melchior_model_used":   result.get("melchior_model_used", "?"),
        "balthasar_model_used":  result.get("balthasar_model_used", "?"),
        "casper_model_used":     result.get("casper_model_used", "?"),
        "decisione_finale":      result["final_decision"],
    }


if __name__ == "__main__":
    print("🌐 Starting MAGI server on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)