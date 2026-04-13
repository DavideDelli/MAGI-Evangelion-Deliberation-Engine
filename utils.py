import os
import re
import json
from datetime import datetime
from openai import RateLimitError
from schemas import MAGIState
from config import MELCHIOR_CONFIGS, BALTHASAR_CONFIGS, CASPER_CONFIGS

def is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause and isinstance(cause, RateLimitError):
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "quota" in msg or "too many requests" in msg

def extract_vote(response: str) -> str:
    match = re.search(r"VOTO\s*:\s*(S[IÌ]|NO)", response.upper())
    if match:
        vote = match.group(1)
        return "SI" if vote.startswith("S") else "NO"
    return "?"

def save_log(state: MAGIState) -> None:
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