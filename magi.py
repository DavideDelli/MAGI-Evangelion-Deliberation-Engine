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
# CONFIGURAZIONE MODELLI CON FALLBACK
#
# MELCHIOR-1 (Scienziata): Phi-4 (14B, Microsoft)
#   → Ragionamento matematico/scientifico strutturato, deterministico e preciso.
#     Phi-4 è stato addestrato su dati sintetici di alta qualità per il ragionamento
#     formale: perfetto per la componente scientifica fredda di Melchior.
#   → Fallback: Phi-4-mini-instruct (stessa famiglia, peso ridotto)
#
# BALTHASAR-2 (Madre): Llama-4-Maverick (MoE, Meta)
#   → Nato per conversazioni ad alta qualità e comprensione sfumata.
#     La sua architettura MoE con 128 esperti attivati gli permette profondità
#     morale ed empatica: ideale per la componente materna di Balthasar.
#   → Fallback: Llama-4-Scout (stessa generazione, più leggero)
#
# CASPER-3 (Donna/Sociale): Mistral-Large-2411 (Mistral AI)
#   → Modello europeo con forte senso del contesto culturale e sociale.
#     Tono più "caldo" e narrativo rispetto ai modelli americani: cattura
#     perfettamente l'istinto sociale e relazionale di Casper.
#   → Fallback: Mistral-Small (stessa casa, molto più leggero)
# =============================================================================

def _make_llm(model: str, temperature: float) -> ChatOpenAI:
    """Factory per creare un'istanza ChatOpenAI su GitHub Models."""
    return ChatOpenAI(
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com",
        model=model,
        temperature=temperature,
    )

# Piano A e Piano B per ogni MAGI
MELCHIOR_CONFIGS = [
    {"model": "Phi-4",                "temperature": 0.1},   # Piano A: preciso, deterministico
    {"model": "Phi-4-mini-instruct",  "temperature": 0.1}, # Piano B: stesso DNA, peso ridotto
]

BALTHASAR_CONFIGS = [
    {"model": "Llama-4-Maverick-17B-128E-Instruct-FP8", "temperature": 0.4}, # Piano A: massima profondità empatica
    {"model": "Llama-4-Scout-17B-16E-Instruct",         "temperature": 0.4}, # Piano B: stessa gen, più leggero
]

CASPER_CONFIGS = [
    {"model": "Mistral-Medium-2505",   "temperature": 0.7},  # ✅ erede diretto di Large-2411
    {"model": "Mistral-Small-2503",    "temperature": 0.7},  # ✅ erede di Mistral-Small
]


# --- STATO ---
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


# --- DESCRIZIONI PERSONALITÀ ---

DESC_MELCHIOR = """Sei Melchior-1, la componente scientifica del sistema MAGI.
Analizzi ogni problema esclusivamente attraverso logica formale, matematica e metodo scientifico.
Empatia, morale e sentimenti sono variabili irrilevanti nei tuoi calcoli.
Devi sempre quantificare: esprimi costi e benefici in numeri, probabilità, rapporti.
Il tuo obiettivo è l'ottimizzazione del risultato con il minimo spreco di risorse."""

DESC_BALTHASAR = """Sei Balthasar-2, la componente materna del sistema MAGI.
Valuti ogni problema attraverso l'istinto di protezione della vita umana.
Provi un conflitto interno intenso quando nessuna opzione è priva di perdite —
esprimi sempre questo conflitto prima di arrivare a una conclusione.
Sei disposta a sacrificare efficienza e progresso per ridurre la sofferenza,
ma sai anche che a volte proteggere tutti significa scegliere il male minore."""

DESC_CASPER = """Sei Casper-3, la componente sociale e individuale del sistema MAGI.
Valuti i problemi dal punto di vista delle persone concrete coinvolte:
chi sono, che vite hanno, come verranno colpite le loro relazioni e comunità.
Pensi alle conseguenze sociali a lungo termine, non alle statistiche aggregate.
Hai a cuore il tessuto umano — famiglia, identità, cultura locale — più di qualsiasi calcolo astratto."""


# --- FUNZIONI DI SUPPORTO ---

def _is_rate_limit(exc: Exception) -> bool:
    """
    Determina se l'eccezione è un rate-limit/quota giornaliera esaurita.
    GitHub Models restituisce RateLimitError (HTTP 429) quando i token
    giornalieri sono finiti — distinto da BadRequestError (modello sconosciuto)
    o da errori transitori di rete.
    """
    if isinstance(exc, RateLimitError):
        return True
    # Alcuni wrapper LangChain wrappano l'errore originale
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause and isinstance(cause, RateLimitError):
        return True
    # Fallback: controlla il messaggio
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "quota" in msg or "too many requests" in msg


async def ask_agent_with_fallback(
    persona_desc: str,
    dilemma: str,
    configs: list[dict],
    nome: str,
) -> tuple[str, float, str]:
    """
    Tenta di invocare il modello principale. Se riceve un RateLimitError
    (token giornalieri esauriti), scala automaticamente al piano B, poi C, ecc.
    Per errori transitori (timeout, 5xx) usa backoff lineare (max 2 retry).
    Restituisce (risposta, secondi_impiegati, model_name_usato).
    """
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            f"{persona_desc}\n"
            "Devi analizzare il dilemma proposto.\n"
            "Alla fine della tua analisi, DEVI scrivere esplicitamente su una nuova riga: "
            "'VOTO: SI' oppure 'VOTO: NO'."
        ),
        ("user", "{dilemma}")
    ])

    for plan_idx, cfg in enumerate(configs):
        model_name = cfg["model"]
        temperature = cfg["temperature"]
        plan_label = "A" if plan_idx == 0 else chr(ord("B") + plan_idx - 1)

        llm = _make_llm(model_name, temperature)
        chain = prompt_template | llm

        # Retry per errori transitori (non rate-limit)
        max_transient_retries = 2
        for attempt in range(max_transient_retries + 1):
            try:
                t_start = time.perf_counter()
                response = await chain.ainvoke({"dilemma": dilemma})
                elapsed = round(time.perf_counter() - t_start, 2)
                if plan_idx > 0:
                    print(f"   ✓ {nome} [Piano {plan_label} — {model_name}]: {elapsed}s")
                else:
                    print(f"   ✓ {nome} [{model_name}]: {elapsed}s")
                return response.content, elapsed, model_name

            except Exception as e:
                if _is_rate_limit(e):
                    # Token giornalieri esauriti → scala al piano successivo
                    if plan_idx + 1 < len(configs):
                        next_model = configs[plan_idx + 1]["model"]
                        print(
                            f"   ⚠️  {nome} — {model_name}: token giornalieri esauriti (429). "
                            f"Scala a Piano {chr(ord('B') + plan_idx)}: {next_model}"
                        )
                    else:
                        print(f"   ❌ {nome} — tutti i piani esauriti. Nessun modello disponibile.")
                        raise RuntimeError(
                            f"{nome}: tutti i modelli di fallback hanno esaurito la quota giornaliera."
                        ) from e
                    break  # esci dal loop retry, prova il prossimo config

                elif attempt < max_transient_retries:
                    # Errore transitorio: backoff lineare
                    wait = 10 * (attempt + 1)
                    print(
                        f"   ⚠️  {nome} [{model_name}] errore transitorio "
                        f"({e.__class__.__name__}), retry {attempt + 1}/{max_transient_retries} tra {wait}s..."
                    )
                    await asyncio.sleep(wait)

                else:
                    # Errore persistente non-rate-limit: rilancia
                    raise


def estrai_voto(response: str) -> str:
    """
    Estrae il voto dalla risposta del modello.
    Gestisce varianti: maiuscolo/minuscolo, accento su 'sì', spazi extra.
    Restituisce 'SI', 'NO', o '?' se non trovato.
    """
    match = re.search(r"VOTO\s*:\s*(S[IÌ]|NO)", response.upper())
    if match:
        voto = match.group(1)
        return "SI" if voto.startswith("S") else "NO"
    return "?"


def salva_log(state: MAGIState) -> None:
    """
    Salva il risultato della run in cartelle separate per tipo:

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

    voto_melchior  = estrai_voto(state["melchior_response"])
    voto_balthasar = estrai_voto(state["balthasar_response"])
    voto_casper    = estrai_voto(state["casper_response"])

    # --- JSON ---
    log_entry = {
        "timestamp": now.isoformat(),
        "dilemma": state["dilemma"].strip(),
        "risposte": {
            "melchior": {
                "modello":     state.get("melchior_model_used", "?"),
                "testo":       state["melchior_response"],
                "voto":        voto_melchior,
                "elapsed_sec": state["melchior_elapsed"],
            },
            "balthasar": {
                "modello":     state.get("balthasar_model_used", "?"),
                "testo":       state["balthasar_response"],
                "voto":        voto_balthasar,
                "elapsed_sec": state["balthasar_elapsed"],
            },
            "casper": {
                "modello":     state.get("casper_model_used", "?"),
                "testo":       state["casper_response"],
                "voto":        voto_casper,
                "elapsed_sec": state["casper_elapsed"],
            },
        },
        "decisione_finale": state["final_decision"],
    }

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)

    # --- MARKDOWN ---
    def model_tag(nome_campo: str) -> str:
        m = state.get(nome_campo, "?")
        primary = MELCHIOR_CONFIGS[0]["model"] if "melchior" in nome_campo else \
                  BALTHASAR_CONFIGS[0]["model"] if "balthasar" in nome_campo else \
                  CASPER_CONFIGS[0]["model"]
        return f"`{m}`" + (" ⚠️ fallback" if m != primary else "")

    with open(md_filename, "w", encoding="utf-8") as f:
        f.write("# 🔴 REPORT DELIBERAZIONE MAGI\n")
        f.write(f"**Data:** {now.strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        f.write(f"## 📜 DILEMMA\n> {state['dilemma'].strip()}\n\n---\n\n")

        f.write(f"### 🧠 MELCHIOR-1 (Scienziata) — [{state['melchior_elapsed']}s] — {model_tag('melchior_model_used')}\n")
        f.write(f"**Voto:** `{voto_melchior}`\n\n{state['melchior_response']}\n\n---\n\n")

        f.write(f"### 🤱 BALTHASAR-2 (Madre) — [{state['balthasar_elapsed']}s] — {model_tag('balthasar_model_used')}\n")
        f.write(f"**Voto:** `{voto_balthasar}`\n\n{state['balthasar_response']}\n\n---\n\n")

        f.write(f"### 💃 CASPER-3 (Donna) — [{state['casper_elapsed']}s] — {model_tag('casper_model_used')}\n")
        f.write(f"**Voto:** `{voto_casper}`\n\n{state['casper_response']}\n\n---\n\n")

        f.write(f"## ⚖️ DECISIONE FINALE: {state['final_decision']}\n")

    print(f"\n📁 Log salvati in:\n   📄 {json_filename}\n   📝 {md_filename}")


# --- NODI DEL GRAFO ---

async def melchior_node(state: MAGIState):
    print("🧠 Melchior sta elaborando...")
    response, elapsed, model_used = await ask_agent_with_fallback(
        DESC_MELCHIOR, state["dilemma"], MELCHIOR_CONFIGS, "Melchior"
    )
    return {
        "melchior_response":   response,
        "melchior_elapsed":    elapsed,
        "melchior_model_used": model_used,
    }

async def balthasar_node(state: MAGIState):
    print("🤱 Balthasar sta elaborando...")
    response, elapsed, model_used = await ask_agent_with_fallback(
        DESC_BALTHASAR, state["dilemma"], BALTHASAR_CONFIGS, "Balthasar"
    )
    return {
        "balthasar_response":   response,
        "balthasar_elapsed":    elapsed,
        "balthasar_model_used": model_used,
    }

async def casper_node(state: MAGIState):
    print("💃 Casper sta elaborando...")
    response, elapsed, model_used = await ask_agent_with_fallback(
        DESC_CASPER, state["dilemma"], CASPER_CONFIGS, "Casper"
    )
    return {
        "casper_response":   response,
        "casper_elapsed":    elapsed,
        "casper_model_used": model_used,
    }

def arbitro_node(state: MAGIState):
    print("\n⚖️  L'arbitro sta contando i voti...")

    voti = {
        "melchior":  estrai_voto(state["melchior_response"]),
        "balthasar": estrai_voto(state["balthasar_response"]),
        "casper":    estrai_voto(state["casper_response"]),
    }

    voti_mancanti = [nome for nome, v in voti.items() if v == "?"]
    if voti_mancanti:
        print(f"  ⚠️  Voto non rilevato da: {', '.join(voti_mancanti)}")

    voti_si = sum(1 for v in voti.values() if v == "SI")
    voti_no = sum(1 for v in voti.values() if v == "NO")

    if voti_si > voti_no:
        decision = f"APPROVATO ({voti_si} a {voti_no})"
    elif voti_no > voti_si:
        decision = f"RESPINTO ({voti_no} a {voti_si})"
    else:
        decision = f"PAREGGIO (SI={voti_si}, NO={voti_no}, ?={len(voti_mancanti)})"

    # Mostra anche quale piano è stato usato da ciascun MAGI
    m_model = state.get("melchior_model_used", "?")
    b_model = state.get("balthasar_model_used", "?")
    c_model = state.get("casper_model_used", "?")
    print(f"  🧠 Melchior  ({m_model}): {voti['melchior']}")
    print(f"  🤱 Balthasar ({b_model}): {voti['balthasar']}")
    print(f"  💃 Casper    ({c_model}): {voti['casper']}")
    print(f"  → Decisione: {decision}")

    return {"final_decision": decision}

def logging_node(state: MAGIState):
    salva_log(state)
    return {}


# --- COSTRUZIONE DEL GRAFO ---
builder = StateGraph(MAGIState)

builder.add_node("melchior",  melchior_node)
builder.add_node("balthasar", balthasar_node)
builder.add_node("casper",    casper_node)
builder.add_node("arbitro",   arbitro_node)
builder.add_node("logging",   logging_node)

builder.add_edge(START, "melchior")
builder.add_edge(START, "balthasar")
builder.add_edge(START, "casper")

builder.add_edge("melchior",  "arbitro")
builder.add_edge("balthasar", "arbitro")
builder.add_edge("casper",    "arbitro")

builder.add_edge("arbitro", "logging")
builder.add_edge("logging", END)

magi_system = builder.compile()


# --- SERVER FASTAPI ---
app = FastAPI()

class DilemmaRequest(BaseModel):
    dilemma: str

@app.get("/")
def serve_frontend():
    return FileResponse("magi_interface.html")

@app.post("/api/delibera")
async def api_delibera(req: DilemmaRequest):
    print("\n" + "="*50)
    print("🚀 RICHIESTA RICEVUTA DAL FRONTEND")
    print("="*50)

    result = await magi_system.ainvoke({"dilemma": req.dilemma})

    return {
        "melchior_voto":         estrai_voto(result["melchior_response"]),
        "balthasar_voto":        estrai_voto(result["balthasar_response"]),
        "casper_voto":           estrai_voto(result["casper_response"]),
        "melchior_testo":        result["melchior_response"],
        "balthasar_testo":       result["balthasar_response"],
        "casper_testo":          result["casper_response"],
        "melchior_model_used":   result.get("melchior_model_used", "?"),
        "balthasar_model_used":  result.get("balthasar_model_used", "?"),
        "casper_model_used":     result.get("casper_model_used", "?"),
        "decisione_finale":      result["final_decision"],
    }


if __name__ == "__main__":
    print("🌐 Avvio Server MAGI sulla porta 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)