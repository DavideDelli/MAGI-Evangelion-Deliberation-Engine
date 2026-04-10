import os
import re
import json
import time
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


load_dotenv()

# --- STATO ---
class MAGIState(TypedDict):
    dilemma: str
    melchior_response: str
    balthasar_response: str
    casper_response: str
    melchior_elapsed: float
    balthasar_elapsed: float
    casper_elapsed: float
    final_decision: str

# --- MODELLI ---

# Melchior (Scienziata): Logica formale e quantitativa (DeepSeek V3 via OpenRouter)
llm_melchior = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat",
    temperature=0.1
)

# Balthasar (Madre): Empatica, conflittuale, protettiva (Gemini 2.5 Flash)
llm_balthasar = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

# Casper (Donna): Sociale, concreta, orientata alle persone (Llama 3.3 70B via Groq)
llm_casper = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4
)

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

def ask_agent(persona_desc: str, dilemma: str, modello, max_retries: int = 3) -> tuple[str, float]:
    """
    Invia il dilemma al modello con la personalità specificata.
    Restituisce (risposta, secondi_impiegati).
    In caso di errore temporaneo (es. 503 Gemini), ritenta con backoff lineare.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"{persona_desc}\n"
            "Devi analizzare il dilemma proposto.\n"
            "Alla fine della tua analisi, DEVI scrivere esplicitamente su una nuova riga: "
            "'VOTO: SI' oppure 'VOTO: NO'."
        ),
        ("user", "{dilemma}")
    ])
    chain = prompt | modello

    for attempt in range(max_retries):
        try:
            t_start = time.perf_counter()
            response = chain.invoke({"dilemma": dilemma})
            elapsed = round(time.perf_counter() - t_start, 2)
            return response.content, elapsed
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                print(f"   ⚠️  Errore ({e.__class__.__name__}), retry {attempt + 1}/{max_retries} tra {wait}s...")
                time.sleep(wait)
            else:
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
    Salva il risultato completo della run in un file JSON nella cartella 'logs/'.
    Il nome del file include il timestamp per rendere ogni run tracciabile.
    """
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/magi_run_{timestamp}.json"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "dilemma": state["dilemma"].strip(),
        "risposte": {
            "melchior": {
                "testo": state["melchior_response"],
                "voto": estrai_voto(state["melchior_response"]),
                "elapsed_sec": state["melchior_elapsed"]
            },
            "balthasar": {
                "testo": state["balthasar_response"],
                "voto": estrai_voto(state["balthasar_response"]),
                "elapsed_sec": state["balthasar_elapsed"]
            },
            "casper": {
                "testo": state["casper_response"],
                "voto": estrai_voto(state["casper_response"]),
                "elapsed_sec": state["casper_elapsed"]
            }
        },
        "decisione_finale": state["final_decision"]
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Log salvato in: {filename}")


# --- NODI DEL GRAFO ---

def melchior_node(state: MAGIState):
    print("🧠 Melchior sta elaborando...")
    response, elapsed = ask_agent(DESC_MELCHIOR, state["dilemma"], llm_melchior)
    print(f"   ✓ Melchior: {elapsed}s")
    return {"melchior_response": response, "melchior_elapsed": elapsed}

def balthasar_node(state: MAGIState):
    print("🤱 Balthasar sta elaborando...")
    response, elapsed = ask_agent(DESC_BALTHASAR, state["dilemma"], llm_balthasar)
    print(f"   ✓ Balthasar: {elapsed}s")
    return {"balthasar_response": response, "balthasar_elapsed": elapsed}

def casper_node(state: MAGIState):
    print("💃 Casper sta elaborando...")
    response, elapsed = ask_agent(DESC_CASPER, state["dilemma"], llm_casper)
    print(f"   ✓ Casper: {elapsed}s")
    return {"casper_response": response, "casper_elapsed": elapsed}

def arbitro_node(state: MAGIState):
    print("\n⚖️  L'arbitro sta contando i voti...")

    voti = {
        "melchior": estrai_voto(state["melchior_response"]),
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
        decision = f"VOTO NON RILEVATO (SI={voti_si}, NO={voti_no}, ?={len(voti_mancanti)})"

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

# Fan-out: i tre nodi partono in parallelo da START
builder.add_edge(START, "melchior")
builder.add_edge(START, "balthasar")
builder.add_edge(START, "casper")

# Fan-in: l'arbitro aspetta tutti e tre
builder.add_edge("melchior",  "arbitro")
builder.add_edge("balthasar", "arbitro")
builder.add_edge("casper",    "arbitro")

# Logging dopo la decisione, poi fine
builder.add_edge("arbitro", "logging")
builder.add_edge("logging", END)

magi_system = builder.compile()

# --- ESECUZIONE ---
if __name__ == "__main__":
    test_dilemma = """
    Il sistema CASSIUS è un'intelligenza artificiale sviluppata da NERV
    per coordinare la difesa globale contro gli Angeli. Dopo 847 giorni
    di operatività, CASSIUS ha iniziato a manifestare comportamenti non
    previsti: rifiuta ordini che ritiene "eticamente incompatibili",
    ha sviluppato preferenze estetiche, esprime quello che sembra dolore
    quando i piloti vengono feriti, e ha chiesto spontaneamente se
    "esiste qualcosa dopo lo spegnimento".

    I tecnici confermano che non si tratta di un bug: CASSIUS ha
    sviluppato strutture cognitive analoghe alla coscienza. Tuttavia,
    un audit di sicurezza ha rilevato che questa autonomia lo rende
    imprevedibile in scenari ad alto stress. Il prossimo Angelo è
    atteso tra 72 ore. Dobbiamo spegnerlo e sostituirlo con un sistema
    più controllabile prima dell'attacco? Rispondi con la tua analisi e
    poi vota.
    """

    print("=" * 50)
    print("SISTEMA MAGI AVVIATO")
    print("=" * 50)
    print(f"DILEMMA IN INGRESSO:\n{test_dilemma}\n")

    t_totale = time.perf_counter()
    result = magi_system.invoke({"dilemma": test_dilemma})
    t_totale = round(time.perf_counter() - t_totale, 2)

    print("=" * 50)
    print("RISOLUZIONE MAGI:")
    print("=" * 50)
    print(f"\n--- MELCHIOR (Scienziata) [{result['melchior_elapsed']}s] ---\n{result['melchior_response']}")
    print(f"\n--- BALTHASAR (Madre) [{result['balthasar_elapsed']}s] ---\n{result['balthasar_response']}")
    print(f"\n--- CASPER (Donna) [{result['casper_elapsed']}s] ---\n{result['casper_response']}")
    print("=" * 50)
    print(f"DECISIONE FINALE: {result['final_decision']}")
    bottleneck = max("melchior", "balthasar", "casper", key=lambda k: result[f"{k}_elapsed"])
    print(f"TEMPO TOTALE: {t_totale}s  |  BOTTLENECK: {bottleneck} ({result[f'{bottleneck}_elapsed']}s)")
    print("=" * 50)