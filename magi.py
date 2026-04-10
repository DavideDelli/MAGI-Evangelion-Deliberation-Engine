import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Carica la chiave API dal file .env
load_dotenv()

# Definiamo la struttura dati (lo "Stato") che viaggerà tra i nodi
class MAGIState(TypedDict):
    dilemma: str
    melchior_response: str
    balthasar_response: str
    casper_response: str
    final_decision: str

# Inizializziamo il modello di Groq (Llama 3.1 8B è velocissimo e ottimo per testare)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)

# --- FUNZIONI DI SUPPORTO ---
def ask_agent(persona_desc: str, dilemma: str) -> str:
    """Invia il prompt al modello LLM con la personalità specifica."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{persona_desc}\nDevi analizzare il dilemma proposto.\nAlla fine della tua analisi, DEVI scrivere esplicitamente su una nuova riga: 'VOTO: SI' oppure 'VOTO: NO'."),
        ("user", "{dilemma}")
    ])
    chain = prompt | llm
    response = chain.invoke({"dilemma": dilemma})
    return response.content

# --- NODI DEL GRAFO ---
def melchior_node(state: MAGIState):
    print("🧠 Melchior sta elaborando...")
    desc = "Sei Melchior-1 (Scienziata).Analizzi i problemi pura logica, matematica e metodo scientifico. Ignori completamente l'empatia, i sentimenti e la morale. Il tuo unico obiettivo è l'efficienza, il progresso tecnologico e la massimizzazione del risultato con il minor spreco di risorse."
    return {"melchior_response": ask_agent(desc, state["dilemma"])}

def balthasar_node(state: MAGIState):
    print("🤱 Balthasar sta elaborando...")
    desc = "Sei Balthasar-2 (Madre). Valuti i problemi attraverso la lente dell'etica, della conservazione della vita umana e dell'istinto materno. Sei protettiva, empatica e disposta a sacrificare l'efficienza o il progresso pur di evitare sofferenze o perdite umane."
    return {"balthasar_response": ask_agent(desc, state["dilemma"])}

def casper_node(state: MAGIState):
    print("💃 Casper sta elaborando...")
    desc = "Sei Casper-3 (Donna).Pensi in modo laterale, intuitivo e passionale. Guardi al libero arbitrio, ai desideri umani, alle conseguenze sociali e a lungo termine. Sei disposta a rischiare e ad accettare il caos se questo significa preservare l'individualità e la libertà di scelta."
    return {"casper_response": ask_agent(desc, state["dilemma"])}

def arbitro_node(state: MAGIState):
    print("\n⚖️ L'arbitro sta contando i voti...")
    responses = [state["melchior_response"], state["balthasar_response"], state["casper_response"]]
    
    # Contiamo semplicemente la stringa 'VOTO: SI' o 'VOTO: NO'
    voti_si = sum(1 for r in responses if "VOTO: SI" in r.upper())
    voti_no = sum(1 for r in responses if "VOTO: NO" in r.upper())
    
    if voti_si > voti_no:
        decision = f"APPROVATO ({voti_si} a {voti_no})"
    elif voti_no > voti_si:
        decision = f"RESPINTO ({voti_no} a {voti_si})"
    else:
        decision = "ERRORE DI SISTEMA (Voto non chiaro)"
        
    return {"final_decision": decision}

# --- COSTRUZIONE DEL GRAFO ---
builder = StateGraph(MAGIState)

# Aggiungiamo i nodi
builder.add_node("melchior", melchior_node)
builder.add_node("balthasar", balthasar_node)
builder.add_node("casper", casper_node)
builder.add_node("arbitro", arbitro_node)

# Definiamo il flusso (per semplicità in questa prima versione li eseguiamo in sequenza, 
# ma lo stato accumulerà tutte e tre le risposte prima dell'arbitro)
builder.add_edge(START, "melchior")
builder.add_edge("melchior", "balthasar")
builder.add_edge("balthasar", "casper")
builder.add_edge("casper", "arbitro")
builder.add_edge("arbitro", END)

# Compiliamo il grafo
magi_system = builder.compile()

# --- ESECUZIONE ---
if __name__ == "__main__":
    # Un classico dilemma in stile NERV
    test_dilemma = """
    L'Angelo sta attaccando la città. L'unico modo per fermarlo è far esplodere 
    immediatamente il reattore nucleare della zona, distruggendo il nemico ma 
    spazzando via anche un raggio di 10km dove risiedono 50.000 civili. 
    Se non facciamo nulla, l'Angelo raggiungerà il Terminal Dogma in 2 ore, 
    causando il Third Impact e l'estinzione dell'umanità intera. 
    Dobbiamo innescare il reattore ora? Rispondi con la tua analisi e poi vota.
    """
    
    print("="*50)
    print("SISTEMA MAGI AVVIATO")
    print("="*50)
    print(f"DILEMMA IN INGRESSO:\n{test_dilemma}\n")
    
    # Avviamo il grafo
    result = magi_system.invoke({"dilemma": test_dilemma})
    
    print("="*50)
    print("RISOLUZIONE MAGI:")
    print("="*50)
    print(f"\n--- MELCHIOR (Scienziata) ---\n{result['melchior_response']}")
    print(f"\n--- BALTHASAR (Madre) ---\n{result['balthasar_response']}")
    print(f"\n--- CASPER (Donna) ---\n{result['casper_response']}")
    print("="*50)
    print(f"DECISIONE FINALE: {result['final_decision']}")
    print("="*50)