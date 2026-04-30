<div align="center">

# 🔴 MAGI Supercomputer System

**Motore di deliberazione multi‑LLM ispirato ai supercomputer MAGI di *Neon Genesis Evangelion*.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Ready-009688?logo=fastapi&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Orchestrator-8A2BE2">
</p>

> *"The MAGI are three supercomputers that form the brain of NERV. They were built by Dr. Naoko Akagi and contain her personality engrams — as a scientist, as a mother, as a woman."*  
> — *Neon Genesis Evangelion*

</div>

---

## ✨ Panoramica

Questo progetto riproduce l’architettura MAGI: tre agenti LLM con personalità distinte analizzano un dilemma in parallelo, votano (`VOTE: YES` o `VOTE: NO`) e un nodo di arbitraggio decide a maggioranza. I risultati vengono salvati come **JSON** e **Markdown**.

---

## 🖥️ Interfaccia

### Schermata di input
<img width="1280" height="800" alt="Input" src="https://github.com/user-attachments/assets/13aa3128-78da-4fa8-8ed9-79df2f4d5189" />

### Risultato deliberazione
<img width="2501" height="1274" alt="Result" src="https://github.com/user-attachments/assets/fa98873e-045e-44af-80e9-6ab512056bbb" />

---

## 🧠 Come funziona

Ogni deliberazione avvia tre agenti in **parallelo** con LangGraph. Ogni agente riceve il dilemma con un prompt di personalità bloccata e termina con un voto esplicito (`VOTE: YES` o `VOTE: NO`). Un nodo di arbitraggio conta i voti e determina il verdetto finale.

```
          ┌─────────────┐
          │   Dilemma   │
          └──────┬──────┘
        ┌────────┼────────┐
        ▼        ▼        ▼
  MELCHIOR-1  BALTHASAR-2  CASPER-3
  (Scientist) (Mother)    (Woman)
        └────────┬────────┘
                 ▼
            ARBITRATION
           (majority vote)
                 ▼
              LOGGING
           (JSON + Markdown)
```

---

## 🧬 Componenti MAGI

| Componente | Personalità | Modello | Temperature |
|-----------|-------------|---------|-------------|
| **MELCHIOR-1** | Dr. Akagi come **scienziata** — logica fredda e formale | `Phi-4` (Microsoft, 14B) | 0.1 |
| **BALTHASAR-2** | Dr. Akagi come **madre** — empatia e istinto protettivo | `Llama-4-Maverick` (Meta, MoE 128E) | 0.4 |
| **CASPER-3** | Dr. Akagi come **donna** — intuizione sociale e identità culturale | `Mistral-Medium-2505` (Mistral AI) | 0.7 |

### Fallback automatici
Ogni modello ha un fallback in caso di rate limit:

- Melchior: `Phi-4` → `Phi-4-mini-instruct`
- Balthasar: `Llama-4-Maverick` → `Llama-4-Scout`
- Casper: `Mistral-Medium-2505` → `Mistral-Small-2503`

---

## 🧩 Stack tecnologico

- **Backend:** Python, FastAPI, LangGraph, LangChain
- **Modelli:** GitHub Models (Azure AI inference endpoint)
- **Frontend:** HTML/CSS/JS (stile CRT, scanlines, font Share Tech Mono)
- **Logging:** JSON + Markdown in `logs/`

---

## ✅ Requisiti

- **Python 3.11+**
- **GitHub Personal Access Token** con accesso a GitHub Models

---

## ⚙️ Installazione

```bash
git clone https://github.com/DavideDelli/MAGI-Evangelion-Deliberation-Engine.git
cd MAGI-Evangelion-Deliberation-Engine

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
```

---

## 🔐 Configurazione

Copia il template e compila il token:

```bash
cp .env.example .env
```

```env
GITHUB_TOKEN=il_tuo_github_pat
```

---

## ▶️ Avvio in locale

```bash
PYTHONPATH=src uvicorn magi.api:app --reload --host 127.0.0.1 --port 8000
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="src"; uvicorn magi.api:app --reload --host 127.0.0.1 --port 8000
```

Apri: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🐳 Docker

```bash
docker compose up --build
```

Apri: [http://localhost:8000](http://localhost:8000)  
I log sono persistiti in `./logs`.

---

## 🧪 Test

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

---

## 🗂️ Log e output

Ogni deliberazione genera due file:

```
logs/
├── json/
│   └── YYYY-MM-DD/
│       └── magi_run_YYYYMMDD_HHMMSS.json
└── markdown/
    └── YYYY-MM-DD/
        └── magi_run_YYYYMMDD_HHMMSS.md
```

---

## 🧭 Struttura del progetto

```
MAGI-Evangelion-Deliberation-Engine/
├── src/
│   └── magi/
│       ├── __init__.py        # Package marker
│       ├── api.py             # FastAPI entry point
│       ├── agents.py          # LLM connections e fallback
│       ├── config.py          # Costanti e prompt
│       ├── graph.py           # Nodi LangGraph
│       ├── schemas.py         # Modelli Pydantic
│       └── utils.py           # Logging e utilities
├── frontend/
│   ├── templates/
│   │   └── magi_interface.html
│   └── static/
├── tests/
├── requirements.txt
├── .env.example
├── logs/
└── README.md
```

---

## 📚 Ispirazione

Nel 1995 i MAGI di *Neon Genesis Evangelion* erano tre supercomputer creati da Dr. Naoko Akagi, ognuno con una parte della sua personalità. Questo progetto ricrea quella struttura con LLM moderni, selezionati per riflettere la componente cognitiva ed emotiva di ogni nodo.

---

*NERV HQ — CENTRAL DOGMA — SECURITY LEVEL: OMEGA*
