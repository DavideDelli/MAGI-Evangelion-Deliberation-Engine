<div align="center">

# 🔴 MAGI Supercomputer System

**A multi‑LLM deliberation engine inspired by the MAGI supercomputers from *Neon Genesis Evangelion*.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Ready-009688?logo=fastapi&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Orchestrator-8A2BE2">
</p>

> *"The MAGI are three supercomputers that form the brain of NERV. They were built by Dr. Naoko Akagi and contain her personality engrams — as a scientist, as a mother, as a woman."*  
> — *Neon Genesis Evangelion*

</div>

---

## ✨ Overview

This project recreates the MAGI architecture: three LLM agents with distinct personalities analyze a dilemma in parallel, vote (`VOTE: YES` or `VOTE: NO`), and an arbitration node selects the majority outcome. Results are saved as **JSON** and **Markdown**.

---

## 🖥️ Interface

### Input screen
<img width="1280" height="800" alt="Input" src="https://github.com/user-attachments/assets/13aa3128-78da-4fa8-8ed9-79df2f4d5189" />

### Deliberation result
<img width="2501" height="1274" alt="Result" src="https://github.com/user-attachments/assets/fa98873e-045e-44af-80e9-6ab512056bbb" />

---

## 🧠 How it works

Each deliberation runs three agents in **parallel** with LangGraph. Each agent receives the dilemma with a personality‑locked system prompt and ends with an explicit vote (`VOTE: YES` or `VOTE: NO`). An arbitration node counts votes and determines the final verdict.

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

## 🧬 MAGI components

| Component | Personality | Model | Temperature |
|-----------|-------------|-------|-------------|
| **MELCHIOR-1** | Dr. Akagi as a **scientist** — cold logic and formal reasoning | `Phi-4` (Microsoft, 14B) | 0.1 |
| **BALTHASAR-2** | Dr. Akagi as a **mother** — empathy and protective instinct | `Llama-4-Maverick` (Meta, MoE 128E) | 0.4 |
| **CASPER-3** | Dr. Akagi as a **woman** — social intuition and cultural identity | `Mistral-Medium-2505` (Mistral AI) | 0.7 |

### Automatic fallbacks
Each primary model has a fallback in case of rate limits:

- Melchior: `Phi-4` → `Phi-4-mini-instruct`
- Balthasar: `Llama-4-Maverick` → `Llama-4-Scout`
- Casper: `Mistral-Medium-2505` → `Mistral-Small-2503`

---

## 🧩 Tech stack

- **Backend:** Python, FastAPI, LangGraph, LangChain
- **Models:** GitHub Models (Azure AI inference endpoint)
- **Frontend:** HTML/CSS/JS (CRT aesthetic, scanlines, Share Tech Mono font)
- **Logging:** JSON + Markdown in `logs/`

---

## ✅ Requirements

- **Python 3.11+**
- **GitHub Personal Access Token** with access to GitHub Models

---

## ⚙️ Installation

```bash
git clone https://github.com/DavideDelli/MAGI-Evangelion-Deliberation-Engine.git
cd MAGI-Evangelion-Deliberation-Engine

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
```

---

## 🔐 Configuration

Copy the template and fill in your token:

```bash
cp .env.example .env
```

```env
GITHUB_TOKEN=your_github_pat
```

---

## ▶️ Run locally

```bash
PYTHONPATH=src uvicorn magi.api:app --reload --host 127.0.0.1 --port 8000
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="src"; uvicorn magi.api:app --reload --host 127.0.0.1 --port 8000
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🐳 Docker

```bash
docker compose up --build
```

Open: [http://localhost:8000](http://localhost:8000)  
Logs are persisted in `./logs`.

---

## 🧪 Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

---

## 🗂️ Logs and output

Each deliberation generates two files:

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

## 🧭 Project structure

```
MAGI-Evangelion-Deliberation-Engine/
├── src/
│   └── magi/
│       ├── __init__.py        # Package marker
│       ├── api.py             # FastAPI entry point
│       ├── agents.py          # LLM connections and fallback logic
│       ├── config.py          # Constants and prompts
│       ├── graph.py           # LangGraph nodes
│       ├── schemas.py         # Pydantic models
│       └── utils.py           # Logging and utilities
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

## 📚 Inspiration

In *Neon Genesis Evangelion* (1995), the MAGI were three supercomputers created by Dr. Naoko Akagi, each containing a facet of her personality. This project recreates that structure with modern LLMs, selected to reflect the cognitive and emotional profile of each node.

---

*NERV HQ — CENTRAL DOGMA — SECURITY LEVEL: OMEGA*
