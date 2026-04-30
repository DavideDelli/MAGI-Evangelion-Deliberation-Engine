# 🔴 MAGI Supercomputer System

> *"The MAGI are three supercomputers that form the brain of NERV. They were built by Dr. Naoko Akagi and contain her personality engrams — as a scientist, as a mother, as a woman."*
> — Neon Genesis Evangelion

A multi-LLM deliberation system inspired by the MAGI supercomputers from *Neon Genesis Evangelion*. Three AI models, each embodying one aspect of Dr. Naoko Akagi's personality, independently analyze a dilemma and vote. Majority rules.

---

## Interface

### Input Screen
<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/13aa3128-78da-4fa8-8ed9-79df2f4d5189" />


### Deliberation Result
<img width="2501" height="1274" alt="image" src="https://github.com/user-attachments/assets/fa98873e-045e-44af-80e9-6ab512056bbb" />


---

## How It Works

Each deliberation runs three AI agents in **parallel** via LangGraph. Each agent receives the dilemma with a personality-locked system prompt and must conclude with an explicit `VOTO: SI` or `VOTO: NO`. An arbitration node counts the votes — majority wins. Results are saved as both JSON and Markdown logs.

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

## The Three Components

| Component | Personality | Model | Temperature |
|-----------|-------------|-------|-------------|
| **MELCHIOR-1** | Dr. Akagi as a **scientist** — cold logic, formal reasoning, quantifies everything | `Phi-4` (Microsoft, 14B) | 0.1 |
| **BALTHASAR-2** | Dr. Akagi as a **mother** — protective instinct, moral conflict, preservation of life | `Llama-4-Maverick` (Meta, MoE 128E) | 0.4 |
| **CASPER-3** | Dr. Akagi as a **woman** — social intuition, human relationships, cultural identity | `Mistral-Medium-2505` (Mistral AI) | 0.7 |

Each model was chosen deliberately: Phi-4's synthetic-data training makes it precise and deterministic (Melchior's cold logic); Llama-4 Maverick's MoE architecture gives it nuanced emotional depth (Balthasar's empathy); Mistral's European cultural sensibility makes it warmer and more narrative (Casper's social instinct).

Each primary model has a **fallback** in case daily rate limits are exhausted:

- Melchior: `Phi-4` → `Phi-4-mini-instruct`
- Balthasar: `Llama-4-Maverick` → `Llama-4-Scout`
- Casper: `Mistral-Medium-2505` → `Mistral-Small-2503`

---

## Stack

- **Backend:** Python, FastAPI, LangGraph, LangChain
- **Models:** GitHub Models (Azure AI inference endpoint)
- **Frontend:** Vanilla HTML/CSS/JS — CRT aesthetic, scanlines, Share Tech Mono font
- **Logging:** JSON + Markdown, organized by date under `logs/`

---

## Setup

### Prerequisites

- Python 3.11+
- A [GitHub Personal Access Token](https://github.com/settings/tokens) with access to GitHub Models

### Installation

```bash
git clone https://github.com/your-username/magi-system.git
cd magi-system

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GITHUB_TOKEN=your_github_pat_here
```

### Run

```bash
PYTHONPATH=src uvicorn magi.api:app --reload --host 127.0.0.1 --port 8000
```

> `PYTHONPATH=src` ensures Python can resolve the `magi` package from the `src/` layout.
>
> **Windows (PowerShell):**
> ```powershell
> $env:PYTHONPATH="src"; uvicorn magi.api:app --reload --host 127.0.0.1 --port 8000
> ```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### Docker (any OS)

Create a `.env` file as shown above (keep it local and never commit it), then run:

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Logs are persisted on the host in `./logs`.

---

## Log Format

Every deliberation is saved in two formats under `logs/`:

```
logs/
├── json/
│   └── YYYY-MM-DD/
│       └── magi_run_YYYYMMDD_HHMMSS.json
└── markdown/
    └── YYYY-MM-DD/
        └── magi_run_YYYYMMDD_HHMMSS.md
```

**JSON example:**
```json
{
  "timestamp": "2026-04-12T02:11:41",
  "dilemma": "Should the operation proceed?",
  "responses": {
    "melchior": { "model": "Phi-4", "vote": "NO", "elapsed_sec": 12.35 },
    "balthasar": { "model": "Llama-4-Maverick-...", "vote": "YES", "elapsed_sec": 3.34 },
    "casper":   { "model": "Mistral-Medium-2505", "vote": "YES", "elapsed_sec": 6.25 }
  },
  "final_decision": "APPROVED (2 to 1)"
}
```

---

## Project Structure

```
magi-system/
├── src/
│   └── magi/
│       ├── __init__.py        # Package marker
│       ├── api.py             # FastAPI entry point
│       ├── agents.py          # OpenAI API connections and failover logic
│       ├── config.py          # Global constants and LLM prompts
│       ├── graph.py           # LangGraph node definitions and edges
│       ├── schemas.py         # Pydantic models and TypedDicts
│       └── utils.py           # Helper functions (logging, regex)
├── frontend/
│   ├── templates/
│   │   └── magi_interface.html  # Frontend: NERV-style UI
│   └── static/                  # Static assets (if needed)
├── tests/                        # Test suite placeholder
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for environment variables
├── logs/                         # Output directory for deliberation logs
└── README.md
```

---

## Inspiration

The MAGI system in *Neon Genesis Evangelion* (1995) was a trio of supercomputers created by Dr. Naoko Akagi, each containing one aspect of her personality engram. In the show, they were used to make critical strategic and ethical decisions for NERV — often deadlocking 2-to-1 at the worst possible moments.

This project recreates that architecture using modern language models, with each model selected to match the cognitive and emotional profile of its corresponding component.

---

*NERV HQ — CENTRAL DOGMA — SECURITY LEVEL: OMEGA*
