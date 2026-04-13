import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from schemas import DilemmaRequest
from graph import magi_system
from utils import extract_vote

app = FastAPI(title="MAGI Deliberation API")

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
        "melchior_vote":         extract_vote(result["melchior_response"]),
        "balthasar_vote":        extract_vote(result["balthasar_response"]),
        "casper_vote":           extract_vote(result["casper_response"]),
        "melchior_text":         result["melchior_response"],
        "balthasar_text":        result["balthasar_response"],
        "casper_text":           result["casper_response"],
        "melchior_model_used":   result.get("melchior_model_used", "?"),
        "balthasar_model_used":  result.get("balthasar_model_used", "?"),
        "casper_model_used":     result.get("casper_model_used", "?"),
        "final_decision":        result["final_decision"],
    }

if __name__ == "__main__":
    print("🌐 Starting MAGI server on port 8000...")
    # Ora avvii 'main:app' invece che il vecchio file
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)