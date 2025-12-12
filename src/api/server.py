from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.pipeline import run_evaluation
import tempfile
import json
import os

app = FastAPI()

class EvalRequest(BaseModel):
    conversation: list
    context_vectors: list

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    tmp_conv = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_ctx = tempfile.NamedTemporaryFile(delete=False, suffix=".json")

    json.dump(req.conversation, open(tmp_conv.name, "w"))
    json.dump({"chunks": req.context_vectors}, open(tmp_ctx.name, "w"))

    result = run_evaluation(tmp_conv.name, tmp_ctx.name)

    os.unlink(tmp_conv.name)
    os.unlink(tmp_ctx.name)

    return result
