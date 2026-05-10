from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import sys
sys.path.append('../')
from utils.model import ToxicityModel

app      = FastAPI()
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN  = 64
MODEL_NAME = "distilbert-base-uncased"


model = ToxicityModel(MODEL_NAME).to(DEVICE)
model.load_state_dict(torch.load(
    "../models/distilbert_toxicity_best.pt",
    map_location=DEVICE, weights_only=True
))
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Model loaded on {DEVICE}")

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(req: PredictRequest):
    enc = tokenizer(
        req.text, return_tensors="pt",
        max_length=MAX_LEN, truncation=True, padding="max_length"
    )
    with torch.no_grad():
        score, _, _ = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE)
        )
    return {"score": score.item()}

# Run with: uvicorn inference_server:app --port 8000