from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

app = FastAPI()
# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class EncodeRequest(BaseModel):
    id: int
    texts: list
    is_tokenized: bool = False

@app.post("/encode")
async def encode(req: EncodeRequest):
    # Encode text using BERT
    encoded_input = tokenizer(req.texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Get sentence embeddings (using CLS token)
    embeddings = model_output.last_hidden_state[:, 0, :].numpy()
    
    # Ensure proper shape for scikit-learn (2D array)
    result = [embedding.reshape(1, -1).tolist() for embedding in embeddings]
    return {"result": result}