from fastapi import FastAPI, HTTPException, Request
import json
import torch
from data import process_data
from embedding import Embedding
from index import Index
from generate import Generate
from langchain.chains import RetrievalQA

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.post("/rag")
async def rag(request: Request):
    try:
        data = await request.json()
        text = process_data(data)

        # Model embed
        model_embed = Embedding(model_embedding_name='keepitreal/vietnamese-sbert', device=device, batch_size=32)
        vectorstore = Index(text).load_build_index(embed_model=model_embed())

        # LLM
        llm = Generate(llm='TinyLlama/TinyLlama-1.1B-Chat-v0.6')
        query = data.get('query', '')
        result = rag_pipeline(query)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
