from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import List,Dict
from ingest import load_doc_text
from preprocess import text_normalize ,docs_text_normalize ,docs_chunk
from retrieval import retrieve
from llm_client import chat_llm
import os

docs_root = Path(os.environ.get("DOC_ROOT", "docs")).expanduser()

app = FastAPI()

class ask_request(BaseModel):
    query:str

class ask_response(BaseModel):
    query:str
    text:str
    note:str

@app.get("/health")
def health() -> Dict:
    return {"status":"ok"}

@app.post("/ask")
def ask(req : ask_request):
    query_norm = text_normalize(req.query)
    docs_i = load_doc_text(docs_root)
    docs_norm = docs_text_normalize(docs_i)
    docs_chunks = docs_chunk(docs_norm)
    retrieve_docs = retrieve(docs_chunks,query_norm)
    rag_context = "\n".join(text["chunk_text"] for text in retrieve_docs)
    prompt = (
        "次のコンテキストだけを根拠に質問に答えてください。\n"
        "コンテキストに無い内容は「不明」と言ってください。\n\n"
        f"【コンテキスト】\n{rag_context}\n\n"
        f"【質問】\n{req.query}\n"
    )
    llm_answer = chat_llm(prompt)
    return ask_response(query = req.query, text= llm_answer, note = "answer ok")
