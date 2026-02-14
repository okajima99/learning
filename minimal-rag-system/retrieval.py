from typing import List,Dict
from preprocess import text_normalize
from rank_bm25 import BM25Okapi

def bigram_tokenizer(text:str) -> List:
    if text in {" ","　"}:
        text_s = text.replace(" ","").replace("　","")
    else:
        text_s = text
    result = []
    for t in range(len(text_s) - 1):
        token = text_s[t:t+2]
        result.append(token)
    return result

def retrieve(chunks:List[Dict], query: str, k:int = 3) ->List[Dict]:
    query_norm = text_normalize(query)
    query_token = bigram_tokenizer(query_norm)
    corpus = []
    for chunk in chunks:
        text = chunk["chunk_text"]
        text_token = bigram_tokenizer(text)
        corpus.append(text_token)
    bm25_text = BM25Okapi(corpus)
    score = bm25_text.get_scores(query_token)
    score_list = []
    for s in range(len(score)):
        score_list.append({
            "title":chunks[s]["title"],
            "chunk_id":s,
            "chunk_text":chunks[s]["chunk_text"],
            "source":chunks[s]["source"],
            "score":score[s]
        })
    return sorted(
        score_list,
        key= lambda x:-(x["score"]),
    )[:k]
