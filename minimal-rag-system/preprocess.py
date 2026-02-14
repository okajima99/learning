from typing import List,Dict

def text_normalize(text:str):
    if not isinstance(text,str):
        raise ValueError("textがstrでないもしくはNoneです")
    text_norm = text.strip().lower()
    return text_norm

def docs_text_normalize(docs:List[Dict]):
    if docs is None:
        raise ValueError("docsがNoneです")
    result = []
    for d in docs:
       doc_text = d["text"]
       doc_text_norm = text_normalize(doc_text)
       result.append({
           "title":d["title"],
           "text_norm":doc_text_norm,
           "source":d["source"]
       })
    return result

def docs_chunk(docs:List[Dict], chunk_size: int = 300):
    if docs is None:
        raise ValueError("docsがNoneです")
    result = []
    chunk_id = 0
    for d in docs:
        text = d["text_norm"]
        n = len(text)
        slip = 0
        while slip < n:
            end = min(slip + chunk_size, n)
            chunk_text = text[slip:end]
            idx = chunk_text.rfind("。")
            if idx == -1:
                result.append({
                    "title":d["title"],
                    "chunk_id":chunk_id,
                    "chunk_text":chunk_text,
                    "source":d["source"]
                })
                slip = end 
            else:
                cut_text = chunk_text[:idx + 1]
                result.append({
                    "title":d["title"],
                    "chunk_id":chunk_id,
                    "chunk_text":cut_text,
                    "source":d["source"]
                })
                slip += len(cut_text)

            chunk_id += 1
    return result

