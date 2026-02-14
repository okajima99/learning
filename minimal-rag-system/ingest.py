from pathlib import Path
from typing import List,Dict
from pypdf import PdfReader

def text_load(doc_loot: Path) -> str:

    if doc_loot.suffix in {".txt",".md"}:
        return doc_loot.read_text(encording = "utf-8")
    
    if doc_loot.suffix == ".pdf":
        r = PdfReader(doc_loot)
        texts = []
        for page in r.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
        return "\n".join(texts)
    
def load_doc_text(docs_loot: Path) -> List[Dict]:
    if docs_loot is None:
        raise ValueError("Pathがありません")
    if not isinstance(docs_loot,Path):
        raise ValueError("Pathの形式で入力されていません")
    if not docs_loot:
        raise ValueError("Pathがありません")
    
    docs = []
    for path in docs_loot.glob("*"):
        if path.suffix in {".txt",".md",".pdf"}:
            doc_text = text_load(path)
            doc_path = str(path)
            doc_p = doc_path.rfind(".")
            doc_s = doc_path.rfind("/")
            doc_title = doc_path[doc_s + 1:doc_p]
            if doc_text.strip():
                docs.append({
                    "title":doc_title,
                    "text":doc_text,
                    "source":doc_path
                })
    return docs
