import requests
import os

def main():
    url = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000/ask")
    payload = {"query": "推薦アルゴリズム"}  # ここ変えて試す

    r = requests.post(url, json=payload, timeout=500)
    print("status:", r.status_code)
    print(r.json())

if __name__ == "__main__":
    main()