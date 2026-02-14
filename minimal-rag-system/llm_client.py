import requests
import re
import os
lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")
lm_studio_model = os.environ.get("LM_STUDIO_MODEL", "lm")

def chat_llm(prompt: str, system: str = "You are a helpful assistant.") -> str:
    url = f"{lm_studio_url}/v1/chat/completions"

    payload = {
        "model": lm_studio_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.2,
    }

    r = requests.post(url, json=payload, timeout=100)
    r.raise_for_status()
    data = r.json()

    answer = data["choices"][0]["message"]["content"]

    answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL).strip()

    return answer