import random
import time
from typing import List

import httpx

from app import config


class NotispaceClient:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {config.NOTISPACE_API_KEY}",
            "Content-Type": "application/json",
        }
        self.base_url = config.NOTISPACE_BASE_URL.rstrip("/")

    def _post_with_retry(self, client: httpx.Client, url: str, payload: dict):
        max_retries = config.EMBED_MAX_RETRIES
        base_delay = config.EMBED_RETRY_BASE_DELAY_SECONDS

        for attempt in range(max_retries + 1):
            resp = client.post(url, headers=self.headers, json=payload)

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == max_retries:
                    resp.raise_for_status()

                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = base_delay * (2 ** attempt)
                else:
                    delay = base_delay * (2 ** attempt)
                delay += random.uniform(0, delay * 0.25)  # jitter

                time.sleep(delay)
                continue

            resp.raise_for_status()
            return resp

        # Unreachable, but keeps type-checkers happy.
        raise RuntimeError("gagal memanggil NotiSpace setelah retry")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        url = self.base_url + "/embeddings"
        batch_size = max(1, config.EMBED_BATCH_SIZE)
        batch_delay = config.EMBED_BATCH_DELAY_SECONDS

        result: List[List[float]] = []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        with httpx.Client(timeout=60.0) as client:
            for i, batch in enumerate(batches):
                payload = {"model": config.NOTISPACE_EMBED_MODEL, "input": batch}
                resp = self._post_with_retry(client, url, payload)
                data = resp.json()
                vectors = [item["embedding"] for item in data["data"]]
                result.extend(vectors)

                # small pacing gap between batches so we don't immediately
                # slam back into the rate limit on the very next request
                if batch_delay > 0 and i < len(batches) - 1:
                    time.sleep(batch_delay)

        return result

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": config.NOTISPACE_CHAT_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        with httpx.Client(timeout=60.0) as client:
            resp = self._post_with_retry(client, url, payload)
            data = resp.json()
            return data["choices"][0]["message"]["content"]


notispace_client = NotispaceClient()
