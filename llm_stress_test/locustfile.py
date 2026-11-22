import os
import time
import random
from locust import HttpUser, task, between

API_KEY = os.getenv("OPENWEBUI_API_KEY", "sk-c90f5bcb524b4f188b8204b06d752d01")
MODEL_NAME = os.getenv("OPENWEBUI_MODEL", "gemma4:latest")

PROMPTS = [
    "인공지능의 역전파 알고리즘을 3단계로 요약해줘.",
    "데이터베이스 정규화의 목적과 제1~3정규형을 간단히 설명해줘.",
    "객체지향 프로그래밍의 4대 특징을 예시와 함께 설명해줘."
]

class OpenWebUIStudent(HttpUser):
    # 학생이 문제를 읽고 다음 질문을 보내기까지의 대기 시간 (초)
    wait_time = between(5, 15)

    @task
    def chat_stream(self):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
            "stream": True
        }

        start_time = time.perf_counter()
        ttft_recorded = False

        with self.client.post(
            "/api/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            catch_response=True,
            name="Chat Completion (Total)"
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:100]}")
                return

            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data = decoded[6:].strip()
                        if data == "[DONE]":
                            break
                        # 첫 토큰 수신 시 TTFT를 별도 지표로 기록
                        if not ttft_recorded:
                            ttft_ms = (time.perf_counter() - start_time) * 1000
                            self.environment.events.request.fire(
                                request_type="METRIC",
                                name="TTFT (First Token)",
                                response_time=ttft_ms,
                                response_length=len(data),
                                exception=None,
                            )
                            ttft_recorded = True
                response.success()
            except Exception as e:
                response.failure(f"Stream Error: {e}")