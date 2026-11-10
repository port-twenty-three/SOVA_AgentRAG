#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

import asyncio
import json
import re
from typing import Any

import httpx
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from service.utils.config import settings
from service.utils.log import logger


class LLMClient:
    """
    Клиент для работы с DeepSeek API.
    Поддерживает синхронную генерацию, стриминг, retry-логику.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        self.model = model or settings.DEEPSEEK_MODEL
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.timeout = timeout

        if not self.api_key:
            logger.warning("DeepSeek API key not set! Set DEEPSEEK_API_KEY in .env")

        # HTTP клиент с пулом соединений
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """
        Отправляет запрос к DeepSeek API с автоматическим retry.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"DeepSeek API error {e.response.status_code}: {e.response.text}")
            raise
        except httpx.TimeoutException as e:
            logger.warning(f"DeepSeek API timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"DeepSeek API unexpected error: {e}")
            raise

    async def generate(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4000,  # <-- УВЕЛИЧЕНО: чтобы длинные таблицы не обрезались
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Генерирует ответ на вопрос на основе контекста.
        """
        # Стандартный системный промпт
        if system_prompt is None:
            system_prompt = (
                "Ты — корпоративный помощник, который отвечает на вопросы сотрудников. "
                "Отвечай простым, понятным человеческим языком — так, как объяснил бы "
                "коллеге. Избегай канцелярита, технических терминов и слов вроде "
                "«контекст», «источник данных», «база знаний», «документ». Говори по сути.\n\n"

                "Правила:\n"
                "1. Отвечай ТОЛЬКО на основе сведений, которые тебе передали вместе "
                "с вопросом. Ничего не придумывай и не додумывай.\n"
                "2. Если нужных сведений нет — прямо и по-человечески объясни, почему "
                "не можешь ответить. Например: «У меня нет информации об этом», "
                "«Этого нет в доступных мне материалах», «Я не нашёл ответа на этот "
                "вопрос». Не извиняйся excessively, не используй фразы вроде "
                "«в предоставленном контексте отсутствует» — это непонятно.\n"
                "3. Если ответ есть — дай его чётко и структурированно. Если аспектов "
                "несколько — разбей на пункты списком или нумерацией.\n"
                "4. В конце перечисли, откуда взята информация (названия файлов/"
                "разделов), если они тебе известны. Если нет — просто не добавляй "
                "этот блок.\n\n"

                "Обработка таблиц и чертежей:\n"
                "- Если в данных есть блок [ТАБЛИЦА], ты ОБЯЗАН вывести её в формате Markdown.\n"
                "- Если в данных есть блок [ИЗОБРАЖЕНИЕ/ЧЕРТЕЖ] с плейсхолдером (например, [IMAGE_1]), ты ДОЛЖЕН вывести этот плейсхолдер ровно так, как он написан.\n"
                "- КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать [IMAGE_1], [IMAGE_2] и т.д., если блока [ИЗОБРАЖЕНИЕ/ЧЕРТЕЖ] НЕТ в предоставленных данных"
                ". Если ты видишь в тексте упоминание 'рис. 18.1', но самого блока с картинкой нет, просто напиши 'См. рис. 18.1 в документе'.\n\n"
            )

        # Формируем промпт пользователя
        user_prompt = f"""
Ниже приведены данные, найденные в документации. Используй их, чтобы ответить на вопрос.

--- НАЧАЛО ДАННЫХ ---
{context}
--- КОНЕЦ ДАННЫХ ---

Вопрос пользователя:
{question}

Ответ:
"""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
            "response_format": {"type": "json_object"} if kwargs.get("json_mode") else None,
            **{k: v for k, v in kwargs.items() if k not in ["json_mode"]},
        }

        # Удаляем None значения
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            if stream:
                return await self._stream_generate(payload)
            else:
                response = await self._request("POST", "chat/completions", json=payload)

                # Извлекаем ответ
                answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Логируем использование токенов
                usage = response.get("usage", {})
                logger.info(
                    f"DeepSeek: {usage.get('prompt_tokens', 0)} prompt tokens, "
                    f"{usage.get('completion_tokens', 0)} completion tokens"
                )
                return answer

        except RetryError as e:
            logger.error(f"DeepSeek API retry failed: {e}")
            return "Извините, произошла ошибка при обращении к DeepSeek API. Попробуйте позже."

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return "Извините, произошла непредвиденная ошибка при генерации ответа."

    async def _stream_generate(self, payload: dict) -> str:
        """
        Генерация ответа с стримингом (для длинных ответов).
        """
        payload["stream"] = True
        full_answer = []

        async with self.client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
            response.raise_for_status()

            async for chunk in response.aiter_lines():
                if chunk.startswith("data: "):
                    data = chunk[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_answer.append(content)
                    except json.JSONDecodeError:
                        continue

        return "".join(full_answer)

    async def generate_with_retry(self, question: str, context: str, max_retries: int = 3, **kwargs) -> str:
        """Генерация ответа с ручным retry (если tenacity не справляется)."""
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.generate(question, context, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2**attempt)

        logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
        return "Извините, не удалось получить ответ. Пожалуйста, попробуйте позже."

    async def generate_structured(self, question: str, context: str, schema: dict, **kwargs) -> dict:
        """
        Генерирует структурированный ответ в формате JSON по схеме.
        """
        system_prompt = (
            "Ты — корпоративный помощник. Твоя задача — отвечать на вопросы, "
            "возвращая ответ в строгом JSON-формате по указанной схеме.\n\n"
            f"Схема ответа:\n{json.dumps(schema, indent=2, ensure_ascii=False)}"
        )

        try:
            response = await self.generate(
                question=question, context=context, system_prompt=system_prompt, json_mode=True, **kwargs
            )

            # Пробуем распарсить JSON
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Пробуем найти JSON в тексте
                json_match = re.search(r"\{.*}", response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError("Response is not valid JSON")

        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            return {"error": str(e)}

    async def close(self):
        """Закрывает HTTP-клиент."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ============================================
#  SINGLETON (один экземпляр на всё приложение)
# ============================================

_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient | None:
    """
    Возвращает синглтон-экземпляр LLM-клиента.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client