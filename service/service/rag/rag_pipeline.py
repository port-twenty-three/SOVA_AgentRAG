#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

import asyncio
import time
import uuid
import re  # Перенесли импорт наверх
from datetime import timedelta
from typing import Any, Optional

from miniopy_async import Minio
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from service.rag.chunker import TextChunker
from service.rag.embedder import Embedder
from service.rag.llm_client import LLMClient
from service.rag.vector_store import VectorStore
from service.utils.config import settings
from service.utils.log import logger


class RAGPipeline:
    def __init__(self):
        self.chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=settings.CHUNK_SEPARATORS,
        )
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()

        self.minio = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

    def _collection_exists(self, name: str) -> bool:
        if not name or not name.strip():
            return False
        try:
            return bool(self.vector_store.client.collection_exists(name))
        except Exception as exc:
            logger.warning("collection_exists check failed for %r: %s", name, exc)
            return False

    @staticmethod
    def _normalize_query_vector(raw: Any) -> Optional[list[float]]:
        if not raw:
            return None
        first = raw[0]
        if isinstance(first, (list, tuple)):
            if not first:
                return None
            return list(first)
        return list(raw)

    async def run_from_text(
            self,
            text: str,
            source_name: str,
            collection_name: str,
            tenant_id: str,
            content_type: str = "text",
            image_url: str | None = None,
    ) -> int:
        if not collection_name or not collection_name.strip():
            logger.error("run_from_text: пустое имя коллекции")
            return 0

        # --- ИСПРАВЛЕНИЕ: Таблицы не режем! ---
        if content_type == "table":
            chunks = [text]
        else:
            chunks = await asyncio.to_thread(self.chunker.split_text, text)
        # --------------------------------------

        if not chunks:
            return 0

        exists = await asyncio.to_thread(self._collection_exists, collection_name)
        if not exists:
            await asyncio.to_thread(
                self.vector_store.create_collection,
                collection_name,
                self.embedder.vector_size,
            )

        vectors = await asyncio.to_thread(self.embedder.embed, chunks)
        if not vectors or len(vectors) != len(chunks):
            logger.error("Embedding failed or size mismatch")
            return 0

        points = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk,
                "source": source_name,
                "tenant_id": tenant_id,
                "collection": collection_name,
                "type": content_type,
            }
            if image_url:
                payload["image_url"] = image_url

            points.append({
                "id": point_id,
                "vector": vec,
                "payload": payload,
            })

        await asyncio.to_thread(self.vector_store.upsert_points, collection_name, points)
        logger.info("Indexed %d chunks (%s) from %s", len(points), content_type, source_name)
        return len(points)

    async def query(
            self,
            query_text: str,
            collection_name: str,
            tenant_id: str,
            top_k: int = 5,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def _empty(answer: str) -> dict[str, Any]:
            return {
                "answer": answer, "sources": [], "context": "", "total_chunks": 0,
                "average_score": 0.0, "top_k_requested": top_k,
                "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        if not collection_name or not collection_name.strip():
            return _empty("Не указано имя коллекции")

        exists = await asyncio.to_thread(self._collection_exists, collection_name)
        if not exists:
            return _empty(f"Коллекция '{collection_name}' не найдена. Сначала проиндексируйте документы.")

        raw_vector = await asyncio.to_thread(self.embedder.embed, query_text)
        single_query_vector = self._normalize_query_vector(raw_vector)
        if single_query_vector is None:
            return _empty("Ошибка эмбеддинга запроса")

        filter_cond = models.Filter(
            must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                models.FieldCondition(key="collection", match=models.MatchValue(value=collection_name)),
            ]
        )

        try:
            # ИСПРАВЛЕНИЕ: Увеличили лимит и снизили порог, чтобы находило картинки
            results = await asyncio.to_thread(
                self.vector_store.search,
                collection_name=collection_name,
                query_vector=single_query_vector,
                limit=max(top_k, 15),
                filter_conditions=filter_cond,
                score_threshold=0.25,
            )
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) == 404:
                return _empty(f"Коллекция '{collection_name}' не найдена.")
            logger.exception("query: ошибка Qdrant при поиске")
            return _empty("Ошибка векторного поиска")

        # --- СБОР КОНТЕКСТА С ПЛЕЙСХОЛДЕРАМИ ---
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        image_map = {}
        img_counter = 1

        for scored in results:
            payload = scored.payload or {}
            text = payload.get("text")
            if not text:
                continue

            c_type = payload.get("type", "text")
            img_url = payload.get("image_url", "")
            final_img_url = ""

            if c_type == "image" and img_url:
                if img_url.startswith("http"):
                    final_img_url = img_url
                else:
                    try:
                        bucket_name, object_name = img_url.split("/", 1)
                        final_img_url = await self.minio.presigned_get_object(
                            bucket_name=bucket_name,
                            object_name=object_name,
                            expires=timedelta(days=7)
                        )
                        public_endpoint = getattr(settings, "MINIO_PUBLIC_ENDPOINT", None)
                        if public_endpoint:
                            final_img_url = final_img_url.replace(settings.MINIO_ENDPOINT, public_endpoint)
                    except Exception as e:
                        logger.error(f"Ошибка генерации Presigned URL для {img_url}: {e}")
                        final_img_url = ""

                if final_img_url:
                    placeholder = f"[IMAGE_{img_counter}]"
                    image_map[placeholder] = final_img_url
                    context_parts.append(
                        f"[ИЗОБРАЖЕНИЕ/ЧЕРТЕЖ]\nОписание: {text}\nПлейсхолдер: {placeholder}"
                    )
                    img_counter += 1
                else:
                    context_parts.append(f"[ИЗОБРАЖЕНИЕ/ЧЕРТЕЖ]\nОписание: {text}\n(Ссылка недоступна)")

            elif c_type == "table":
                context_parts.append(f"[ТАБЛИЦА]\n{text}")
            else:
                context_parts.append(text)

            sources.append({
                "id": scored.id,
                "score": scored.score,
                "source": payload.get("source", "unknown"),
                "text": text,
                "type": c_type,
                "image_url": final_img_url
            })

        total = len(sources)
        avg_score = (sum(s["score"] for s in sources) / total if total > 0 else 0.0)
        context = ("\n\n---\n\n".join(context_parts) if context_parts else "Информация не найдена.")

        # Генерируем ответ
        answer = await self.llm_client.generate(
            question=query_text,
            context=context,
            temperature=0.2,
        )

        # --- ЖЕСТКАЯ ОЧИСТКА ОТВЕТА НЕЙРОСЕТИ ---
        # 1. Удаляем ЛЮБЫЕ картинки Markdown, которые нейросеть придумала сама
        answer = re.sub(r'!\[.*?\]\(.*?\)', '', answer)
        # 2. Удаляем "голые" ссылки
        answer = re.sub(r'https?://[^\s]+', '', answer)
        # 3. Убираем пустые скобки
        answer = answer.replace("[]()", "").replace("[]", "")

        # --- ПОДСТАНОВКА РЕАЛЬНЫХ ССЫЛОК ---
        for placeholder, real_url in image_map.items():
            answer = answer.replace(placeholder, f"![Чертеж]({real_url})")

        # ЗАЩИТА: Если плейсхолдер не заменился (картинки нет), пишем текст, а не мусор
        answer = re.sub(r'\[IMAGE_\d+\]', '*(Чертеж недоступен)*', answer)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "total_chunks": total,
            "average_score": round(avg_score, 4),
            "top_k_requested": top_k,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    async def delete_document(
            self, collection_name: str, tenant_id: str, file_name: str
    ) -> int:
        if not collection_name or not collection_name.strip():
            return 0

        exists = await asyncio.to_thread(self._collection_exists, collection_name)
        if not exists:
            return 0

        point_ids: list[Any] = []
        offset = None
        while True:
            try:
                result = await asyncio.to_thread(
                    self.vector_store.client.scroll,
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(key="collection", match=models.MatchValue(value=collection_name)),
                            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                            models.FieldCondition(key="source", match=models.MatchValue(value=file_name)),
                        ]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
            except UnexpectedResponse as exc:
                if getattr(exc, "status_code", None) == 404:
                    return 0
                raise

            points_batch = result.points
            if not points_batch:
                break
            point_ids.extend(p.id for p in points_batch)
            offset = result.next_page_offset
            if offset is None:
                break

        if point_ids:
            await asyncio.to_thread(self.vector_store.delete_points, collection_name, point_ids)

        logger.info("Удалено %d чанков для документа %s", len(point_ids), file_name)
        return len(point_ids)