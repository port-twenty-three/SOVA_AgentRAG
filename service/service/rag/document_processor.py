import asyncio
import io
import os
import uuid
from typing import Any

import fitz  # PyMuPDF
from docx import Document
from miniopy_async import Minio
from miniopy_async.error import MinioException

from service.utils.config import settings
from service.utils.log import logger


class DocumentProcessor:
    def __init__(self):
        self.minio = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        logger.info("DocumentProcessor: инициализирован асинхронный клиент MinIO")

    @staticmethod
    def _table_to_markdown(table_data: list[list[Any]]) -> str:
        if not table_data: return ""
        headers = [str(cell).replace("\n", " ").strip() if cell else "" for cell in table_data[0]]
        md = "| " + " | ".join(headers) + " |\n"
        md += "|" + "|".join(["---"] * len(headers)) + "|\n"
        for row in table_data[1:]:
            row_text = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
            md += "| " + " | ".join(row_text) + " |\n"
        return md

    @staticmethod
    def _extract_content_from_bytes(file_data: bytes, file_extension: str) -> list[dict]:
        blocks = []
        try:
            if file_extension == ".pdf":
                with fitz.open(stream=file_data, filetype="pdf") as doc:
                    for page_num, page in enumerate(doc):
                        text = page.get_text()
                        if text.strip():
                            blocks.append({"type": "text", "text": text})

                        try:
                            tabs = page.find_tables()
                            for i, tab in enumerate(tabs):
                                md_table = DocumentProcessor._table_to_markdown(tab.extract())
                                if md_table:
                                    header = f"Таблица {i + 1} на странице {page_num + 1}. "
                                    blocks.append({"type": "table", "text": header + "\n\n" + md_table})
                        except Exception as e:
                            logger.warning(f"Не удалось извлечь таблицу со стр. {page_num + 1}: {e}")

                        # --- ИСПРАВЛЕНИЕ: Привязываем текст страницы к картинке ---
                        # Берем первые 300 символов текста, чтобы поиск находил картинку по смыслу
                        page_text_snippet = text[:300].replace("\n", " ") if text else ""

                        for img_index, img in enumerate(page.get_images(full=True)):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            if base_image:
                                blocks.append({
                                    "type": "image",
                                    "ext": base_image["ext"],
                                    "bytes": base_image["image"],
                                    # Теперь описание содержит слова из текста, и поиск сработает!
                                    "text": f"Иллюстрация (страница {page_num + 1}). Текст рядом: {page_text_snippet}"
                                })
                        # --------------------------------------------------------

            elif file_extension == ".docx":
                doc_stream = io.BytesIO(file_data)
                doc = Document(doc_stream)

                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if text:
                    blocks.append({"type": "text", "text": text})

                for i, table in enumerate(doc.tables):
                    table_data = [[cell.text for cell in row.cells] for row in table.rows]
                    md_table = DocumentProcessor._table_to_markdown(table_data)
                    if md_table:
                        header = f"Таблица {i + 1}. "
                        blocks.append({"type": "table", "text": header + "\n\n" + md_table})
            return blocks
        except Exception as e:
            logger.error(f"Ошибка парсинга {file_extension}: {e}")
            return []

    async def process_file_from_minio(
            self, bucket_name: str, object_name: str, collection_name: str, tenant_id: str, rag_pipeline: Any
    ) -> int:
        response = None
        try:
            logger.info(f"Обработка файла {object_name} из {bucket_name}")

            if not await self.minio.bucket_exists(bucket_name):
                await self.minio.make_bucket(bucket_name)
                logger.info(f"Создан бакет {bucket_name}")

            response = await self.minio.get_object(bucket_name, object_name)
            max_size = getattr(settings, "MAX_FILE_SIZE_BYTES", 50 * 1024 * 1024)
            file_data = await response.read()
            if len(file_data) > max_size:
                logger.error(f"Файл {object_name} превышает лимит {max_size} байт")
                return 0

            _, ext = os.path.splitext(object_name.lower())
            if ext not in (".pdf", ".docx"):
                return 0

            blocks = await asyncio.to_thread(self._extract_content_from_bytes, file_data, ext)
            if not blocks:
                return 0

            total_chunks = 0

            for block in blocks:
                if block["type"] == "text":
                    chunks = await rag_pipeline.run_from_text(
                        text=block["text"], source_name=object_name,
                        collection_name=collection_name, tenant_id=tenant_id, content_type="text"
                    )
                    total_chunks += chunks

                elif block["type"] == "table":
                    chunks = await rag_pipeline.run_from_text(
                        text=block["text"], source_name=object_name,
                        collection_name=collection_name, tenant_id=tenant_id, content_type="table"
                    )
                    total_chunks += chunks

                elif block["type"] == "image":
                    img_ext = block["ext"]
                    img_name = f"images/{uuid.uuid4().hex}.{img_ext}"
                    img_bytes = block["bytes"]

                    try:
                        await self.minio.put_object(
                            bucket_name, img_name, io.BytesIO(img_bytes), len(img_bytes),
                            content_type=f"image/{img_ext}"
                        )

                        image_path = f"{bucket_name}/{img_name}"

                        chunks = await rag_pipeline.run_from_text(
                            text=block["text"], source_name=object_name,
                            collection_name=collection_name, tenant_id=tenant_id,
                            content_type="image", image_url=image_path
                        )
                        total_chunks += chunks
                    except Exception as e:
                        logger.error(f"Ошибка загрузки картинки {img_name} в MinIO: {e}")

            logger.info(f"Файл {object_name} обработан, создано чанков: {total_chunks}")
            return total_chunks

        except MinioException as e:
            logger.error(f"Ошибка MinIO при обработке {object_name}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Критическая ошибка при обработке {object_name}: {e}", exc_info=True)
            return 0
        finally:
            if response:
                response.close()

    async def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        try:
            cleaned_bucket_name = bucket_name.replace("_", "-").lower()
            cleaned_bucket_name = "".join(
                c for c in cleaned_bucket_name if c.isalnum() or c == "-"
            )
            if len(cleaned_bucket_name) < 3:
                cleaned_bucket_name = f"bucket-{cleaned_bucket_name}" if cleaned_bucket_name else "bucket-default"

            if not await self.minio.bucket_exists(cleaned_bucket_name):
                await self.minio.make_bucket(cleaned_bucket_name)
                logger.info(f"Создан бакет {cleaned_bucket_name}")

            await self.minio.fput_object(cleaned_bucket_name, object_name, file_path)
            logger.info(f"Файл {file_path} загружен как {object_name} в бакет {cleaned_bucket_name}")
            return True
        except MinioException as e:
            logger.error(f"Ошибка загрузки файла {file_path} в MinIO: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке {file_path}: {e}", exc_info=True)
            return False

    async def file_exists_in_minio(self, bucket_name: str, object_name: str) -> bool:
        try:
            await self.minio.stat_object(bucket_name, object_name)
            return True
        except MinioException as e:
            if e.code == "NoSuchKey":
                return False
            logger.error(f"Ошибка при проверке существования {object_name} в {bucket_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при проверке существования {object_name}: {e}")
            return False

    async def delete_file_from_minio(self, bucket_name: str, object_name: str) -> bool:
        try:
            await self.minio.remove_object(bucket_name, object_name)
            logger.info(f"Файл {object_name} удалён из бакета {bucket_name}")
            return True
        except MinioException as e:
            logger.error(f"Ошибка удаления {object_name} из {bucket_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при удалении {object_name}: {e}")
            return False