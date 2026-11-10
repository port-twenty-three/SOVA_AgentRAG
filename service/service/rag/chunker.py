# ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from service.utils.config import settings


class TextChunker:
    """
    Рекурсивный сплиттер текста, который защищает таблицы от разрыва.
    """

    def __init__(
            self,
            chunk_size: int | None = None,
            chunk_overlap: int | None = None,
            separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size or getattr(settings, "CHUNK_SIZE", 500)
        self.chunk_overlap = chunk_overlap or getattr(settings, "CHUNK_OVERLAP", 50)
        self.separators = separators or getattr(settings, "CHUNK_SEPARATORS", ["\n\n", "\n", ". ", " ", ""])

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def _extract_tables(self, text: str) -> tuple[str, list[str]]:
        """
        Находит Markdown-таблицы, вырезает их и возвращает текст без таблиц + список таблиц.
        Регулярка ищет строки, начинающиеся с | и содержащие |---|.
        """
        # Паттерн для поиска Markdown-таблиц
        table_pattern = re.compile(
            r'((?:^\|.*\|[ \t]*\n)+)',
            re.MULTILINE
        )

        tables = table_pattern.findall(text)
        # Заменяем таблицы на плейсхолдеры, чтобы сплиттер не запутался
        text_without_tables = table_pattern.sub('\n\n[TABLE_PLACEHOLDER]\n\n', text)

        return text_without_tables, tables

    def split_text(self, text: str) -> list[str]:
        """Разбивает текст на чанки, сохраняя таблицы целиком."""
        if not text:
            return []

        # 1. Вырезаем таблицы
        text_no_tables, tables = self._extract_tables(text)

        # 2. Режем оставшийся текст
        chunks = self.splitter.split_text(text_no_tables)

        # 3. Возвращаем таблицы на место
        # Если чанк состоит только из плейсхолдера, заменяем его на оригинальную таблицу
        final_chunks = []
        table_idx = 0

        for chunk in chunks:
            if "[TABLE_PLACEHOLDER]" in chunk:
                # Если в чанке есть и текст, и плейсхолдер (редкий случай)
                # Можно разбить, но для простоты заменим плейсхолдер на саму таблицу
                parts = chunk.split("[TABLE_PLACEHOLDER]")
                for i, part in enumerate(parts):
                    if part.strip():
                        final_chunks.append(part.strip())
                    if i < len(parts) - 1 and table_idx < len(tables):
                        final_chunks.append(tables[table_idx].strip())
                        table_idx += 1
            else:
                if chunk.strip():
                    final_chunks.append(chunk.strip())

        # Если таблица была больше chunk_size, она всё равно сохранится целиком.
        # Если таблиц не было, вернется обычный список чанков.
        return final_chunks

    def split_documents(self, documents: list[dict]) -> list[dict]:
        chunks = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {}).copy()
            parts = self.split_text(text)
            for idx, part in enumerate(parts):
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = idx
                chunk_meta["total_chunks"] = len(parts)

                # Определяем тип контента (таблица или текст)
                content_type = "table" if part.strip().startswith("|") else "text"

                chunks.append(
                    {
                        "text": part,
                        "metadata": {**chunk_meta, "type": content_type},
                    }
                )
        return chunks