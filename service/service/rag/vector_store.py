#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

from typing import Any, cast

from qdrant_client import QdrantClient, models

from service.utils.config import settings
from service.utils.log import logger


class VectorStore:
    """
    Клиент для работы с векторным хранилищем Qdrant.

    Предоставляет методы для создания коллекций, вставки/обновления точек,
    поиска с фильтрацией, удаления и получения точек по ID.
    """

    def __init__(self) -> None:
        """
        Инициализирует подключение к Qdrant.

        Параметры подключения берутся из глобального конфига (settings):
        - QDRANT_HOST
        - QDRANT_PORT
        - QDRANT_GRPC_PORT (используется как grpc_port)
        - QDRANT_PREFER_GRPC
        """
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
        )

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """
        Создаёт коллекцию, если она ещё не существует.
        Включает On-Disk хранение векторов и payload для предотвращения OOM.
        """
        if self.client.collection_exists(collection_name):
            logger.debug(f"Коллекция {collection_name} уже существует")
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=getattr(models.Distance, distance.upper()),
                # НАСТРОЙКА 1: Хранить сами векторы на диске, а не в RAM
                on_disk=True,
            ),
            # НАСТРОЙКА 2: Хранить метаданные (payload) на диске
            hnsw_config=models.HnswConfigDiff(on_disk=True),
        )

        # Payload-индексы. Без них фильтрация по этим полям
        # (в query и delete_document) будет full-scan'ом.
        for field_name in ("tenant_id", "collection", "source"):
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        logger.debug(
            f"Коллекция {collection_name} создана с On-Disk индексами "
            f"(payload indexes: tenant_id, collection, source)"
        )

    def upsert_points(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
    ) -> None:
        qdrant_points = [
            models.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]

        # wait=False: Qdrant принимает батч в буфер и индексирует в фоне,
        # что сглаживает пиковые нагрузки на RAM и CPU.
        self.client.upsert(
            collection_name=collection_name,
            points=qdrant_points,
            wait=False,
        )
        logger.debug(f"Добавлено/Обновлено {len(points)} точек в {collection_name}")

    @staticmethod
    def _build_filter(
        filter_conditions: dict | models.Filter | None,
    ) -> models.Filter | None:
        """
        Нормализует filter_conditions к models.Filter.

        Принимает:
        - None            -> None (без фильтрации)
        - models.Filter   -> как есть
        - dict            -> models.Filter(**dict) (pydantic рекурсивно
                             разберёт вложенные dict в FieldCondition и т.д.)
        """
        if filter_conditions is None:
            return None
        if isinstance(filter_conditions, models.Filter):
            return filter_conditions
        if isinstance(filter_conditions, dict):
            return models.Filter(**filter_conditions)
        raise TypeError(
            "filter_conditions must be dict | models.Filter | None, "
            f"got {type(filter_conditions).__name__}"
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_conditions: dict | models.Filter | None = None,
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        """
        Выполняет поиск ближайших соседей по вектору с возможной фильтрацией.

        Аргументы:
            collection_name (str): Имя коллекции.
            query_vector (list[float]): Вектор запроса (одномерный).
            limit (int): Максимальное количество возвращаемых результатов.
            filter_conditions (dict | models.Filter | None): Условия фильтрации.
                Поддерживаются оба формата:
                  * dict: {"must": [{"key": "tenant_id", "match": {"value": "123"}}]}
                  * models.Filter(...)
            score_threshold (float | None): Минимальный порог score.

        Возвращает:
            list[models.ScoredPoint]: Найденные точки (id, vector, payload, score).
        """
        qdrant_filter = self._build_filter(filter_conditions)

        response = self.client.query_points(
            collection_name=collection_name,
            query=cast(Any, query_vector),
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return response.points

    def delete_points(self, collection_name: str, point_ids: list[str]) -> None:
        """
        Удаляет точки из коллекции по списку идентификаторов.

        Аргументы:
            collection_name (str): Имя коллекции.
            point_ids (list[str]): Список ID точек для удаления.
        """
        if not point_ids:
            return
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )
        logger.info(f"Удалено {len(point_ids)} точек из {collection_name}")

    def get_point(self, collection_name: str, point_id: str) -> dict | None:
        """
        Получает одну точку по её ID вместе с payload.

        Аргументы:
            collection_name (str): Имя коллекции.
            point_id (str): Идентификатор точки.

        Возвращает:
            dict | None: Словарь с данными точки (id, vector, payload)
                         или None, если точка не найдена.
        """
        results = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
        )
        return results[0].model_dump() if results else None