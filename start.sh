#!/bin/bash
cat << "EOF"

 ____   ___  _   _    _
/ ___| / _ \| | | |  / \
\___ \| | | | | | | / _ \
 ___) | |_| |\ V / / ___ \
|____/ \___/  \_/ /_/   \_\

  developer alakei1
EOF

echo ""
echo "  Сборка образа с обходом бага Docker Compose..."
python service/service/main.py

echo " Запуск контейнеров..."
docker compose up -d --no-build

echo ""
echo "======================================================================"
echo " ВСЁ ЗАПУЩЕНО! ДОСТУПНЫЕ СЕРВИСЫ:"
echo "======================================================================"
echo " Python API (Swagger UI): http://localhost:8000/docs"
echo " RedisInsight (БД):       http://localhost:5540"
echo "  MinIO (Хранилище):      http://localhost:9001  (Логин: minioadmin | Пароль: minioadmin)"
echo " Qdrant (Векторная БД):   http://localhost:6333/dashboard"
echo "======================================================================"

