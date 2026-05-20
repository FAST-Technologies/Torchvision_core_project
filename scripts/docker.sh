#!/bin/bash
set -e

case "$1" in
  dev)
    echo "🚀 Запуск в режиме разработки..."
    docker compose --env-file .env.dev up --build
    ;;
  prod)
    echo "📦 Запуск в продакшене..."
    docker compose --env-file .env.prod up -d --build
    ;;
  test)
    echo "🧪 Запуск тестов..."
    docker compose run --rm backend pytest "${@:2}"
    ;;
  logs)
    echo "📋 Логи: ${2:-all}"
    docker compose logs -f "${2:-}"
    ;;
  shell)
    echo "🐚 Shell в ${2:-backend}"
    docker compose exec "${2:-backend}" /bin/bash
    ;;
  *)
    echo "Использование: $0 {dev|prod|test|logs|shell} [args...]"
    exit 1
    ;;
esac