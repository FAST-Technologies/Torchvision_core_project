#!/usr/bin/env python3
"""Генерация сводного отчёта по тестам."""
import subprocess
import json
import sys
from pathlib import Path


def check_json_plugin():
    """Проверяет наличие плагина pytest-json-report."""
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "--help"], capture_output=True, text=True, check=True)
        return "--json-report" in result.stdout
    except subprocess.CalledProcessError:
        return False


def run_tests_with_json():
    """Запуск тестов с JSON-отчётом."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-m",
        "not slow",
        "--json-report",
        "--json-report-file=report.json",
        "--cov=segmenters",
        "--cov-report=json:coverage.json",
        "-v",  # подробный вывод для парсинга при необходимости
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Вывод стандартного вывода/ошибок для отладки
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)

    return result.returncode == 0


def run_tests_basic():
    """Запуск тестов без JSON-плагина (фоллбэк)."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-m",
        "not slow",
        "--cov=segmenters",
        "--cov-report=term",  # вывод в терминал
        "-v",
        "--tb=short",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def parse_basic_output():
    """Парсинг базового вывода pytest (если JSON недоступен)."""
    # Запускаем pytest с --collect-only для получения списка тестов
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-m", "not slow", "--collect-only", "-q"],
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().split("\n")
    # Последняя строка обычно содержит итог: "X passed, Y failed, Z skipped"
    summary_line = [l for l in lines if "passed" in l or "failed" in l or "skipped" in l]

    if summary_line:
        print(f"📋 Сводка: {summary_line[-1]}")
    else:
        print("⚠️  Не удалось распарсить сводку")


def main():
    print("🔍 Проверка окружения...")

    if check_json_plugin():
        print("✅ Плагин pytest-json-report обнаружен")
        success = run_tests_with_json()

        if not success:
            print("❌ Тесты завершились с ошибками")
            return 1

        # Парсинг JSON-отчёта
        try:
            with open("report.json") as f:
                report = json.load(f)

            summary = report.get("summary", {})
            print(f"\n{'='*60}")
            print(f"✅ Пройдено: {summary.get('passed', 0)}")
            print(f"❌ Провалено: {summary.get('failed', 0)}")
            print(f"⏭  Пропущено: {summary.get('skipped', 0)}")
            print(f"⏱  Всего: {summary.get('total', 0)}")
            print(f"{'='*60}")

            if summary.get("failed", 0) > 0:
                print("\n❗ Детали ошибок:")
                for test in report.get("tests", []):
                    if test.get("outcome") == "failed":
                        nodeid = test.get("nodeid", "unknown")
                        # Безопасное извлечение сообщения об ошибке
                        crash = test.get("call", {}).get("crash", {})
                        message = crash.get("message", "No message")
                        print(f"  • {nodeid}: {message[:200]}")

        except FileNotFoundError:
            print("❌ Файл report.json не найден")
            return 1
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            return 1

    else:
        print("⚠️  Плагин pytest-json-report не установлен")
        print("💡 Установите: pip install pytest-json-report")
        print("\n🔄 Запуск тестов в базовом режиме...")

        success = run_tests_basic()
        parse_basic_output()

        if not success:
            print("\n❌ Тесты завершились с ошибками")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
