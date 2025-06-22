# Makefile для управления проектом анализа данных фрилансеров

.PHONY: help install dev build run test clean docker-build docker-run docker-stop lint format

# Цвета для вывода
GREEN=\033[0;32m
YELLOW=\033[1;33m
RED=\033[0;31m
NC=\033[0m # No Color

# Переменные
PYTHON=python3
PIP=pip3
DOCKER_IMAGE=freelancer-analyzer
DOCKER_TAG=latest

help: ## Показать эту справку
	@echo "${GREEN}Доступные команды:${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${YELLOW}%-15s${NC} %s\n", $$1, $$2}'

install: ## Установить зависимости
	@echo "${GREEN}Установка зависимостей...${NC}"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Установить зависимости для разработки
	@echo "${GREEN}Установка зависимостей для разработки...${NC}"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install black flake8 mypy pytest-cov

setup: ## Первоначальная настройка проекта
	@echo "${GREEN}Настройка проекта...${NC}"
	mkdir -p data logs models
	cp .env.example .env
	@echo "${YELLOW}Не забудьте отредактировать файл .env${NC}"

download-data: ## Скачать данные с Kaggle (требует kaggle API)
	@echo "${GREEN}Скачивание данных...${NC}"
	mkdir -p data
	kaggle datasets download -d shohinurpervezshohan/freelancer-earnings-and-job-trends -p data --unzip
	@echo "${GREEN}Данные загружены в папку data/${NC}"

dev: ## Запустить сервер разработки
	@echo "${GREEN}Запуск сервера разработки...${NC}"
	$(PYTHON) -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

run: ## Запустить приложение
	@echo "${GREEN}Запуск приложения...${NC}"
	$(PYTHON) main.py

cli: ## Запустить CLI интерфейс
	@echo "${GREEN}Запуск CLI...${NC}"
	$(PYTHON) -m src.cli_interface

test: ## Запустить тесты
	@echo "${GREEN}Запуск тестов...${NC}"
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Запустить тесты с покрытием
	@echo "${GREEN}Запуск тестов с анализом покрытия...${NC}"
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Проверить код линтерами
	@echo "${GREEN}Проверка кода...${NC}"
	flake8 src/ main.py
	mypy src/ main.py

format: ## Форматировать код
	@echo "${GREEN}Форматирование кода...${NC}"
	black src/ main.py tests/
	@echo "${GREEN}Код отформатирован${NC}"

clean: ## Очистить временные файлы
	@echo "${GREEN}Очистка временных файлов...${NC}"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete

# Docker команды
docker-build: ## Собрать Docker образ
	@echo "${GREEN}Сборка Docker образа...${NC}"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run: ## Запустить контейнер
	@echo "${GREEN}Запуск Docker контейнера...${NC}"
	docker run -d --name freelancer-analyzer \
		-p 8000:8000 \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-stop: ## Остановить контейнер
	@echo "${GREEN}Остановка Docker контейнера...${NC}"
	docker stop freelancer-analyzer || true
	docker rm freelancer-analyzer || true

docker-compose-up: ## Запустить с docker-compose
	@echo "${GREEN}Запуск с docker-compose...${NC}"
	docker-compose up -d

docker-compose-down: ## Остановить docker-compose
	@echo "${GREEN}Остановка docker-compose...${NC}"
	docker-compose down

docker-logs: ## Показать логи контейнера
	docker logs -f freelancer-analyzer

# Мониторинг и отладка
logs: ## Показать логи приложения
	@echo "${GREEN}Показ логов...${NC}"
	tail -f logs/app.log

health: ## Проверить здоровье приложения
	@echo "${GREEN}Проверка здоровья приложения...${NC}"
	curl -f http://localhost:8000/health || echo "${RED}Приложение недоступно${NC}"

check-gpu: ## Проверить GPU и совместимость
	@echo "${GREEN}Проверка GPU...${NC}"
	$(PYTHON) scripts/check_gpu.py

install-gpu: ## Установить зависимости с поддержкой GPU
	@echo "${GREEN}Установка зависимостей с поддержкой GPU...${NC}"
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install -r requirements.txt

info: ## Показать информацию о системе
	@echo "${GREEN}Информация о системе:${NC}"
	@echo "Python версия: $$($(PYTHON) --version)"
	@echo "pip версия: $$($(PIP) --version)"
	@echo "Docker версия: $$(docker --version 2>/dev/null || echo 'не установлен')"
	@echo "Доступная память: $$(free -h 2>/dev/null | grep Mem || echo 'неизвестно')"

# Документация
docs-serve: ## Запустить сервер документации
	@echo "${GREEN}Запуск сервера документации...${NC}"
	mkdocs serve

docs-build: ## Собрать документацию
	@echo "${GREEN}Сборка документации...${NC}"
	mkdocs build

# Развертывание
deploy-staging: ## Развернуть на staging
	@echo "${GREEN}Развертывание на staging...${NC}"
	docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

deploy-prod: ## Развернуть на production
	@echo "${GREEN}Развертывание на production...${NC}"
	docker-compose --profile production up -d

# Резервное копирование
backup: ## Создать резервную копию данных
	@echo "${GREEN}Создание резервной копии...${NC}"
	mkdir -p backups
	tar -czf backups/backup_$$(date +%Y%m%d_%H%M%S).tar.gz data/ logs/

# Полная очистка
purge: clean docker-stop ## Полная очистка (файлы + контейнеры)
	@echo "${GREEN}Полная очистка...${NC}"
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true
	docker volume prune -f

# Benchmark
benchmark: ## Запустить бенчмарки
	@echo "${GREEN}Запуск бенчмарков...${NC}"
	$(PYTHON) tests/benchmark.py