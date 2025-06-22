"""
Конфигурация приложения
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Основные настройки приложения
    app_name: str = Field(default="Freelancer Data Analyzer", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Пути к файлам
    data_path: str = Field(default="./data/freelancer_earnings_bd.csv", env="DATA_PATH")
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    logs_dir: str = Field(default="./logs", env="LOGS_DIR")
    
    # FastAPI настройки
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # LLM настройки
    model_name: str = Field(default="microsoft/DialoGPT-medium", env="MODEL_NAME")
    use_cuda: bool = Field(default=True, env="USE_CUDA")
    max_length: int = Field(default=512, env="MAX_LENGTH")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Безопасность
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1"], env="ALLOWED_HOSTS")
    
    # База данных (если используется)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Redis (если используется)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # секунды
    
    # Мониторинг
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Производительность
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    timeout: int = Field(default=300, env="TIMEOUT")  # секунды
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Создание необходимых директорий"""
        dirs_to_create = [
            Path(self.model_cache_dir),
            Path(self.logs_dir),
            Path(self.data_path).parent
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def data_file_path(self) -> Path:
        """Путь к файлу данных"""
        return Path(self.data_path)
    
    @property
    def model_cache_path(self) -> Path:
        """Путь к кэшу моделей"""
        return Path(self.model_cache_dir)
    
    @property
    def logs_path(self) -> Path:
        """Путь к логам"""
        return Path(self.logs_dir)
    
    @property
    def is_production(self) -> bool:
        """Проверка production окружения"""
        return not self.debug and not self.reload
    
    def get_log_file_path(self, log_name: str = "app") -> Path:
        """Получение пути к файлу лога"""
        return self.logs_path / f"{log_name}.log"

# Глобальный экземпляр настроек
settings = Settings()

# Константы для анализа данных
class DataConstants:
    """Константы для обработки данных"""
    
    # Исходные колонки датасета (как в Kaggle)
    ORIGINAL_COLUMNS = {
        "FREELANCER_ID": "Freelancer_ID",
        "JOB_CATEGORY": "Job_Category", 
        "PLATFORM": "Platform",
        "EXPERIENCE_LEVEL": "Experience_Level",
        "CLIENT_REGION": "Client_Region",
        "PAYMENT_METHOD": "Payment_Method",
        "JOB_COMPLETED": "Job_Completed",
        "EARNINGS_USD": "Earnings_USD",
        "HOURLY_RATE": "Hourly_Rate",
        "JOB_SUCCESS_RATE": "Job_Success_Rate",
        "CLIENT_RATING": "Client_Rating",
        "JOB_DURATION_DAYS": "Job_Duration_Days",
        "PROJECT_TYPE": "Project_Type",
        "REHIRE_RATE": "Rehire_Rate",
        "MARKETING_SPEND": "Marketing_Spend"
    }
    
    # Унифицированные колонки (после обработки)
    EARNINGS_COLUMN = "earnings"
    PAYMENT_METHOD_COLUMN = "payment_method"
    LOCATION_COLUMN = "location"
    PROJECTS_COMPLETED_COLUMN = "projects_completed"
    SKILL_LEVEL_COLUMN = "skill_level"
    HOURLY_RATE_COLUMN = "hourly_rate"
    
    # Значения уровней навыков (возможные варианты в данных)
    SKILL_LEVELS = {
        "EXPERT": ["expert", "advanced", "senior"],
        "INTERMEDIATE": ["intermediate", "middle", "mid-level"],
        "JUNIOR": ["junior", "beginner", "entry-level"],
        "ENTRY": ["entry", "new", "trainee"]
    }
    
    # Способы оплаты (возможные варианты)
    PAYMENT_METHODS = {
        "CRYPTO": ["cryptocurrency", "crypto", "bitcoin", "btc"],
        "BANK": ["bank_transfer", "bank", "wire", "banking"],
        "PAYPAL": ["paypal", "pp"],
        "ESCROW": ["escrow"],
        "CASH": ["cash", "direct"]
    }
    
    # Пороги для категоризации проектов
    PROJECT_COUNT_THRESHOLDS = {
        "BEGINNER": (0, 10),
        "EXPERIENCED": (10, 50), 
        "PROFESSIONAL": (50, 100),
        "EXPERT": (100, float('inf'))
    }

# Константы для LLM
class LLMConstants:
    """Константы для языковой модели"""
    
    # Альтернативные модели
    ALTERNATIVE_MODELS = [
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill",
        "microsoft/DialoGPT-small",
        "distilbert-base-uncased"
    ]
    
    # Типы запросов
    QUERY_TYPES = {
        "COMPARISON": "comparison",
        "DISTRIBUTION": "distribution", 
        "PERCENTAGE": "percentage",
        "STATISTICS": "statistics",
        "GENERAL": "general"
    }
    
    # Ключевые слова для классификации
    COMPARISON_KEYWORDS = [
        "сравнить", "сравнение", "выше", "ниже", "разница", 
        "больше", "меньше", "против", "vs"
    ]
    
    DISTRIBUTION_KEYWORDS = [
        "распределение", "по регионам", "по областям", "география",
        "как распределяется", "распределить"
    ]
    
    PERCENTAGE_KEYWORDS = [
        "процент", "%", "доля", "сколько", "какой процент",
        "какая доля"
    ]
    
    STATISTICS_KEYWORDS = [
        "средний", "среднее", "медиана", "статистика",
        "максимальный", "минимальный"
    ]

# Настройки API
class APIConstants:
    """Константы для API"""
    
    # HTTP статус коды
    HTTP_OK = 200
    HTTP_BAD_REQUEST = 400
    HTTP_NOT_FOUND = 404
    HTTP_INTERNAL_ERROR = 500
    HTTP_SERVICE_UNAVAILABLE = 503
    
    # Заголовки
    CONTENT_TYPE_JSON = "application/json"
    CONTENT_TYPE_TEXT = "text/plain"
    
    # Лимиты запросов
    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 3
    
    # Таймауты
    DEFAULT_TIMEOUT = 30
    LONG_TIMEOUT = 120