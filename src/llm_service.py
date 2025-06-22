"""
Сервис интеграции с языковой моделью для анализа запросов
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import asyncio
import json
import re

logger = logging.getLogger(__name__)

class LLMService:
    """Сервис для работы с языковой моделью"""
    
    def __init__(self):
        # Используем более легкую модель для лучшей производительности
        self.model_name = "microsoft/DialoGPT-medium"
        # Альтернативные модели для fallback
        self.alternative_models = [
            "microsoft/DialoGPT-small",
            "gpt2",
            "distilgpt2"
        ]
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.ready = False
        
        # Улучшенная детекция GPU
        self.device = self._detect_best_device()
        logger.info(f"Выбрано устройство для вычислений: {self.device}")
        
        # Настройки для генерации
        self.generation_config = {
            "max_new_tokens": 400,
            "temperature": 0.8,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "pad_token_id": None  # Установим позже
        }
        
    def _detect_best_device(self) -> str:
        """Определение лучшего доступного устройства"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown GPU"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
            
            logger.info(f"Обнаружено GPU: {gpu_name}")
            logger.info(f"Доступная GPU память: {gpu_memory:.1f} GB")
            logger.info(f"Количество доступных GPU: {gpu_count}")
            
            # Проверяем, достаточно ли памяти для модели
            if gpu_memory >= 2.0:  # Минимум 2GB для DialoGPT-medium
                return "cuda"
            else:
                logger.warning(f"Недостаточно GPU памяти ({gpu_memory:.1f}GB < 2.0GB), используем CPU")
                return "cpu"
        else:
            logger.info("CUDA недоступна, используем CPU")
            return "cpu"
        
    async def initialize(self):
        """Инициализация языковой модели"""
        try:
            logger.info(f"Загрузка языковой модели {self.model_name}...")
            
            # Загружаем модель асинхронно
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self._load_model)
            
            if success:
                self.ready = True
                logger.info(f"Языковая модель успешно загружена на устройство: {self.device}")
                
                # Тестируем модель
                await self._test_model()
            else:
                # Пробуем альтернативные модели
                await self._try_alternative_models()
            
        except Exception as e:
            logger.error(f"Ошибка загрузки языковой модели: {e}")
            # Пробуем альтернативные модели
            await self._try_alternative_models()
    
    def _load_model(self) -> bool:
        """Загрузка модели и токенизатора"""
        try:
            logger.info(f"Загрузка токенизатора {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Добавляем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Добавлен pad_token в токенизатор")
            
            # Устанавливаем pad_token_id в конфигурацию генерации
            self.generation_config["pad_token_id"] = self.tokenizer.eos_token_id
            
            logger.info(f"Загрузка модели {self.model_name} на устройство {self.device}...")
            
            # Настройки для разных устройств
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,  # Используем половинную точность для экономии памяти
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                logger.info("Модель загружена на GPU с половинной точностью")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                logger.info("Модель загружена на CPU")
            
            # Создаем pipeline для генерации текста
            device_id = 0 if self.device == "cuda" else -1
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                truncation=True
            )
            
            logger.info(f"Pipeline создан успешно на устройстве: {self.device}")
            
            # Тестовая генерация для проверки работы
            if self.device == "cuda":
                self._test_gpu_inference()
            
            return True
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {self.model_name}: {e}")
            return False
    
    def _test_gpu_inference(self):
        """Тестирование работы модели на GPU"""
        try:
            test_input = "Тест GPU"
            with torch.no_grad():
                result = self.pipeline(test_input, **{
                    "max_new_tokens": 20,
                    "temperature": 0.7,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id
                })
            logger.info("✅ GPU инференс работает корректно")
            
            # Показываем использование GPU памяти
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                logger.info(f"Использование GPU памяти: {memory_allocated:.2f}GB выделено, {memory_reserved:.2f}GB зарезервировано")
                
        except Exception as e:
            logger.warning(f"Ошибка тестирования GPU: {e}")
            # Переключаемся на CPU в случае проблем с GPU
            self.device = "cpu"
            self._fallback_to_cpu()
    
    def _fallback_to_cpu(self):
        """Переключение на CPU в случае проблем с GPU"""
        try:
            logger.warning("Переключение на CPU...")
            
            # Очищаем GPU память
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Перезагружаем модель на CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Пересоздаем pipeline для CPU
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                truncation=True
            )
            
            logger.info("Успешно переключились на CPU")
            
        except Exception as e:
            logger.error(f"Ошибка переключения на CPU: {e}")
            self.pipeline = None
    
    async def _try_alternative_models(self):
        """Попытка загрузить альтернативные модели"""
        for model_name in self.alternative_models:
            try:
                logger.info(f"Попытка загрузки альтернативной модели: {model_name}")
                self.model_name = model_name
                
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(None, self._load_model)
                
                if success:
                    self.ready = True
                    logger.info(f"Альтернативная модель {model_name} успешно загружена")
                    await self._test_model()
                    return
                    
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель {model_name}: {e}")
                continue
        
        # Если ни одна модель не загрузилась
        logger.warning("Ни одна модель не загрузилась, используется режим без LLM")
        self.ready = True
        self.pipeline = None
    
    async def _test_model(self):
        """Тестирование модели после загрузки"""
        if not self.pipeline:
            logger.warning("Pipeline не создан, пропускаем тест модели")
            return
        
        try:
            logger.info("Тестирование модели...")
            test_prompt = "Привет, как дела?"
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.pipeline(test_prompt, max_new_tokens=20, do_sample=False)
            )
            
            if result and len(result) > 0:
                response = result[0].get('generated_text', '')
                logger.info(f"Тест модели успешен. Результат: {response[:100]}...")
            else:
                logger.warning("Тест модели не дал результата")
                
        except Exception as e:
            logger.error(f"Ошибка тестирования модели: {e}")
            self.pipeline = None
    
    def is_ready(self) -> bool:
        """Проверка готовности сервиса"""
        return self.ready
    
    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Анализ намерения пользователя в запросе"""
        logger.info(f"Анализ запроса: {query}")
        
        # Базовый анализ ключевых слов (работает без LLM)
        query_lower = query.lower()
        intent = self._classify_intent_by_keywords(query_lower)
        
        # Если доступна LLM, используем её для улучшения анализа
        if self.pipeline:
            try:
                enhanced_intent = await self._enhance_intent_with_llm(query, intent)
                intent.update(enhanced_intent)
            except Exception as e:
                logger.warning(f"Ошибка использования LLM для анализа: {e}")
        
        return intent
    
    def _classify_intent_by_keywords(self, query: str) -> Dict[str, Any]:
        """Классификация намерения на основе ключевых слов для 10 вопросов"""
        intent = {
            "query_type": "general",
            "entities": [],
            "metrics": [],
            "filters": {},
            "confidence": 0.8,
            "question_id": None
        }
        
        # Определяем конкретный вопрос по ключевым фразам
        if "криптовалют" in query and "выше доход" in query:
            intent["question_id"] = 1
            intent["query_type"] = "comparison"
            intent["entities"] = ["cryptocurrency"]
            intent["metrics"] = ["earnings"]
            intent["filters"]["payment_method"] = "cryptocurrency"
            
        elif "распределяется доход" in query and "регион" in query:
            intent["question_id"] = 2
            intent["query_type"] = "distribution"
            intent["entities"] = ["location"]
            intent["metrics"] = ["earnings"]
            
        elif "процент" in query and "эксперт" in query and "100 проект" in query:
            intent["question_id"] = 3
            intent["query_type"] = "percentage"
            intent["entities"] = ["expert", "projects"]
            intent["metrics"] = ["projects_completed"]
            intent["filters"]["skill_level"] = "expert"
            
        elif "средняя стоимость проект" in query and "опыт" in query:
            intent["question_id"] = 4
            intent["query_type"] = "statistics"
            intent["entities"] = ["experience"]
            intent["metrics"] = ["avg_project_value"]
            
        elif "регион" in query and ("больше всего" in query or "меньше всего" in query):
            intent["question_id"] = 5
            intent["query_type"] = "distribution"
            intent["entities"] = ["location"]
            intent["metrics"] = ["earnings"]
            
        elif "количество" in query and "проект" in query and "влияет" in query and "доход" in query:
            intent["question_id"] = 6
            intent["query_type"] = "correlation"
            intent["entities"] = ["projects"]
            intent["metrics"] = ["earnings", "projects_completed"]
            
        elif "процент" in query and "способ оплаты" in query:
            intent["question_id"] = 7
            intent["query_type"] = "percentage"
            intent["entities"] = ["payment_method"]
            intent["metrics"] = ["payment_method"]
            
        elif "разница" in query and "новичк" in query and "эксперт" in query and "регион" in query:
            intent["question_id"] = 8
            intent["query_type"] = "multifactor"
            intent["entities"] = ["expert", "beginner", "location"]
            intent["metrics"] = ["earnings"]
            
        # ИСПРАВЛЕНО: Улучшенное определение вопроса 9 о почасовой ставке
        elif (("средняя почасовая ставка" in query or "почасовая ставка" in query or "ставка" in query) 
              and "способ оплаты" in query):
            intent["question_id"] = 9
            intent["query_type"] = "hourly_rate_analysis"  # Изменен тип запроса
            intent["entities"] = ["payment_method"]
            intent["metrics"] = ["hourly_rate"]
            
        elif "сколько" in query and "проект" in query and "квалификац" in query:
            intent["question_id"] = 10
            intent["query_type"] = "statistics"
            intent["entities"] = ["skill_level"]
            intent["metrics"] = ["projects_completed"]
        
        # Если не распознали конкретный вопрос, используем общую логику
        if intent["question_id"] is None:
            # Общая классификация типа запроса
            if any(word in query for word in ["сравнить", "сравнение", "выше", "ниже", "разница", "больше", "меньше"]):
                intent["query_type"] = "comparison"
                
            elif any(word in query for word in ["распределение", "по регионам", "по областям", "география", "регион"]):
                intent["query_type"] = "distribution"
                
            elif any(word in query for word in ["процент", "%", "доля", "сколько", "какой процент"]):
                intent["query_type"] = "percentage"
                
            elif any(word in query for word in ["средний", "среднее", "медиана", "статистика"]):
                intent["query_type"] = "statistics"
                
            # ДОБАВЛЕНО: Обнаружение почасовых ставок в общем случае
            elif any(word in query for word in ["почасовая", "ставка", "час", "hourly"]):
                intent["query_type"] = "hourly_rate_analysis"
                intent["metrics"].append("hourly_rate")
            
            # Извлечение сущностей
            entities = []
            
            # Способы оплаты
            if any(word in query for word in ["криптовалют", "крипто", "cryptocurrency", "bitcoin"]):
                entities.append("cryptocurrency")
                intent["filters"]["payment_method"] = "cryptocurrency"
                
            if any(word in query for word in ["банк", "bank", "перевод", "transfer"]):
                entities.append("bank_transfer")
                intent["filters"]["payment_method"] = "bank"
                
            if any(word in query for word in ["paypal", "пэйпал"]):
                entities.append("paypal")
                intent["filters"]["payment_method"] = "paypal"
            
            # Уровни экспертизы
            if any(word in query for word in ["эксперт", "экспертов", "expert", "продвинут", "опытн"]):
                entities.append("expert")
                intent["filters"]["skill_level"] = "expert"
                
            if any(word in query for word in ["новичок", "beginner", "начинающ", "junior"]):
                entities.append("beginner")
                intent["filters"]["skill_level"] = "beginner"
            
            # Регионы/локации
            if any(word in query for word in ["регион", "регионов", "область", "страна", "location", "география"]):
                entities.append("location")
                
            # Проекты
            if any(word in query for word in ["проект", "проектов", "работ", "заказ"]):
                entities.append("projects")
                
            intent["entities"] = entities
            
            # Извлечение метрик
            metrics = []
            if any(word in query for word in ["доход", "заработок", "зарплат", "earnings", "деньги", "оплат"]):
                metrics.append("earnings")
                
            if any(word in query for word in ["проект", "работ", "заказ", "completed"]):
                metrics.append("projects_completed")
                
            if any(word in query for word in ["рейтинг", "rating", "оценк"]):
                metrics.append("rating")
                
            if any(word in query for word in ["час", "hourly", "ставк", "почасов"]):
                metrics.append("hourly_rate")
                
            intent["metrics"] = metrics
        
        logger.info(f"Определен intent: question_id={intent.get('question_id')}, query_type={intent.get('query_type')}")
        return intent
    
    async def _enhance_intent_with_llm(self, query: str, base_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Улучшение анализа намерения с помощью LLM"""
        if not self.pipeline:
            return {}
        
        # Создаем промпт для анализа намерения
        prompt = f"""Проанализируй вопрос о фрилансерах и определи:
1. Тип анализа: сравнение, статистика, распределение, процент
2. Ключевые сущности: регион, способ оплаты, уровень экспертизы, платформа
3. Метрики: доход, количество проектов, рейтинг

Вопрос: "{query}"

Ответ должен быть кратким и структурированным."""
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.pipeline(prompt, **{
                    "max_new_tokens": 100,
                    "temperature": 0.3,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id
                })
            )
            
            if response and len(response) > 0:
                generated_text = response[0].get('generated_text', '').strip()
                # Попытка извлечь полезную информацию из ответа LLM
                enhanced_data = self._extract_intent_from_response(generated_text)
                return enhanced_data if enhanced_data else {}
            
            return {}
            
        except Exception as e:
            logger.error(f"Ошибка обработки LLM для анализа намерения: {e}")
            return {}
    
    def _extract_intent_from_response(self, response: str) -> Optional[Dict]:
        """Извлечение структурированной информации из ответа LLM"""
        try:
            # Простое извлечение ключевых слов из ответа
            enhanced = {}
            
            response_lower = response.lower()
            
            # Определение типа анализа
            if "сравнение" in response_lower or "comparison" in response_lower:
                enhanced["llm_query_type"] = "comparison"
            elif "статистика" in response_lower or "statistics" in response_lower:
                enhanced["llm_query_type"] = "statistics"
            elif "распределение" in response_lower or "distribution" in response_lower:
                enhanced["llm_query_type"] = "distribution"
            elif "процент" in response_lower or "percentage" in response_lower:
                enhanced["llm_query_type"] = "percentage"
            
            return enhanced if enhanced else None
        except:
            return None
    
    async def generate_answer(self, query: str, data_analysis: Dict[str, Any]) -> str:
        """Генерация интеллектуального ответа на основе данных"""
        
        # Если LLM доступна, используем её для полного анализа
        if self.pipeline:
            try:
                logger.info("Генерация ответа с использованием LLM...")
                llm_answer = await self._generate_intelligent_answer(query, data_analysis)
                if llm_answer and len(llm_answer.strip()) > 50:
                    logger.info("LLM успешно сгенерировал ответ")
                    return llm_answer
                else:
                    logger.warning("LLM сгенерировал слишком короткий ответ, используем fallback")
            except Exception as e:
                logger.error(f"Ошибка генерации LLM ответа: {e}")
        
        # Fallback на аналитический ответ
        logger.info("Использование аналитического fallback")
        return self._generate_analytical_fallback(query, data_analysis)
    
    async def _generate_intelligent_answer(self, query: str, data_analysis: Dict[str, Any]) -> str:
        """Генерация интеллектуального ответа с LLM"""
        
        # Подготавливаем контекст данных
        data_context = self._prepare_data_context(data_analysis)
        
        # Создаем детальный промпт для анализа
        prompt = self._create_analysis_prompt(query, data_context)
        
        try:
            logger.info(f"Отправка промпта в LLM (длина: {len(prompt)} символов)")
            
            # Генерируем ответ
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._safe_generate(prompt)
            )
            
            if result:
                # Очищаем и форматируем ответ
                cleaned_answer = self._clean_llm_response(result, query)
                if cleaned_answer and len(cleaned_answer) > 50:
                    logger.info(f"LLM ответ получен (длина: {len(cleaned_answer)} символов)")
                    return cleaned_answer
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при генерации интеллектуального ответа: {e}")
            return None
    
    def _safe_generate(self, prompt: str) -> str:
        """Безопасная генерация текста с обработкой ошибок"""
        try:
            result = self.pipeline(
                prompt,
                max_new_tokens=self.generation_config["max_new_tokens"],
                temperature=self.generation_config["temperature"],
                do_sample=self.generation_config["do_sample"],
                top_p=self.generation_config["top_p"],
                top_k=self.generation_config["top_k"],
                repetition_penalty=self.generation_config["repetition_penalty"],
                no_repeat_ngram_size=self.generation_config["no_repeat_ngram_size"],
                pad_token_id=self.generation_config["pad_token_id"]
            )
            
            if result and len(result) > 0:
                return result[0].get('generated_text', '')
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка safe_generate: {e}")
            return None
    
    def _create_analysis_prompt(self, query: str, data_context: str) -> str:
        """Создание промпта для анализа данных"""
        
        prompt = f"""Ты профессиональный аналитик данных о фрилансерах. Проанализируй предоставленные данные и ответь на вопрос пользователя.

ВОПРОС: {query}

ДАННЫЕ ДЛЯ АНАЛИЗА:
{data_context}

ИНСТРУКЦИИ:
1. Внимательно изучи данные
2. Найди закономерности и тренды
3. Дай конкретный ответ с числами
4. Сделай практические выводы
5. Сравни показатели если нужно
6. Ответ на русском языке
7. Будь конкретным и информативным

АНАЛИЗ:"""
        
        return prompt
    
    def _prepare_data_context(self, data_analysis: Dict[str, Any]) -> str:
        """Подготовка контекста данных для LLM"""
        context_parts = []
        
        # Общая статистика
        if "summary" in data_analysis:
            summary = data_analysis["summary"]
            context_parts.append(f"""ОБЩАЯ СТАТИСТИКА:
- Всего фрилансеров: {summary.get('total_records', 0)}
- Средний доход: ${summary.get('mean_earnings', 0):.2f}
- Медианный доход: ${summary.get('median_earnings', 0):.2f}
- Диапазон: ${summary.get('min_earnings', 0):.2f} - ${summary.get('max_earnings', 0):.2f}""")
        
        # ДОБАВЛЕНО: Почасовые ставки по способам оплаты
        if "hourly_rate_by_payment" in data_analysis:
            hourly_data = data_analysis["hourly_rate_by_payment"]
            if "mean" in hourly_data:
                context_parts.append("ПОЧАСОВЫЕ СТАВКИ ПО СПОСОБАМ ОПЛАТЫ:")
                sorted_methods = sorted(hourly_data["mean"].items(), key=lambda x: x[1], reverse=True)
                for method, rate in sorted_methods:
                    count = hourly_data.get("count", {}).get(method, "?")
                    context_parts.append(f"- {method}: ${rate:.2f}/час (количество: {count})")
        
        # Доходы по способам оплаты
        if "earnings_by_payment_method" in data_analysis:
            earnings_data = data_analysis["earnings_by_payment_method"]
            if "mean" in earnings_data:
                context_parts.append("ДОХОДЫ ПО СПОСОБАМ ОПЛАТЫ:")
                sorted_methods = sorted(earnings_data["mean"].items(), key=lambda x: x[1], reverse=True)
                for method, income in sorted_methods:
                    count = earnings_data.get("count", {}).get(method, "?")
                    context_parts.append(f"- {method}: ${income:.2f} (количество: {count})")
        
        # Доходы по регионам
        if "regional_earnings" in data_analysis:
            regional_data = data_analysis["regional_earnings"]
            if "mean" in regional_data:
                context_parts.append("ДОХОДЫ ПО РЕГИОНАМ:")
                sorted_regions = sorted(regional_data["mean"].items(), key=lambda x: x[1], reverse=True)
                for region, income in sorted_regions[:8]:  # Топ 8
                    count = regional_data.get("count", {}).get(region, "?")
                    context_parts.append(f"- {region}: ${income:.2f} (количество: {count})")
        
        # Доходы по уровню навыков
        if "experience_earnings" in data_analysis:
            exp_data = data_analysis["experience_earnings"]
            if "mean" in exp_data:
                context_parts.append("ДОХОДЫ ПО УРОВНЮ НАВЫКОВ:")
                sorted_skills = sorted(exp_data["mean"].items(), key=lambda x: x[1], reverse=True)
                for skill, income in sorted_skills:
                    count = exp_data.get("count", {}).get(skill, "?")
                    context_parts.append(f"- {skill}: ${income:.2f} (количество: {count})")
        
        # Статистика экспертов
        if "total_experts" in data_analysis:
            experts_under_100 = data_analysis.get("experts_under_100_projects", 0)
            percentage = data_analysis.get("percentage_under_100_projects", 0)
            total_experts = data_analysis.get("total_experts", 0)
            
            context_parts.append(f"""СТАТИСТИКА ЭКСПЕРТОВ:
- Всего экспертов: {total_experts}
- С менее 100 проектами: {experts_under_100} ({percentage:.1f}%)
- Со 100+ проектами: {total_experts - experts_under_100} ({100-percentage:.1f}%)""")
        
        return "\n\n".join(context_parts) if context_parts else "Данные для анализа недоступны"
    
    def _clean_llm_response(self, response: str, original_query: str) -> str:
        """Очистка и форматирование ответа LLM"""
        try:
            cleaned = response.strip()
            
            # Удаляем повторение промпта
            if "АНАЛИЗ:" in cleaned:
                parts = cleaned.split("АНАЛИЗ:")
                if len(parts) > 1:
                    cleaned = parts[-1].strip()
            
            # Удаляем лишние префиксы
            prefixes_to_remove = ["Ответ:", "ОТВЕТ:", "Анализ:", "Аналитик:"]
            for prefix in prefixes_to_remove:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            # Проверяем минимальную длину
            if len(cleaned) < 30:
                return None
            
            # Обрезаем слишком длинные ответы
            if len(cleaned) > 1500:
                # Находим последнее предложение в пределах лимита
                sentences = cleaned[:1500].split('.')
                if len(sentences) > 1:
                    cleaned = '.'.join(sentences[:-1]) + '.'
                else:
                    cleaned = cleaned[:1500] + "..."
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Ошибка очистки ответа LLM: {e}")
            return None
    
    def _generate_analytical_fallback(self, query: str, data_analysis: Dict[str, Any]) -> str:
        """Аналитический fallback без LLM с поддержкой всех 10 вопросов"""
        
        if not data_analysis:
            return "Извините, не удалось получить данные для анализа вашего запроса."
        
        query_lower = query.lower()
        
        # Анализ по специфическим типам данных в результате
        
        # Вопрос 4: Средняя стоимость проекта по опыту
        if "avg_project_value_by_skill" in data_analysis:
            return self._analyze_avg_project_value(data_analysis)
        
        # Вопрос 6: Корреляция проектов и доходов
        elif "correlation_analysis" in data_analysis:
            return self._analyze_correlation(data_analysis)
        
        # Вопрос 7: Распределение способов оплаты
        elif "payment_method_distribution" in data_analysis:
            return self._analyze_payment_distribution(data_analysis)
        
        # Вопрос 8: Многофакторный анализ
        elif "multifactor_earnings" in data_analysis:
            return self._analyze_multifactor(data_analysis)
        
        # ИСПРАВЛЕНО: Вопрос 9: Почасовые ставки по способам оплаты
        elif "hourly_rate_by_payment" in data_analysis:
            return self._analyze_hourly_rates(data_analysis)
        
        # Вопрос 10: Активность по квалификации
        elif "activity_by_qualification" in data_analysis:
            return self._analyze_activity_by_qualification(data_analysis)
        
        # Остальные вопросы - используем существующую логику
        elif "эксперт" in query_lower and "процент" in query_lower:
            return self._analyze_expert_percentage(data_analysis)
        
        elif "связь" in query_lower or "влия" in query_lower:
            return self._analyze_relationships(data_analysis, query)
        
        elif "криптовалют" in query_lower:
            return self._analyze_crypto_comparison(data_analysis)
        
        elif any(word in query_lower for word in ["регион", "география", "распределение"]):
            return self._analyze_regional_distribution(data_analysis)
        
        else:
            return self._generate_general_analysis(data_analysis, query)
    
    def _analyze_avg_project_value(self, data_analysis: Dict[str, Any]) -> str:
        """Анализ средней стоимости проекта по уровню опыта (Вопрос 4)"""
        
        avg_value_data = data_analysis.get("avg_project_value_by_skill", {})
        summary = data_analysis.get("summary", {})
        
        if not avg_value_data or "mean" not in avg_value_data:
            return "Данные о средней стоимости проектов по уровню опыта недоступны."
        
        result = f"""💰 **Средняя стоимость проекта по уровню опыта:**

📊 **Базовая статистика** ({summary.get('total_records', 0)} фрилансеров):
• Общий средний доход: ${summary.get('mean_earnings', 0):.2f}

💼 **Стоимость проекта по уровням навыков:**
"""
        
        # Сортируем по средней стоимости проекта
        sorted_skills = sorted(avg_value_data["mean"].items(), key=lambda x: x[1], reverse=True)
        
        for skill, avg_value in sorted_skills:
            count = avg_value_data.get("count", {}).get(skill, 0)
            median_value = avg_value_data.get("median", {}).get(skill, avg_value)
            
            if not pd.isna(avg_value):
                result += f"• **{skill.capitalize()}**: ${avg_value:.2f} (медиана: ${median_value:.2f}) — {count} фрилансеров\n"
        
        # Анализ разброса
        if len(sorted_skills) >= 2:
            highest = sorted_skills[0]
            lowest = sorted_skills[-1]
            
            if not pd.isna(highest[1]) and not pd.isna(lowest[1]):
                difference = highest[1] - lowest[1]
                percentage_diff = (difference / lowest[1] * 100) if lowest[1] > 0 else 0
                
                result += f"\n💡 **Ключевые выводы:**\n"
                result += f"• Наивысшая стоимость проекта у **{highest[0]}**: ${highest[1]:.2f}\n"
                result += f"• Наименьшая стоимость проекта у **{lowest[0]}**: ${lowest[1]:.2f}\n"
                result += f"• Разница в стоимости: ${difference:.2f} ({percentage_diff:.1f}%)\n"
                result += f"• Опыт существенно влияет на стоимость проектов"
        
        return result
    
    def _analyze_correlation(self, data_analysis: Dict[str, Any]) -> str:
        """Анализ корреляции между проектами и доходом (Вопрос 6)"""
        
        correlation_data = data_analysis.get("correlation_analysis", {})
        summary = data_analysis.get("summary", {})
        
        if not correlation_data:
            return "Данные о корреляции между количеством проектов и доходом недоступны."
        
        correlation_coef = correlation_data.get("correlation_coefficient", 0)
        earnings_by_range = correlation_data.get("earnings_by_project_range", {})
        
        result = f"""📈 **Влияние количества проектов на доход фрилансера:**

🔍 **Корреляционный анализ**:
• Коэффициент корреляции: {correlation_coef:.3f}
• Сила связи: {self._interpret_correlation(correlation_coef)}

📊 **Доходы по диапазонам проектов:**
"""
        
        if earnings_by_range and "mean" in earnings_by_range:
            # Сортируем диапазоны логически
            range_order = ['1-10', '11-50', '51-100', '101-500', '500+']
            
            for range_name in range_order:
                if range_name in earnings_by_range["mean"]:
                    avg_earnings = earnings_by_range["mean"][range_name]
                    count = earnings_by_range.get("count", {}).get(range_name, 0)
                    
                    if not pd.isna(avg_earnings):
                        vs_overall = avg_earnings - summary.get('mean_earnings', 0)
                        result += f"• **{range_name} проектов**: ${avg_earnings:.2f} ({vs_overall:+.0f}$ к среднему) — {count} фрилансеров\n"
        
        result += f"\n💡 **Выводы:**\n"
        
        if correlation_coef > 0.3:
            result += "• Количество проектов положительно влияет на доходы\n"
            result += "• Чем больше проектов, тем выше средний доход\n"
            result += "• Опыт и репутация растут с количеством выполненных работ"
        elif correlation_coef > 0.1:
            result += "• Слабая положительная связь между количеством проектов и доходом\n"
            result += "• Качество проектов может быть важнее количества"
        else:
            result += "• Количество проектов слабо влияет на доходы\n"
            result += "• Другие факторы (навыки, специализация) более важны"
        
        return result
    
    def _analyze_payment_distribution(self, data_analysis: Dict[str, Any]) -> str:
        """Анализ распределения способов оплаты (Вопрос 7)"""
        
        distribution_data = data_analysis.get("payment_method_distribution", {})
        summary = data_analysis.get("summary", {})
        
        if not distribution_data:
            return "Данные о распределении способов оплаты недоступны."
        
        counts = distribution_data.get("counts", {})
        percentages = distribution_data.get("percentages", {})
        
        result = f"""💳 **Распределение способов оплаты среди фрилансеров:**

📊 **Общая статистика** ({summary.get('total_records', 0)} фрилансеров):

**Популярность способов оплаты:**
"""
        
        # Сортируем по популярности
        if percentages:
            sorted_methods = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
            
            for i, (method, percentage) in enumerate(sorted_methods, 1):
                count = counts.get(method, 0)
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔸"
                
                result += f"{emoji} **{method}**: {percentage:.1f}% ({count} фрилансеров)\n"
        
        # Анализ доминирования
        if percentages:
            most_popular = max(percentages.items(), key=lambda x: x[1])
            least_popular = min(percentages.items(), key=lambda x: x[1])
            
            result += f"\n💡 **Анализ популярности:**\n"
            result += f"• Самый популярный: **{most_popular[0]}** ({most_popular[1]:.1f}%)\n"
            result += f"• Наименее популярный: **{least_popular[0]}** ({least_popular[1]:.1f}%)\n"
            
            # Определяем концентрацию
            top_two = sorted(percentages.values(), reverse=True)[:2]
            top_two_share = sum(top_two)
            
            if top_two_share > 70:
                result += f"• Высокая концентрация: два лидирующих способа покрывают {top_two_share:.1f}% рынка\n"
            elif top_two_share > 50:
                result += f"• Умеренная концентрация: два лидера занимают {top_two_share:.1f}% рынка\n"
            else:
                result += f"• Равномерное распределение между всеми способами оплаты\n"
        
        return result
    
    def _analyze_multifactor(self, data_analysis: Dict[str, Any]) -> str:
        """Многофакторный анализ новичков vs экспертов по регионам (Вопрос 8)"""
        
        multifactor_data = data_analysis.get("multifactor_earnings", {})
        regional_differences = data_analysis.get("regional_skill_differences", {})
        summary = data_analysis.get("summary", {})
        
        if not multifactor_data and not regional_differences:
            return "Данные для многофакторного анализа недоступны."
        
        result = f"""🔍 **Сравнение доходов новичков и экспертов по регионам:**

📊 **Базовая статистика** ({summary.get('total_records', 0)} фрилансеров):
• Общий средний доход: ${summary.get('mean_earnings', 0):.2f}

🌍 **Региональные различия:**
"""
        
        if regional_differences:
            # Сортируем регионы по разнице между экспертами и новичками
            sorted_regions = sorted(
                regional_differences.items(), 
                key=lambda x: x[1].get('difference', 0), 
                reverse=True
            )
            
            for region, stats in sorted_regions[:8]:  # Топ 8 регионов
                expert_avg = stats.get('expert_avg', 0)
                beginner_avg = stats.get('beginner_avg', 0)
                difference = stats.get('difference', 0)
                
                if expert_avg > 0 and beginner_avg > 0:
                    percentage_diff = (difference / beginner_avg * 100) if beginner_avg > 0 else 0
                    
                    result += f"**{region}:**\n"
                    result += f"• Эксперты: ${expert_avg:.2f}\n"
                    result += f"• Новички: ${beginner_avg:.2f}\n"
                    result += f"• Разница: ${difference:.2f} ({percentage_diff:+.1f}%)\n\n"
        
        # Общие выводы
        if regional_differences:
            avg_difference = sum(stats.get('difference', 0) for stats in regional_differences.values()) / len(regional_differences)
            max_diff_region = max(regional_differences.items(), key=lambda x: x[1].get('difference', 0))
            min_diff_region = min(regional_differences.items(), key=lambda x: x[1].get('difference', 0))
            
            result += f"💡 **Ключевые выводы:**\n"
            result += f"• Средняя разница экспертов и новичков: ${avg_difference:.2f}\n"
            result += f"• Наибольший разрыв в {max_diff_region[0]}: ${max_diff_region[1].get('difference', 0):.2f}\n"
            result += f"• Наименьший разрыв в {min_diff_region[0]}: ${min_diff_region[1].get('difference', 0):.2f}\n"
            
            if avg_difference > 1000:
                result += "• Существенная разница в доходах между уровнями опыта во всех регионах"
            else:
                result += "• Умеренная разница в доходах между новичками и экспертами"
        
        return result
    
    def _analyze_hourly_rates(self, data_analysis: Dict[str, Any]) -> str:
        """ИСПРАВЛЕНО: Анализ почасовых ставок по способам оплаты (Вопрос 9)"""
        
        hourly_rate_data = data_analysis.get("hourly_rate_by_payment", {})
        summary = data_analysis.get("summary", {})
        
        if not hourly_rate_data:
            return "Данные о почасовых ставках по способам оплаты недоступны."
        
        result = f"""💰 **Средняя почасовая ставка у фрилансеров с разными способами оплаты:**

📊 **Общая статистика** ({summary.get('total_records', 0)} фрилансеров):
• Общий средний доход: ${summary.get('mean_earnings', 0):.2f}

⏰ **Почасовые ставки по способам оплаты:**
"""
        
        if "mean" in hourly_rate_data:
            # Сортируем по почасовой ставке
            sorted_methods = sorted(hourly_rate_data["mean"].items(), key=lambda x: x[1], reverse=True)
            
            for i, (method, rate) in enumerate(sorted_methods, 1):
                count = hourly_rate_data.get("count", {}).get(method, 0)
                median_rate = hourly_rate_data.get("median", {}).get(method, rate)
                
                emoji = "🏆" if i == 1 else "🥇" if i == 2 else "🥈" if i == 3 else "🔸"
                
                if not pd.isna(rate):
                    result += f"{emoji} **{method}**: ${rate:.2f}/час (медиана: ${median_rate:.2f}) — {count} фрилансеров\n"
        
        # Анализ различий в ставках
        if "mean" in hourly_rate_data and len(hourly_rate_data["mean"]) >= 2:
            rates = [v for v in hourly_rate_data["mean"].values() if not pd.isna(v)]
            if rates:
                highest_rate = max(rates)
                lowest_rate = min(rates)
                rate_spread = highest_rate - lowest_rate
                
                result += f"\n💡 **Ключевые выводы:**\n"
                result += f"• Наивысшая ставка: ${highest_rate:.2f}/час\n"
                result += f"• Наименьшая ставка: ${lowest_rate:.2f}/час\n"
                result += f"• Разброс ставок: ${rate_spread:.2f}/час\n"
                
                if rate_spread > 10:
                    result += "• Способ оплаты существенно влияет на почасовую ставку\n"
                    result += "• Некоторые методы оплаты позволяют устанавливать более высокие ставки"
                else:
                    result += "• Способ оплаты слабо влияет на почасовую ставку\n"
                    result += "• Ставки примерно одинаковы для всех методов оплаты"
                
                # Дополнительный анализ
                avg_rate = sum(rates) / len(rates)
                result += f"• Средняя почасовая ставка по рынку: ${avg_rate:.2f}/час\n"
                
                # Рекомендации
                best_method = max(hourly_rate_data["mean"].items(), key=lambda x: x[1])[0]
                result += f"• Наилучший способ оплаты по ставке: **{best_method}**"
        
        return result
    
    def _analyze_activity_by_qualification(self, data_analysis: Dict[str, Any]) -> str:
        """Анализ активности по уровню квалификации (Вопрос 10)"""
        
        activity_data = data_analysis.get("activity_by_qualification", {})
        summary = data_analysis.get("summary", {})
        
        if not activity_data:
            return "Данные об активности по уровню квалификации недоступны."
        
        result = f"""📋 **Количество проектов по уровню квалификации:**

📊 **Общая статистика** ({summary.get('total_records', 0)} фрилансеров):

**Активность по уровням квалификации:**
"""
        
        if "mean" in activity_data:
            # Сортируем по среднему количеству проектов
            sorted_skills = sorted(activity_data["mean"].items(), key=lambda x: x[1], reverse=True)
            
            for skill, avg_projects in sorted_skills:
                count = activity_data.get("count", {}).get(skill, 0)
                median_projects = activity_data.get("median", {}).get(skill, avg_projects)
                std_projects = activity_data.get("std", {}).get(skill, 0)
                
                if not pd.isna(avg_projects):
                    result += f"• **{skill.capitalize()}**: {avg_projects:.1f} проектов в среднем\n"
                    result += f"  └ Медиана: {median_projects:.1f}, Стандартное отклонение: {std_projects:.1f}, Количество: {count}\n\n"
        
        # Анализ закономерностей
        if "mean" in activity_data and len(activity_data["mean"]) >= 2:
            projects_by_skill = {k: v for k, v in activity_data["mean"].items() if not pd.isna(v)}
            
            if projects_by_skill:
                most_active = max(projects_by_skill.items(), key=lambda x: x[1])
                least_active = min(projects_by_skill.items(), key=lambda x: x[1])
                
                result += f"💡 **Ключевые наблюдения:**\n"
                result += f"• Наиболее активные: **{most_active[0]}** ({most_active[1]:.1f} проектов)\n"
                result += f"• Наименее активные: **{least_active[0]}** ({least_active[1]:.1f} проектов)\n"
                
                activity_ratio = most_active[1] / least_active[1] if least_active[1] > 0 else 0
                result += f"• Разница в активности: {activity_ratio:.1f}x\n"
                
                # Логическая интерпретация
                if 'expert' in projects_by_skill and 'beginner' in projects_by_skill:
                    expert_projects = projects_by_skill['expert']
                    beginner_projects = projects_by_skill['beginner']
                    
                    if expert_projects > beginner_projects:
                        result += "• Эксперты более активны и берутся за больше проектов\n"
                    else:
                        result += "• Новички более активны, возможно, набирают опыт\n"
        
        return result
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Интерпретация коэффициента корреляции"""
        if abs(correlation) >= 0.7:
            return "Сильная связь"
        elif abs(correlation) >= 0.3:
            return "Умеренная связь"
        elif abs(correlation) >= 0.1:
            return "Слабая связь"
        else:
            return "Очень слабая связь"
    
    def _analyze_expert_percentage(self, data_analysis: Dict[str, Any]) -> str:
        """Анализ процента экспертов"""
        
        if "total_experts" in data_analysis:
            total_experts = data_analysis.get("total_experts", 0)
            experts_under_100 = data_analysis.get("experts_under_100_projects", 0)
            percentage = data_analysis.get("percentage_under_100_projects", 0)
            
            result = f"""📊 **Анализ экспертов по количеству проектов:**

👥 **Общее количество экспертов**: {total_experts}
📈 **Эксперты с менее чем 100 проектами**: {experts_under_100}
📊 **Процент экспертов с <100 проектами**: {percentage:.1f}%
🎯 **Эксперты со 100+ проектами**: {total_experts - experts_under_100} ({100-percentage:.1f}%)

💡 **Вывод**: """
            
            if percentage > 70:
                result += "Большинство экспертов имеет относительно небольшое количество проектов, что указывает на важность качества работы над количеством."
            elif percentage > 50:
                result += "Примерно половина экспертов имеет менее 100 проектов, что показывает разнообразие путей к экспертизе."
            else:
                result += "Большинство экспертов имеет значительный опыт с 100+ проектами, что подчеркивает важность накопленного опыта."
            
            return result
        
        return "Данные об экспертах недоступны для анализа."
    
    def _analyze_relationships(self, data_analysis: Dict[str, Any], query: str) -> str:
        """Анализ связей между факторами"""
        
        skills_data = data_analysis.get("experience_earnings", {})
        payment_data = data_analysis.get("earnings_by_payment_method", {})
        regional_data = data_analysis.get("regional_earnings", {})
        summary = data_analysis.get("summary", {})
        
        result = f"""🔗 **Анализ взаимосвязей в данных о фрилансерах:**

📊 **Базовые показатели** ({summary.get('total_records', 0)} фрилансеров):
• Средний доход: ${summary.get('mean_earnings', 0):.2f}
• Медианный доход: ${summary.get('median_earnings', 0):.2f}

"""
        
        # Анализ по навыкам если есть данные
        if skills_data and "mean" in skills_data:
            result += "💡 **Влияние уровня навыков на доходы:**\n"
            sorted_skills = sorted(skills_data["mean"].items(), key=lambda x: x[1], reverse=True)
            
            highest_skill = sorted_skills[0] if sorted_skills else None
            lowest_skill = sorted_skills[-1] if sorted_skills else None
            
            for skill, income in sorted_skills:
                count = skills_data.get("count", {}).get(skill, 0)
                vs_average = ((income - summary.get('mean_earnings', 0)) / summary.get('mean_earnings', 1)) * 100
                result += f"• **{skill.capitalize()}**: ${income:.2f} ({vs_average:+.1f}% к среднему) — {count} чел.\n"
            
            if highest_skill and lowest_skill:
                skill_diff = highest_skill[1] - lowest_skill[1]
                result += f"\n📈 **Разница между высшим и низшим уровнем**: ${skill_diff:.2f}\n"
        
        # Анализ способов оплаты если есть данные
        if payment_data and "mean" in payment_data:
            result += "\n💳 **Влияние способа оплаты на доходы:**\n"
            sorted_payments = sorted(payment_data["mean"].items(), key=lambda x: x[1], reverse=True)
            
            for method, income in sorted_payments:
                count = payment_data.get("count", {}).get(method, 0)
                vs_average = ((income - summary.get('mean_earnings', 0)) / summary.get('mean_earnings', 1)) * 100
                result += f"• **{method}**: ${income:.2f} ({vs_average:+.1f}% к среднему) — {count} чел.\n"
        
        # Региональный анализ если есть данные
        if regional_data and "mean" in regional_data:
            result += "\n🌍 **Региональные различия:**\n"
            sorted_regions = sorted(regional_data["mean"].items(), key=lambda x: x[1], reverse=True)
            
            top_region = sorted_regions[0] if sorted_regions else None
            bottom_region = sorted_regions[-1] if sorted_regions else None
            
            if top_region and bottom_region:
                regional_diff = top_region[1] - bottom_region[1]
                result += f"• Наибольшие доходы: **{top_region[0]}** (${top_region[1]:.2f})\n"
                result += f"• Наименьшие доходы: **{bottom_region[0]}** (${bottom_region[1]:.2f})\n"
                result += f"• Региональная разница: ${regional_diff:.2f}\n"
        
        # Выводы
        result += "\n🎯 **Ключевые закономерности:**\n"
        
        if skills_data and payment_data:
            result += "• Уровень навыков и способ оплаты существенно влияют на доходы\n"
            result += "• Эксперты могут выбирать более выгодные условия работы\n"
            result += "• Комбинация высокого уровня навыков и правильного способа оплаты максимизирует доходы\n"
        
        return result
    
    def _analyze_crypto_comparison(self, data_analysis: Dict[str, Any]) -> str:
        """Анализ сравнения с криптовалютой"""
        
        earnings_data = data_analysis.get("earnings_by_payment_method", {})
        summary = data_analysis.get("summary", {})
        
        if not earnings_data or "mean" not in earnings_data:
            return "Данные о доходах по способам оплаты недоступны."
        
        methods = earnings_data["mean"]
        counts = earnings_data.get("count", {})
        
        # Находим криптовалюту
        crypto_method = None
        crypto_income = 0
        for method, income in methods.items():
            if "crypto" in method.lower():
                crypto_method = method
                crypto_income = income
                break
        
        if not crypto_method:
            return "Данные о доходах с криптовалютой не найдены в датасете."
        
        result = f"""💰 **Сравнительный анализ доходов с криптовалютой:**

🔸 **{crypto_method}**: ${crypto_income:.2f} (среднее) — {counts.get(crypto_method, 0)} фрилансеров

**Сравнение с другими способами оплаты:**
"""
        
        # Сравниваем с другими методами
        other_methods = [(method, income) for method, income in methods.items() if method != crypto_method]
        other_methods.sort(key=lambda x: x[1], reverse=True)
        
        for method, income in other_methods:
            diff = crypto_income - income
            percentage = (diff / income * 100) if income > 0 else 0
            count = counts.get(method, 0)
            
            if diff > 0:
                comp_text = f"на ${diff:.2f} ({percentage:.1f}%) выше"
                emoji = "📈"
            elif diff < 0:
                comp_text = f"на ${abs(diff):.2f} ({abs(percentage):.1f}%) ниже"
                emoji = "📉"
            else:
                comp_text = "примерно одинаково"
                emoji = "⚖️"
            
            result += f"{emoji} **{method}**: ${income:.2f} — {count} чел. ({comp_text})\n"
        
        # Общий вывод
        avg_other = sum(income for _, income in other_methods) / len(other_methods) if other_methods else 0
        diff_vs_avg = crypto_income - avg_other
        
        result += f"\n💡 **Вывод**: Фрилансеры с криптооплатой зарабатывают "
        if diff_vs_avg > 0:
            result += f"в среднем на ${diff_vs_avg:.2f} больше других способов оплаты."
        else:
            result += f"в среднем на ${abs(diff_vs_avg):.2f} меньше других способов оплаты."
        
        return result
    
    def _analyze_regional_distribution(self, data_analysis: Dict[str, Any]) -> str:
        """Анализ регионального распределения"""
        
        regional_data = data_analysis.get("regional_earnings", {})
        summary = data_analysis.get("summary", {})
        
        if not regional_data or "mean" not in regional_data:
            return "Данные о региональном распределении недоступны."
        
        regions = regional_data["mean"]
        counts = regional_data.get("count", {})
        
        # Сортируем по убыванию дохода
        sorted_regions = sorted(regions.items(), key=lambda x: x[1], reverse=True)
        
        result = f"""🌍 **Региональное распределение доходов фрилансеров:**

📊 **Средний доход по рынку**: ${summary.get('mean_earnings', 0):.2f}

**Рейтинг регионов по доходам:**
"""
        
        for i, (region, income) in enumerate(sorted_regions[:8], 1):
            count = counts.get(region, 0)
            vs_avg = income - summary.get('mean_earnings', 0)
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔸"
            
            result += f"{emoji} **{region}**: ${income:.2f} ({vs_avg:+.0f}$ к среднему) — {count} фрилансеров\n"
        
        # Анализ разброса
        if len(sorted_regions) >= 2:
            highest = sorted_regions[0][1]
            lowest = sorted_regions[-1][1]
            spread = highest - lowest
            
            result += f"\n📈 **Региональный анализ**:\n"
            result += f"• Максимальный доход: {sorted_regions[0][0]} (${highest:.2f})\n"
            result += f"• Минимальный доход: {sorted_regions[-1][0]} (${lowest:.2f})\n"
            result += f"• Разброс доходов: ${spread:.2f} ({(spread/lowest*100):.1f}%)\n"
        
        return result
    
    def _generate_general_analysis(self, data_analysis: Dict[str, Any], query: str) -> str:
        """Общий анализ для неспецифических запросов"""
        
        summary = data_analysis.get("summary", {})
        
        result = f"""📊 **Общий анализ данных о фрилансерах:**

🔢 **Основные показатели**:
• Всего фрилансеров в выборке: {summary.get('total_records', 0)}
• Средний доход: ${summary.get('mean_earnings', 0):.2f}
• Медианный доход: ${summary.get('median_earnings', 0):.2f}
• Диапазон доходов: ${summary.get('min_earnings', 0):.2f} - ${summary.get('max_earnings', 0):.2f}
"""
        
        # Добавляем доступные анализы
        available_analyses = []
        
        if "earnings_by_payment_method" in data_analysis:
            available_analyses.append("анализ по способам оплаты")
        
        if "regional_earnings" in data_analysis:
            available_analyses.append("региональный анализ")
        
        if "experience_earnings" in data_analysis:
            available_analyses.append("анализ по уровню навыков")
        
        if "total_experts" in data_analysis:
            available_analyses.append("статистика экспертов")
        
        if available_analyses:
            result += f"\n📋 **Доступные виды анализа**: {', '.join(available_analyses)}\n"
        
        result += f"""\n💡 **Рекомендации для получения детального анализа**:
• Задайте более конкретный вопрос о факторах, которые вас интересуют
• Спросите о сравнении групп фрилансеров
• Уточните интересующую вас метрику (доходы, проекты, регионы)

🎯 **Примеры конкретных вопросов**:
• "Какие факторы больше всего влияют на доходы?"
• "Сравните доходы экспертов и новичков"
• "Какая связь между регионом и способом оплаты?"
"""
        
        return result
    
    def get_device_info(self) -> Dict[str, Any]:
        """Получение информации об используемом устройстве"""
        info = {
            "device": self.device,
            "model_name": self.model_name,
            "cuda_available": torch.cuda.is_available(),
            "model_loaded": self.pipeline is not None,
            "ready": self.ready
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
            })
            
            if self.device == "cuda":
                info.update({
                    "gpu_memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),
                    "gpu_memory_reserved": torch.cuda.memory_reserved(0) / (1024**3)
                })
        
        return info