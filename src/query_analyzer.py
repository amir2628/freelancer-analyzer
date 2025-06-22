"""
Модуль анализа и обработки запросов пользователей
"""

from typing import Dict, List, Optional, Any
import logging
import asyncio
from .data_processor import DataProcessor
from .llm_service import LLMService

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Анализатор запросов пользователей"""
    
    def __init__(self, data_processor: DataProcessor, llm_service: LLMService):
        self.data_processor = data_processor
        self.llm_service = llm_service
        
    async def process_query(self, query: str, language: str = "ru") -> Dict[str, Any]:
        """Основной метод обработки запроса"""
        try:
            logger.info(f"Обработка запроса: {query}")
            
            # 1. Анализ намерения пользователя
            intent = await self.llm_service.analyze_query_intent(query)
            logger.info(f"Определено намерение: {intent}")
            
            # 2. Выполнение анализа данных на основе намерения
            data_analysis = await self._execute_data_analysis(intent)
            logger.info(f"Результат анализа данных: ключи={list(data_analysis.keys())}")
            
            # 3. Генерация ответа
            answer = await self.llm_service.generate_answer(query, data_analysis)
            
            # 4. Оценка уверенности в ответе
            confidence = self._calculate_confidence(intent, data_analysis)
            
            result = {
                "answer": answer,
                "data_summary": data_analysis,
                "confidence": confidence,
                "query_type": intent.get("query_type"),
                "question_id": intent.get("question_id"),  # Добавляем для отладки
                "entities": intent.get("entities", []),
                "metrics": intent.get("metrics", [])
            }
            
            logger.info(f"Запрос успешно обработан с уверенностью {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "answer": f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}",
                "data_summary": {},
                "confidence": 0.1,
                "query_type": "error",
                "question_id": None,
                "entities": [],
                "metrics": []
            }
    
    async def _execute_data_analysis(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение анализа данных на основе намерения"""
        query_type = intent.get("query_type", "general")
        question_id = intent.get("question_id")
        entities = intent.get("entities", [])
        filters = intent.get("filters", {})
        
        try:
            # Определяем тип анализа на основе ID вопроса или общего типа
            if question_id == 1:
                # Вопрос 1: Сравнение криптовалюты с другими способами оплаты
                return await self._handle_comparison_query(entities, filters)
                
            elif question_id == 2:
                # Вопрос 2: Региональное распределение доходов
                return self.data_processor.analyze_query_data("regional_analysis", filters)
                
            elif question_id == 3:
                # Вопрос 3: Процент экспертов с <100 проектами
                return await self._handle_percentage_query(entities, filters)
                
            elif question_id == 4:
                # Вопрос 4: Средняя стоимость проекта по уровню опыта
                return self.data_processor.analyze_query_data("avg_project_value_analysis", filters)
                
            elif question_id == 5:
                # Вопрос 5: Лучшие и худшие регионы по доходам
                return self.data_processor.analyze_query_data("regional_analysis", filters)
                
            elif question_id == 6:
                # Вопрос 6: Влияние количества проектов на доход
                return self.data_processor.analyze_query_data("correlation_analysis", filters)
                
            elif question_id == 7:
                # Вопрос 7: Распределение способов оплаты
                return self.data_processor.analyze_query_data("payment_method_distribution", filters)
                
            elif question_id == 8:
                # Вопрос 8: Сравнение новичков и экспертов по регионам
                return self.data_processor.analyze_query_data("multifactor_analysis", filters)
                
            elif question_id == 9:
                # ИСПРАВЛЕНО: Вопрос 9: Почасовая ставка по способам оплаты
                logger.info("Выполняем анализ почасовых ставок (вопрос 9)")
                return self.data_processor.analyze_query_data("hourly_rate_analysis", filters)
                
            elif question_id == 10:
                # Вопрос 10: Активность по уровню квалификации
                return self.data_processor.analyze_query_data("activity_by_qualification", filters)
            
            # ДОБАВЛЕНО: Если нет конкретного ID, но тип - hourly_rate_analysis
            elif query_type == "hourly_rate_analysis":
                logger.info("Выполняем анализ почасовых ставок (по типу запроса)")
                return self.data_processor.analyze_query_data("hourly_rate_analysis", filters)
            
            # Если нет конкретного ID, используем общую логику по типу запроса
            elif query_type == "comparison":
                return await self._handle_comparison_query(entities, filters)
                
            elif query_type == "distribution":
                return await self._handle_distribution_query(entities, filters)
                
            elif query_type == "percentage":
                return await self._handle_percentage_query(entities, filters)
                
            elif query_type == "statistics":
                return await self._handle_statistics_query(entities, filters)
                
            elif query_type == "correlation":
                return self.data_processor.analyze_query_data("correlation_analysis", filters)
                
            elif query_type == "multifactor":
                return self.data_processor.analyze_query_data("multifactor_analysis", filters)
                
            else:
                # Общий анализ
                return await self._handle_general_query(entities, filters)
                
        except Exception as e:
            logger.error(f"Ошибка анализа данных: {e}")
            return {"error": str(e)}
    
    async def _handle_comparison_query(self, entities: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запросов сравнения"""
        logger.info("Выполнение анализа сравнения")
        
        if "cryptocurrency" in entities or "payment_method" in filters:
            # Специальный случай: сравнение криптовалюты с другими способами
            if filters.get("payment_method") == "cryptocurrency":
                # Получаем данные по всем способам оплаты для сравнения
                all_payment_analysis = self.data_processor.analyze_query_data("earnings_comparison", {})
                
                # Получаем данные только по криптовалюте
                crypto_analysis = self.data_processor.analyze_query_data("earnings_comparison", filters)
                
                # Объединяем результаты для полного сравнения
                if all_payment_analysis.get("earnings_by_payment_method") and crypto_analysis.get("summary"):
                    all_payment_analysis["crypto_specific_summary"] = crypto_analysis["summary"]
                    return all_payment_analysis
                
                return all_payment_analysis
            else:
                # Обычное сравнение по способам оплаты
                return self.data_processor.analyze_query_data("earnings_comparison", filters)
        
        elif "location" in entities:
            # Сравнение по регионам
            return self.data_processor.analyze_query_data("regional_analysis", filters)
        
        else:
            # Общее сравнение доходов
            return self.data_processor.analyze_query_data("earnings_comparison", filters)
    
    async def _handle_distribution_query(self, entities: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запросов распределения"""
        logger.info("Выполнение анализа распределения")
        
        if "location" in entities:
            # Распределение по регионам
            return self.data_processor.analyze_query_data("regional_analysis", filters)
        
        else:
            # Общее распределение доходов
            return self.data_processor.analyze_query_data("earnings_comparison", filters)
    
    async def _handle_percentage_query(self, entities: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запросов процентного анализа"""
        logger.info("Выполнение процентного анализа")
        
        if "expert" in entities:
            # Попробуем несколько способов анализа экспертов
            logger.info("Анализ экспертов - пробуем разные методы")
            
            # Способ 1: Стандартный анализ
            expert_stats = self.data_processor.get_expert_stats()
            if "error" not in expert_stats:
                logger.info("Стандартный анализ экспертов успешен")
                return expert_stats
            
            logger.warning(f"Стандартный анализ экспертов не удался: {expert_stats.get('error')}")
            
            # Способ 2: Альтернативный анализ
            alternative_stats = self._alternative_expert_analysis()
            if "error" not in alternative_stats:
                logger.info("Альтернативный анализ экспертов успешен")
                return alternative_stats
            
            logger.warning(f"Альтернативный анализ экспертов не удался: {alternative_stats.get('error')}")
            
            # Способ 3: Принудительный анализ по всем данным
            forced_stats = self._forced_expert_analysis()
            if forced_stats:
                logger.info("Принудительный анализ экспертов выполнен")
                return forced_stats
        
        # Дополнительный анализ проектов
        project_analysis = self.data_processor.analyze_query_data("project_analysis", filters)
        return project_analysis
    
    def _forced_expert_analysis(self) -> Dict[str, Any]:
        """Принудительный анализ экспертов используя все доступные данные"""
        try:
            df = self.data_processor.df
            logger.info(f"Принудительный анализ: датасет содержит {len(df)} записей")
            
            # Показать уникальные значения skill_level для отладки
            if 'skill_level' in df.columns:
                unique_skills = df['skill_level'].value_counts()
                logger.info(f"Уникальные значения skill_level: {unique_skills.to_dict()}")
                
                # Попробуем найти экспертов по точному совпадению
                experts_exact = df[df['skill_level'].str.lower() == 'expert']
                logger.info(f"Найдено экспертов (точное совпадение): {len(experts_exact)}")
                
                if len(experts_exact) > 0 and 'projects_completed' in df.columns:
                    experts_under_100 = experts_exact[experts_exact['projects_completed'] < 100]
                    
                    return {
                        "total_experts": len(experts_exact),
                        "experts_under_100_projects": len(experts_under_100),
                        "percentage_under_100_projects": (len(experts_under_100) / len(experts_exact)) * 100,
                        "analysis_method": "forced_exact_match",
                        "expert_skill_distribution": unique_skills.to_dict()
                    }
            
            # Если нет точного совпадения, используем топ-25% по доходам как экспертов
            if 'earnings' in df.columns:
                earnings_threshold = df['earnings'].quantile(0.75)
                top_earners = df[df['earnings'] >= earnings_threshold]
                logger.info(f"Используем топ-25% по доходам как экспертов: {len(top_earners)} человек")
                
                if 'projects_completed' in df.columns:
                    experts_under_100 = top_earners[top_earners['projects_completed'] < 100]
                    
                    return {
                        "total_experts": len(top_earners),
                        "experts_under_100_projects": len(experts_under_100),
                        "percentage_under_100_projects": (len(experts_under_100) / len(top_earners)) * 100,
                        "analysis_method": "top_25_percent_earnings",
                        "earnings_threshold": earnings_threshold,
                        "note": "Эксперты определены как топ-25% по доходам"
                    }
            
            # Последняя попытка - используем топ-25% по количеству проектов
            if 'projects_completed' in df.columns:
                projects_threshold = df['projects_completed'].quantile(0.75)
                experienced = df[df['projects_completed'] >= projects_threshold]
                logger.info(f"Используем топ-25% по проектам как экспертов: {len(experienced)} человек")
                
                experts_under_100 = experienced[experienced['projects_completed'] < 100]
                
                return {
                    "total_experts": len(experienced),
                    "experts_under_100_projects": len(experts_under_100),
                    "percentage_under_100_projects": (len(experts_under_100) / len(experienced)) * 100 if len(experienced) > 0 else 0,
                    "analysis_method": "top_25_percent_projects",
                    "projects_threshold": projects_threshold,
                    "note": "Эксперты определены как топ-25% по количеству проектов"
                }
            
            return {"error": "Невозможно провести анализ экспертов - недостаточно данных"}
            
        except Exception as e:
            logger.error(f"Ошибка принудительного анализа экспертов: {e}")
            return {"error": f"Критическая ошибка анализа: {str(e)}"}
    
    async def _handle_general_query(self, entities: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка общих запросов с расширенным анализом"""
        logger.info("Выполнение расширенного общего анализа")
        
        # Собираем данные по всем категориям для комплексного анализа
        result = {}
        
        # Анализ по способам оплаты
        payment_analysis = self.data_processor.analyze_query_data("earnings_comparison", {})
        if payment_analysis and "earnings_by_payment_method" in payment_analysis:
            result.update(payment_analysis)
        
        # Анализ по регионам
        regional_analysis = self.data_processor.analyze_query_data("regional_analysis", {})
        if regional_analysis and "regional_earnings" in regional_analysis:
            result.update(regional_analysis)
        
        # Анализ по навыкам
        skills_analysis = self.data_processor.analyze_query_data("experience_analysis", {})
        if skills_analysis and "experience_earnings" in skills_analysis:
            result.update(skills_analysis)
        
        # Анализ экспертов (если возможно)
        try:
            expert_stats = self._forced_expert_analysis()
            if expert_stats and "error" not in expert_stats:
                result.update(expert_stats)
        except:
            pass
        
        # Если у нас есть хотя бы базовая сводка
        if "summary" in result:
            return result
        
        # Последняя попытка - базовый анализ данных
        return self.data_processor.analyze_query_data("earnings_comparison", filters)
    
    def _alternative_expert_analysis(self) -> Dict[str, Any]:
        """Альтернативный анализ экспертов когда стандартный не работает"""
        try:
            df = self.data_processor.df
            
            # Попробуем найти экспертов по разным критериям
            expert_conditions = []
            
            # По уровню навыков
            if 'skill_level' in df.columns:
                unique_skills = df['skill_level'].dropna().unique()
                logger.info(f"Доступные уровни навыков: {unique_skills}")
                
                expert_mask = df['skill_level'].str.lower() == 'expert'
                expert_conditions.append(expert_mask)
            
            # По количеству проектов (высокое количество = эксперт)
            if 'projects_completed' in df.columns:
                # Считаем экспертами тех, у кого в топ 25% по количеству проектов
                threshold = df['projects_completed'].quantile(0.75)
                expert_mask = df['projects_completed'] >= threshold
                expert_conditions.append(expert_mask)
            
            # По доходу (высокий доход = эксперт)
            if 'earnings' in df.columns:
                # Считаем экспертами тех, у кого в топ 25% по доходу
                threshold = df['earnings'].quantile(0.75)
                expert_mask = df['earnings'] >= threshold
                expert_conditions.append(expert_mask)
            
            if not expert_conditions:
                return {"error": "Не удалось определить критерии для экспертов"}
            
            # Объединяем все условия через OR
            final_expert_mask = expert_conditions[0]
            for condition in expert_conditions[1:]:
                final_expert_mask = final_expert_mask | condition
            
            experts = df[final_expert_mask]
            
            if len(experts) == 0:
                return {"error": "Эксперты не найдены даже по альтернативным критериям"}
            
            # Анализируем количество проектов у экспертов
            if 'projects_completed' in experts.columns:
                experts_under_100 = experts[experts['projects_completed'] < 100]
                
                result = {
                    "total_experts": len(experts),
                    "experts_under_100_projects": len(experts_under_100),
                    "percentage_under_100_projects": (len(experts_under_100) / len(experts)) * 100 if len(experts) > 0 else 0,
                    "analysis_method": "alternative_criteria",
                    "expert_criteria_used": []
                }
                
                if 'skill_level' in df.columns:
                    result["expert_criteria_used"].append("skill_level")
                if 'projects_completed' in df.columns:
                    result["expert_criteria_used"].append("top_25%_projects")
                if 'earnings' in df.columns:
                    result["expert_criteria_used"].append("top_25%_earnings")
                
                return result
            else:
                return {"error": "Нет данных о количестве проектов для анализа"}
                
        except Exception as e:
            logger.error(f"Ошибка альтернативного анализа экспертов: {e}")
            return {"error": f"Ошибка анализа: {str(e)}"}
    
    async def _handle_statistics_query(self, entities: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка статистических запросов"""
        logger.info("Выполнение статистического анализа")
        
        # ДОБАВЛЕНО: Проверяем на почасовые ставки
        if "hourly_rate" in entities or any(word in str(entities).lower() for word in ["час", "ставка", "hourly"]):
            logger.info("Обнаружен запрос о почасовых ставках в статистическом анализе")
            return self.data_processor.analyze_query_data("hourly_rate_analysis", filters)
        
        if "projects" in entities:
            return self.data_processor.analyze_query_data("project_analysis", filters)
        
        elif "location" in entities:
            return self.data_processor.analyze_query_data("regional_analysis", filters)
        
        else:
            return self.data_processor.analyze_query_data("experience_analysis", filters)
    
    async def _handle_general_query(self, entities: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка общих запросов"""
        logger.info("Выполнение общего анализа")
        
        # Определяем наиболее подходящий тип анализа на основе сущностей
        if "location" in entities:
            return self.data_processor.analyze_query_data("regional_analysis", filters)
        
        elif "projects" in entities:
            return self.data_processor.analyze_query_data("project_analysis", filters)
        
        elif "expert" in entities:
            expert_stats = self.data_processor.get_expert_stats()
            if "error" not in expert_stats:
                return expert_stats
        
        # По умолчанию возвращаем анализ доходов
        return self.data_processor.analyze_query_data("earnings_comparison", filters)
    
    def _calculate_confidence(self, intent: Dict[str, Any], data_analysis: Dict[str, Any]) -> float:
        """Расчет уверенности в ответе"""
        base_confidence = intent.get("confidence", 0.5)
        
        # Проверяем, есть ли question_id (предопределенный вопрос)
        question_id = intent.get("question_id")
        if question_id:
            base_confidence = 0.9  # Высокая уверенность для предопределенных вопросов
        
        # Увеличиваем уверенность, если есть релевантные данные для анализа
        relevant_data_found = False
        
        if question_id == 1 and "earnings_by_payment_method" in data_analysis:
            relevant_data_found = True
        elif question_id == 2 and "regional_earnings" in data_analysis:
            relevant_data_found = True
        elif question_id == 3 and "total_experts" in data_analysis:
            relevant_data_found = True
        elif question_id == 4 and "avg_project_value_by_skill" in data_analysis:
            relevant_data_found = True
        elif question_id == 5 and "regional_earnings" in data_analysis:
            relevant_data_found = True
        elif question_id == 6 and "correlation_analysis" in data_analysis:
            relevant_data_found = True
        elif question_id == 7 and "payment_method_distribution" in data_analysis:
            relevant_data_found = True
        elif question_id == 8 and "multifactor_earnings" in data_analysis:
            relevant_data_found = True
        elif question_id == 9 and "hourly_rate_by_payment" in data_analysis:
            relevant_data_found = True
        elif question_id == 10 and "activity_by_qualification" in data_analysis:
            relevant_data_found = True
        elif "earnings_by_payment_method" in data_analysis or "regional_earnings" in data_analysis:
            relevant_data_found = True
        
        if relevant_data_found:
            base_confidence += 0.1
        else:
            # Снижаем уверенность если нет релевантных данных
            base_confidence *= 0.6
        
        # Увеличиваем уверенность, если найдены релевантные сущности
        if intent.get("entities"):
            base_confidence += 0.05
        
        # Увеличиваем уверенность, если определены метрики
        if intent.get("metrics"):
            base_confidence += 0.05
        
        # Проверяем качество данных в summary
        if data_analysis.get("summary", {}).get("total_records", 0) > 0:
            base_confidence += 0.05
        
        # Если получили ошибку в данных, сильно снижаем уверенность
        if "error" in data_analysis:
            base_confidence *= 0.3
        
        # Ограничиваем диапазон от 0.1 до 0.95
        return max(0.1, min(base_confidence, 0.95))
    
    async def get_suggested_queries(self) -> List[Dict[str, Any]]:
        """Получение предложенных запросов"""
        return [
            # Оригинальные 3 вопроса из задания
            {
                "id": 1,
                "question": "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте, по сравнению с другими способами оплаты?",
                "category": "Сравнительный анализ",
                "difficulty": "средний"
            },
            {
                "id": 2,
                "question": "Как распределяется доход фрилансеров в зависимости от региона проживания?",
                "category": "Региональный анализ", 
                "difficulty": "легкий"
            },
            {
                "id": 3,
                "question": "Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?",
                "category": "Статистический анализ",
                "difficulty": "средний"
            },
            
            # 7 дополнительных вопросов
            {
                "id": 4,
                "question": "Какая средняя стоимость проекта у фрилансеров с разным уровнем опыта?",
                "category": "Анализ по опыту",
                "difficulty": "легкий"
            },
            {
                "id": 5,
                "question": "В каких регионах фрилансеры зарабатывают больше всего и меньше всего?",
                "category": "Региональная статистика",
                "difficulty": "легкий"
            },
            {
                "id": 6,
                "question": "Как количество завершенных проектов влияет на средний доход фрилансера?",
                "category": "Корреляционный анализ",
                "difficulty": "сложный"
            },
            {
                "id": 7,
                "question": "Какой процент фрилансеров использует каждый способ оплаты?",
                "category": "Процентное распределение",
                "difficulty": "легкий"
            },
            {
                "id": 8,
                "question": "Есть ли разница в доходах между новичками и экспертами в разных регионах?",
                "category": "Многофакторный анализ",
                "difficulty": "сложный"
            },
            {
                "id": 9,
                "question": "Какая средняя почасовая ставка у фрилансеров с разными способами оплаты?",
                "category": "Анализ почасовых ставок",
                "difficulty": "средний"
            },
            {
                "id": 10,
                "question": "Сколько в среднем проектов выполняют фрилансеры разного уровня квалификации?",
                "category": "Анализ активности",
                "difficulty": "легкий"
            }
        ]
    
    async def validate_query(self, query: str) -> Dict[str, Any]:
        """Валидация запроса пользователя"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "suggestions": []
        }
        
        # Проверка длины запроса
        if len(query.strip()) < 5:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Запрос слишком короткий")
            validation_result["suggestions"].append("Пожалуйста, сформулируйте более детальный вопрос")
        
        # Проверка на наличие вопросительных слов или знаков
        question_indicators = ["что", "как", "какой", "какая", "какие", "сколько", "где", "когда", "почему", "?"]
        if not any(indicator in query.lower() for indicator in question_indicators):
            validation_result["suggestions"].append("Рекомендуется формулировать запрос в виде вопроса")
        
        # Проверка на релевантность к данным о фрилансерах
        relevant_keywords = ["фрилансер", "доход", "заработок", "проект", "оплата", "регион", "эксперт", "опыт"]
        if not any(keyword in query.lower() for keyword in relevant_keywords):
            validation_result["suggestions"].append("Попробуйте сформулировать вопрос, связанный с данными о фрилансерах")
        
        return validation_result