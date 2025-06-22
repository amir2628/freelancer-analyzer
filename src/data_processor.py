"""
Модуль для обработки и анализа данных фрилансеров
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)

class DataProcessor:
    """Класс для обработки данных о фрилансерах"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.data_path = Path("data/freelancer_earnings_bd.csv")
        self.ready = False
        
    async def initialize(self):
        """Инициализация и загрузка данных"""
        try:
            logger.info("Загрузка данных фрилансеров...")
            await self._load_data()
            await self._preprocess_data()
            self.ready = True
            logger.info(f"Данные успешно загружены. Размер датасета: {len(self.df)} записей")
        except Exception as e:
            logger.error(f"Ошибка инициализации данных: {e}")
            raise
    
    async def _load_data(self):
        """Загрузка данных из CSV файла"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")
        
        # Загружаем данные асинхронно
        loop = asyncio.get_event_loop()
        self.df = await loop.run_in_executor(None, pd.read_csv, self.data_path)
    
    async def _preprocess_data(self):
        """Предобработка данных"""
        if self.df is None:
            raise ValueError("Данные не загружены")
        
        logger.info("Предобработка данных...")
        
        # Показываем исходные колонки для отладки
        logger.info(f"Исходные колонки: {list(self.df.columns)}")
        
        # Стандартизация названий колонок
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        
        # Создаем mapping для колонок
        column_mapping = {
            'earnings_usd': 'earnings',
            'payment_method': 'payment_method', 
            'client_region': 'location',
            'job_completed': 'projects_completed',
            'experience_level': 'skill_level',
            'hourly_rate': 'hourly_rate',
            'job_success_rate': 'success_rate',
            'client_rating': 'client_rating'
        }
        
        # Переименовываем колонки для соответствия ожидаемым названиям
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns:
                self.df = self.df.rename(columns={old_name: new_name})
        
        logger.info(f"Колонки после переименования: {list(self.df.columns)}")
        
        # Определяем обязательные колонки для очистки
        required_columns = []
        if 'earnings' in self.df.columns:
            required_columns.append('earnings')
        if 'payment_method' in self.df.columns:
            required_columns.append('payment_method')
        if 'location' in self.df.columns:
            required_columns.append('location')
        
        # Очистка данных только по существующим колонкам
        if required_columns:
            self.df = self.df.dropna(subset=required_columns)
            logger.info(f"Удалены строки с пустыми значениями в колонках: {required_columns}")
        
        # Преобразование типов данных
        if 'earnings' in self.df.columns:
            self.df['earnings'] = pd.to_numeric(self.df['earnings'], errors='coerce')
            logger.info("Колонка earnings преобразована в числовой тип")
        
        if 'projects_completed' in self.df.columns:
            self.df['projects_completed'] = pd.to_numeric(self.df['projects_completed'], errors='coerce')
            logger.info("Колонка projects_completed преобразована в числовой тип")
        
        if 'hourly_rate' in self.df.columns:
            self.df['hourly_rate'] = pd.to_numeric(self.df['hourly_rate'], errors='coerce')
        
        # Создание дополнительных признаков
        if 'earnings' in self.df.columns and 'projects_completed' in self.df.columns:
            # Избегаем деления на ноль
            self.df['avg_project_value'] = self.df['earnings'] / self.df['projects_completed'].replace(0, np.nan)
            logger.info("Создана колонка avg_project_value")
        
        # Категоризация уровня опыта на основе существующих данных
        if 'skill_level' in self.df.columns:
            # Стандартизируем значения уровня навыков
            self.df['skill_level'] = self.df['skill_level'].str.lower().str.strip()
            logger.info("Стандартизирована колонка skill_level")
        
        # Если есть данные о количестве проектов, создаем категории
        if 'projects_completed' in self.df.columns:
            self.df['project_category'] = pd.cut(
                self.df['projects_completed'], 
                bins=[0, 10, 50, 100, float('inf')], 
                labels=['Новичок', 'Опытный', 'Профессионал', 'Эксперт'],
                include_lowest=True
            )
            logger.info("Создана категоризация по количеству проектов")
        
        # Финальная проверка данных
        logger.info(f"Размер датасета после обработки: {len(self.df)} записей")
        logger.info(f"Колонки в финальном датасете: {list(self.df.columns)}")
        
        logger.info("Предобработка данных завершена")
    
    def is_ready(self) -> bool:
        """Проверка готовности данных"""
        return self.ready and self.df is not None
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Получение информации о датасете"""
        if not self.is_ready():
            return {"error": "Данные не готовы"}
        
        return {
            "total_records": len(self.df),
            "columns": list(self.df.columns),
            "column_types": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_summary": self.df.describe().to_dict(),
            "categorical_columns": list(self.df.select_dtypes(include=['object']).columns)
        }
    
    def get_earnings_by_payment_method(self) -> Dict[str, float]:
        """Анализ доходов по способам оплаты"""
        if not self.is_ready():
            return {}
        
        return self.df.groupby('payment_method')['earnings'].mean().to_dict()
    
    def get_earnings_by_location(self) -> Dict[str, float]:
        """Анализ доходов по регионам"""
        if not self.is_ready():
            return {}
        
        return self.df.groupby('location')['earnings'].mean().to_dict()
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Статистика по экспертам"""
        if not self.is_ready():
            return {}
        
        # Ищем экспертов в разных возможных колонках и значениях
        expert_conditions = []
        
        if 'skill_level' in self.df.columns:
            # Проверяем различные варианты названий экспертов
            expert_values = ['expert', 'advanced', 'senior', 'профессионал', 'эксперт']
            for val in expert_values:
                expert_conditions.append(self.df['skill_level'].str.lower().str.contains(val, na=False))
        
        if 'experience_level' in self.df.columns:
            expert_values = ['expert', 'advanced', 'senior', 'эксперт']
            for val in expert_values:
                expert_conditions.append(self.df['experience_level'].str.lower().str.contains(val, na=False))
        
        # Также можем считать экспертами тех, у кого много проектов
        if 'projects_completed' in self.df.columns:
            expert_conditions.append(self.df['projects_completed'] >= 100)
        
        if not expert_conditions:
            return {"error": "Не удалось определить критерии для экспертов в данных"}
        
        # Объединяем все условия через OR
        expert_mask = expert_conditions[0]
        for condition in expert_conditions[1:]:
            expert_mask = expert_mask | condition
        
        experts = self.df[expert_mask]
        
        if len(experts) == 0:
            return {"error": "Эксперты не найдены в данных"}
        
        # Считаем экспертов с менее чем 100 проектами
        if 'projects_completed' in self.df.columns:
            experts_under_100_projects = experts[experts['projects_completed'] < 100]
            
            return {
                "total_experts": len(experts),
                "experts_under_100_projects": len(experts_under_100_projects),
                "percentage_under_100_projects": (len(experts_under_100_projects) / len(experts)) * 100 if len(experts) > 0 else 0
            }
        else:
            return {
                "total_experts": len(experts),
                "experts_under_100_projects": "Неизвестно (нет данных о проектах)",
                "percentage_under_100_projects": "Неизвестно (нет данных о проектах)"
            }
    
    def analyze_query_data(self, query_type: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Универсальный анализ данных на основе типа запроса"""
        if not self.is_ready():
            return {"error": "Данные не готовы"}
        
        logger.info(f"Анализ данных: тип={query_type}, фильтры={filters}")
        logger.info(f"Исходный размер датасета: {len(self.df)} записей")
        
        df_filtered = self.df.copy()
        
        # Применение фильтров с улучшенной логикой
        if filters:
            for column, value in filters.items():
                logger.info(f"Применение фильтра: {column} = {value}")
                
                if column == "payment_method":
                    # Улучшенная фильтрация по способам оплаты
                    if 'payment_method' in df_filtered.columns:
                        # Логируем уникальные значения в колонке
                        unique_values = df_filtered['payment_method'].dropna().unique()
                        logger.info(f"Уникальные способы оплаты в данных: {unique_values}")
                        
                        # Ищем совпадения по различным вариантам
                        if value == "cryptocurrency":
                            mask = df_filtered['payment_method'].str.lower().str.contains(
                                'crypto|bitcoin|btc|digital', case=False, na=False
                            )
                        elif value == "bank":
                            mask = df_filtered['payment_method'].str.lower().str.contains(
                                'bank|wire|transfer', case=False, na=False
                            )
                        elif value == "paypal":
                            mask = df_filtered['payment_method'].str.lower().str.contains(
                                'paypal|pp', case=False, na=False
                            )
                        else:
                            # Точное совпадение
                            mask = df_filtered['payment_method'].str.lower() == value.lower()
                        
                        df_filtered = df_filtered[mask]
                        logger.info(f"После фильтрации по {column}: {len(df_filtered)} записей")
                        
                elif column == "skill_level":
                    if 'skill_level' in df_filtered.columns:
                        unique_values = df_filtered['skill_level'].dropna().unique()
                        logger.info(f"Уникальные уровни навыков в данных: {unique_values}")
                        
                        if value == "expert":
                            mask = df_filtered['skill_level'].str.lower().str.contains(
                                'expert|advanced|senior|professional', case=False, na=False
                            )
                        elif value == "beginner":
                            mask = df_filtered['skill_level'].str.lower().str.contains(
                                'beginner|junior|entry|новичок', case=False, na=False
                            )
                        else:
                            mask = df_filtered['skill_level'].str.lower() == value.lower()
                        
                        df_filtered = df_filtered[mask]
                        logger.info(f"После фильтрации по {column}: {len(df_filtered)} записей")
                        
                elif column in df_filtered.columns:
                    if isinstance(value, list):
                        df_filtered = df_filtered[df_filtered[column].isin(value)]
                    else:
                        df_filtered = df_filtered[df_filtered[column] == value]
                    logger.info(f"После фильтрации по {column}: {len(df_filtered)} записей")
        
        # Если после фильтрации нет данных, вернем анализ без фильтров
        if len(df_filtered) == 0:
            logger.warning("После применения фильтров нет данных, выполняем анализ без фильтров")
            df_filtered = self.df.copy()
        
        result = {}
        
        if query_type == "earnings_comparison":
            # Сравнение доходов
            if 'payment_method' in df_filtered.columns and 'earnings' in df_filtered.columns:
                # Группировка по способам оплаты
                earnings_by_method = df_filtered.groupby('payment_method')['earnings'].agg(['mean', 'median', 'count'])
                
                # Логируем результаты группировки
                logger.info(f"Группировка по способам оплаты:")
                for method, stats in earnings_by_method.iterrows():
                    logger.info(f"  {method}: средний={stats['mean']:.2f}, медиана={stats['median']:.2f}, количество={stats['count']}")
                
                result['earnings_by_payment_method'] = earnings_by_method.to_dict()
                
                # Дополнительная статистика для сравнения
                if filters and 'payment_method' in filters:
                    # Сравниваем с общей статистикой
                    general_stats = self.df.groupby('payment_method')['earnings'].agg(['mean', 'median', 'count'])
                    result['general_earnings_by_payment_method'] = general_stats.to_dict()
            
        elif query_type == "regional_analysis":
            # Региональный анализ
            if 'location' in df_filtered.columns and 'earnings' in df_filtered.columns:
                regional_stats = df_filtered.groupby('location')['earnings'].agg(['mean', 'median', 'count'])
                result['regional_earnings'] = regional_stats.to_dict()
            
        elif query_type == "experience_analysis":
            # Анализ по опыту
            if 'skill_level' in df_filtered.columns and 'earnings' in df_filtered.columns:
                exp_stats = df_filtered.groupby('skill_level')['earnings'].agg(['mean', 'median', 'count'])
                result['experience_earnings'] = exp_stats.to_dict()
            elif 'project_category' in df_filtered.columns and 'earnings' in df_filtered.columns:
                # Используем категории проектов как показатель опыта
                exp_stats = df_filtered.groupby('project_category')['earnings'].agg(['mean', 'median', 'count'])
                result['experience_earnings'] = exp_stats.to_dict()
            
        elif query_type == "project_analysis":
            # Анализ проектов
            if 'projects_completed' in df_filtered.columns:
                result['project_stats'] = {
                    'mean_projects': df_filtered['projects_completed'].mean(),
                    'median_projects': df_filtered['projects_completed'].median(),
                    'projects_distribution': df_filtered['projects_completed'].describe().to_dict()
                }
        
        elif query_type == "avg_project_value_analysis":
            # Вопрос 4: Средняя стоимость проекта по уровню опыта
            if 'avg_project_value' in df_filtered.columns and 'skill_level' in df_filtered.columns:
                avg_value_by_skill = df_filtered.groupby('skill_level')['avg_project_value'].agg(['mean', 'median', 'count'])
                result['avg_project_value_by_skill'] = avg_value_by_skill.to_dict()
            elif 'earnings' in df_filtered.columns and 'projects_completed' in df_filtered.columns and 'skill_level' in df_filtered.columns:
                # Рассчитываем среднюю стоимость проекта на лету
                df_temp = df_filtered.copy()
                df_temp['calculated_avg_value'] = df_temp['earnings'] / df_temp['projects_completed'].replace(0, np.nan)
                avg_value_by_skill = df_temp.groupby('skill_level')['calculated_avg_value'].agg(['mean', 'median', 'count'])
                result['avg_project_value_by_skill'] = avg_value_by_skill.to_dict()
        
        elif query_type == "correlation_analysis":
            # Вопрос 6: Корреляция между проектами и доходом
            if 'projects_completed' in df_filtered.columns and 'earnings' in df_filtered.columns:
                correlation = df_filtered['projects_completed'].corr(df_filtered['earnings'])
                
                # Группируем по диапазонам проектов
                df_filtered['project_range'] = pd.cut(
                    df_filtered['projects_completed'], 
                    bins=[0, 10, 50, 100, 500, float('inf')], 
                    labels=['1-10', '11-50', '51-100', '101-500', '500+']
                )
                
                correlation_stats = df_filtered.groupby('project_range')['earnings'].agg(['mean', 'median', 'count'])
                result['correlation_analysis'] = {
                    'correlation_coefficient': correlation,
                    'earnings_by_project_range': correlation_stats.to_dict()
                }
        
        elif query_type == "payment_method_distribution":
            # Вопрос 7: Процентное распределение способов оплаты
            if 'payment_method' in df_filtered.columns:
                payment_counts = df_filtered['payment_method'].value_counts()
                payment_percentages = (payment_counts / len(df_filtered) * 100).to_dict()
                result['payment_method_distribution'] = {
                    'counts': payment_counts.to_dict(),
                    'percentages': payment_percentages
                }
        
        elif query_type == "multifactor_analysis":
            # Вопрос 8: Сравнение новичков и экспертов по регионам
            if 'skill_level' in df_filtered.columns and 'location' in df_filtered.columns and 'earnings' in df_filtered.columns:
                multifactor_stats = df_filtered.groupby(['location', 'skill_level'])['earnings'].agg(['mean', 'median', 'count'])
                result['multifactor_earnings'] = multifactor_stats.to_dict()
                
                # Дополнительно: разница между экспертами и новичками по регионам
                expert_earnings = df_filtered[df_filtered['skill_level'] == 'expert'].groupby('location')['earnings'].mean()
                beginner_earnings = df_filtered[df_filtered['skill_level'] == 'beginner'].groupby('location')['earnings'].mean()
                
                regional_differences = {}
                for region in expert_earnings.index:
                    if region in beginner_earnings.index:
                        regional_differences[region] = {
                            'expert_avg': expert_earnings[region],
                            'beginner_avg': beginner_earnings[region],
                            'difference': expert_earnings[region] - beginner_earnings[region]
                        }
                
                result['regional_skill_differences'] = regional_differences
        
        elif query_type == "hourly_rate_analysis":
            # Вопрос 9: Почасовая ставка по способам оплаты
            if 'hourly_rate' in df_filtered.columns and 'payment_method' in df_filtered.columns:
                hourly_rate_by_payment = df_filtered.groupby('payment_method')['hourly_rate'].agg(['mean', 'median', 'count'])
                result['hourly_rate_by_payment'] = hourly_rate_by_payment.to_dict()
                result['hourly_rate_metric_used'] = 'hourly_rate'
        
        elif query_type == "activity_by_qualification":
            # Вопрос 10: Количество проектов по уровню квалификации
            if 'projects_completed' in df_filtered.columns and 'skill_level' in df_filtered.columns:
                activity_stats = df_filtered.groupby('skill_level')['projects_completed'].agg(['mean', 'median', 'count', 'std'])
                result['activity_by_qualification'] = activity_stats.to_dict()
        
        # Общая статистика
        if 'earnings' in df_filtered.columns:
            result['summary'] = {
                'total_records': int(len(df_filtered)),
                'mean_earnings': float(df_filtered['earnings'].mean()),
                'median_earnings': float(df_filtered['earnings'].median()),
                'min_earnings': float(df_filtered['earnings'].min()),
                'max_earnings': float(df_filtered['earnings'].max())
            }
        else:
            result['summary'] = {
                'total_records': int(len(df_filtered)),
                'mean_earnings': None,
                'median_earnings': None,
                'min_earnings': None,
                'max_earnings': None
            }
        
        # Конвертируем все numpy типы в стандартные Python типы
        result = self._convert_numpy_types(result)
        
        # Логируем финальный результат
        logger.info(f"Результат анализа: {len(df_filtered)} записей обработано")
        if 'earnings' in df_filtered.columns:
            logger.info(f"Средний доход: {df_filtered['earnings'].mean():.2f}")
        
        return result
    
    def _convert_numpy_types(self, obj):
        """Конвертация numpy типов в стандартные Python типы"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_column_values(self, column: str) -> List[Any]:
        """Получение уникальных значений колонки"""
        if not self.is_ready() or column not in self.df.columns:
            return []
        
        return self.df[column].dropna().unique().tolist()
    
    def get_statistical_summary(self, columns: List[str] = None) -> Dict[str, Any]:
        """Получение статистического резюме по указанным колонкам"""
        if not self.is_ready():
            return {}
        
        if columns is None:
            columns = list(self.df.select_dtypes(include=[np.number]).columns)
        
        summary = {}
        for col in columns:
            if col in self.df.columns:
                summary[col] = {
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'count': self.df[col].count()
                }
        
        return summary