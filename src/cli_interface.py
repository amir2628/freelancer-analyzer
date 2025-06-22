"""
Интерфейс командной строки для системы анализа данных фрилансеров
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import argparse
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.markdown import Markdown

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import DataProcessor
from src.llm_service import LLMService
from src.query_analyzer import QueryAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class FreelancerAnalysisCLI:
    """Командный интерфейс для анализа данных фрилансеров"""
    
    def __init__(self):
        self.console = Console()
        self.data_processor: Optional[DataProcessor] = None
        self.llm_service: Optional[LLMService] = None
        self.query_analyzer: Optional[QueryAnalyzer] = None
        self.initialized = False
    
    async def initialize(self):
        """Инициализация системы"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Инициализация обработчика данных
            task1 = progress.add_task("Загрузка данных фрилансеров...", total=None)
            self.data_processor = DataProcessor()
            await self.data_processor.initialize()
            progress.update(task1, completed=True)
            
            # Инициализация LLM сервиса
            task2 = progress.add_task("Загрузка языковой модели...", total=None)
            self.llm_service = LLMService()
            await self.llm_service.initialize()
            progress.update(task2, completed=True)
            
            # Инициализация анализатора запросов
            task3 = progress.add_task("Инициализация анализатора...", total=None)
            self.query_analyzer = QueryAnalyzer(self.data_processor, self.llm_service)
            progress.update(task3, completed=True)
            
        self.initialized = True
        self.console.print("✅ Система успешно инициализирована!", style="green bold")
    
    def show_welcome(self):
        """Отображение приветственного сообщения"""
        welcome_text = """
# 🔍 Система анализа данных фрилансеров

Добро пожаловать в интеллектуальную систему анализа данных о доходах и трендах фрилансеров!

## Возможности системы:
- Анализ доходов по способам оплаты
- Региональная статистика заработков
- Статистика по уровню экспертизы
- Анализ влияния опыта на доходы
- Сравнительный анализ различных метрик

## Примеры вопросов:
- "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте?"
- "Как распределяется доход фрилансеров по регионам?"
- "Какой процент экспертов выполнил менее 100 проектов?"
        """
        
        self.console.print(Panel(Markdown(welcome_text), title="Система анализа данных фрилансеров", border_style="blue"))
    
    def show_dataset_info(self):
        """Отображение информации о датасете"""
        if not self.initialized:
            self.console.print("❌ Система не инициализирована", style="red")
            return
        
        info = self.data_processor.get_dataset_info()
        
        table = Table(title="Информация о датасете")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="green")
        
        table.add_row("Общее количество записей", str(info.get("total_records", "Неизвестно")))
        table.add_row("Количество колонок", str(len(info.get("columns", []))))
        table.add_row("Колонки", ", ".join(info.get("columns", [])[:5]) + "...")
        
        if "numeric_summary" in info and "earnings" in info["numeric_summary"]:
            earnings_stats = info["numeric_summary"]["earnings"]
            table.add_row("Средний доход", f"${earnings_stats.get('mean', 0):.2f}")
            table.add_row("Медианный доход", f"${earnings_stats.get('50%', 0):.2f}")
            table.add_row("Максимальный доход", f"${earnings_stats.get('max', 0):.2f}")
        
        self.console.print(table)
    
    async def process_single_query(self, query: str):
        """Обработка одного запроса с расширенной диагностикой"""
        if not self.initialized:
            self.console.print("❌ Система не инициализирована", style="red")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Обработка запроса...", total=None)
            
            try:
                # Проверяем доступность LLM
                llm_available = self.llm_service.pipeline is not None
                progress.update(task, description=f"Анализ запроса (LLM: {'✅' if llm_available else '❌'})...")
                
                result = await self.query_analyzer.process_query(query)
                progress.update(task, completed=True)
                
                # Отображение результата
                self.display_result(query, result, llm_available)
                
            except Exception as e:
                progress.update(task, completed=True)
                self.console.print(f"❌ Ошибка обработки запроса: {e}", style="red")
                
                # Показываем дополнительную диагностику при ошибке
                self.console.print("\n🔍 Диагностическая информация:", style="yellow")
                self.console.print(f"• LLM доступен: {'Да' if self.llm_service.pipeline else 'Нет'}")
                self.console.print(f"• Модель: {self.llm_service.model_name}")
                self.console.print(f"• Устройство: {self.llm_service.device}")
    
    def display_result(self, query: str, result: dict, llm_available: bool = False):
        """Отображение результата анализа с дополнительной информацией"""
        # Заголовок с запросом
        self.console.print(Panel(f"Вопрос: {query}", title="Запрос", border_style="blue"))
        
        # Показываем статус LLM
        llm_status = "🤖 LLM активен" if llm_available else "⚠️ LLM недоступен (используются шаблоны)"
        
        # Ответ
        confidence = result.get('confidence', 0) * 100
        answer_panel = Panel(
            result.get("answer", "Ответ не получен"),
            title=f"Ответ (уверенность: {confidence:.1f}%) • {llm_status}",
            border_style="green" if llm_available else "yellow"
        )
        self.console.print(answer_panel)
        
        # Дополнительная информация
        if result.get("data_summary") and isinstance(result["data_summary"], dict):
            summary = result["data_summary"].get("summary", {})
            if summary:
                info_table = Table(title="Статистика анализа")
                info_table.add_column("Параметр", style="cyan")
                info_table.add_column("Значение", style="yellow")
                
                info_table.add_row("Обработано записей", str(summary.get("total_records", "Неизвестно")))
                if summary.get("mean_earnings"):
                    info_table.add_row("Средний доход", f"${summary['mean_earnings']:.2f}")
                if summary.get("median_earnings"):
                    info_table.add_row("Медианный доход", f"${summary['median_earnings']:.2f}")
                
                self.console.print(info_table)
        
        # Техническая информация для отладки
        tech_info = []
        if result.get("query_type"):
            tech_info.append(f"Тип запроса: {result['query_type']}")
        if result.get("question_id"):
            tech_info.append(f"ID вопроса: {result['question_id']}")
        if result.get("entities"):
            tech_info.append(f"Сущности: {', '.join(result['entities'])}")
        if result.get("metrics"):
            tech_info.append(f"Метрики: {', '.join(result['metrics'])}")
        
        if tech_info:
            self.console.print(f"[dim]🔧 {' • '.join(tech_info)}[/dim]")
        
        self.console.print()  # Пустая строка для разделения
    
    async def interactive_mode(self):
        """Интерактивный режим работы"""
        self.console.print("🔄 Запуск интерактивного режима", style="blue bold")
        self.console.print("Введите 'выход' или 'exit' для завершения работы")
        self.console.print("Введите 'помощь' или 'help' для получения справки")
        self.console.print()
        
        while True:
            try:
                query = Prompt.ask("\n[bold blue]Ваш вопрос[/bold blue]")
                
                if query.lower() in ['выход', 'exit', 'quit', 'q']:
                    self.console.print("👋 До свидания!", style="green")
                    break
                
                elif query.lower() in ['помощь', 'help', 'h']:
                    self.show_help()
                    continue
                
                elif query.lower() in ['info', 'инфо', 'информация']:
                    self.show_dataset_info()
                    continue
                
                elif query.lower() in ['примеры', 'examples']:
                    await self.show_examples()
                    continue
                
                elif query.lower() in ['debug', 'отладка', 'данные']:
                    await self.show_debug_info()
                    continue
                
                elif len(query.strip()) < 3:
                    self.console.print("❌ Слишком короткий запрос. Попробуйте сформулировать вопрос подробнее.", style="red")
                    continue
                
                await self.process_single_query(query)
                
            except KeyboardInterrupt:
                self.console.print("\n👋 Прерывание работы. До свидания!", style="yellow")
                break
            except Exception as e:
                self.console.print(f"❌ Неожиданная ошибка: {e}", style="red")
    
    def show_help(self):
        """Отображение справки"""
        help_text = """
## Система анализа данных фрилансеров

### Доступные команды:
- **выход/exit** - завершение работы
- **помощь/help** - показать эту справку
- **инфо/info** - информация о датасете
- **примеры/examples** - показать 10 доступных вопросов
- **отладка/debug** - отладочная информация о данных

### Как использовать систему:

**Рекомендуемый способ:**
1. Введите **примеры** для просмотра 10 готовых вопросов
2. Введите номер вопроса (1-10) для получения анализа
3. Система даст подробный ответ с использованием ИИ

**10 доступных вопросов:**
1. Сравнение доходов по способам оплаты (криптовалюта vs остальные)
2. Региональное распределение доходов
3. Статистика экспертов по количеству проектов
4. Средняя стоимость проекта по уровню опыта
5. Лучшие и худшие регионы по доходам
6. Влияние количества проектов на доход
7. Распределение способов оплаты среди фрилансеров
8. Сравнение доходов новичков и экспертов по регионам
9. Рейтинг успешности по способам оплаты
10. Активность фрилансеров по уровню квалификации

**Альтернативно:**
- Вы можете задать свой вопрос о данных фрилансеров
- ИИ постарается проанализировать данные и ответить

### Примеры использования:
```
> примеры          # Показать все вопросы
> 1                # Выбрать вопрос №1
> 5                # Выбрать вопрос №5
> Какой средний доход в Канаде?  # Свой вопрос
```
        """
        
        self.console.print(Panel(Markdown(help_text), title="Справка", border_style="yellow"))
    
    async def show_debug_info(self):
        """Показать отладочную информацию о данных"""
        if not self.initialized:
            self.console.print("❌ Система не инициализирована", style="red")
            return
        
        df = self.data_processor.df
        
        # Основная информация
        info_table = Table(title="Отладочная информация о данных")
        info_table.add_column("Параметр", style="cyan")
        info_table.add_column("Значение", style="green")
        
        info_table.add_row("Общее количество строк", str(len(df)))
        info_table.add_row("Количество колонок", str(len(df.columns)))
        info_table.add_row("Колонки", ", ".join(df.columns))
        
        self.console.print(info_table)
        
        # Уникальные значения способов оплаты
        if 'payment_method' in df.columns:
            payment_table = Table(title="Способы оплаты в данных")
            payment_table.add_column("Способ оплаты", style="yellow")
            payment_table.add_column("Количество", style="green")
            
            payment_counts = df['payment_method'].value_counts()
            for method, count in payment_counts.head(10).items():
                payment_table.add_row(str(method), str(count))
            
            self.console.print(payment_table)
        
        # Уникальные значения уровней навыков
        if 'skill_level' in df.columns:
            skill_table = Table(title="Уровни навыков в данных")
            skill_table.add_column("Уровень", style="yellow")
            skill_table.add_column("Количество", style="green")
            
            skill_counts = df['skill_level'].value_counts()
            for level, count in skill_counts.head(10).items():
                skill_table.add_row(str(level), str(count))
            
            self.console.print(skill_table)
        
        # Статистика по доходам
        if 'earnings' in df.columns:
            earnings_table = Table(title="Статистика доходов")
            earnings_table.add_column("Метрика", style="cyan")
            earnings_table.add_column("Значение", style="green")
            
            earnings_table.add_row("Минимум", f"${df['earnings'].min():.2f}")
            earnings_table.add_row("Максимум", f"${df['earnings'].max():.2f}")
            earnings_table.add_row("Среднее", f"${df['earnings'].mean():.2f}")
            earnings_table.add_row("Медиана", f"${df['earnings'].median():.2f}")
            earnings_table.add_row("Количество записей", str(df['earnings'].count()))
            
            self.console.print(earnings_table)

    async def show_examples(self):
        """Показать примеры вопросов с возможностью выбора"""
        if not self.initialized:
            self.console.print("❌ Система не инициализирована", style="red")
            return
        
        # Получаем структурированные примеры
        examples_data = await self.query_analyzer.get_suggested_queries()
        
        # Отображаем заголовок
        self.console.print(Panel("Выберите вопрос для анализа", title="Доступные вопросы", border_style="blue"))
        
        # Создаем таблицу с вопросами
        table = Table(title="10 вопросов для анализа данных фрилансеров")
        table.add_column("№", style="cyan", width=4)
        table.add_column("Вопрос", style="green", width=80)
        table.add_column("Категория", style="yellow", width=20)
        table.add_column("Сложность", style="magenta", width=10)
        
        for question in examples_data:
            table.add_row(
                str(question["id"]), 
                question["question"], 
                question["category"],
                question["difficulty"]
            )
        
        self.console.print(table)
        
        # Предлагаем выбрать вопрос
        self.console.print("\n💡 Введите номер вопроса (1-10) для анализа, или введите свой вопрос:")
    
    async def interactive_mode(self):
        """Интерактивный режим работы с поддержкой выбора вопросов"""
        self.console.print("🔄 Запуск интерактивного режима", style="blue bold")
        self.console.print("Введите 'выход' или 'exit' для завершения работы")
        self.console.print("Введите 'помощь' или 'help' для получения справки")
        self.console.print("Введите 'примеры' или 'examples' для просмотра доступных вопросов")
        self.console.print()
        
        # Показываем примеры при запуске
        await self.show_examples()
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]Ваш выбор[/bold blue]")
                
                if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                    self.console.print("👋 До свидания!", style="green")
                    break
                
                elif user_input.lower() in ['помощь', 'help', 'h']:
                    self.show_help()
                    continue
                
                elif user_input.lower() in ['info', 'инфо', 'информация']:
                    self.show_dataset_info()
                    continue
                
                elif user_input.lower() in ['примеры', 'examples']:
                    await self.show_examples()
                    continue
                
                elif user_input.lower() in ['debug', 'отладка', 'данные']:
                    await self.show_debug_info()
                    continue
                
                # Проверяем, выбрал ли пользователь номер вопроса
                elif user_input.isdigit():
                    question_num = int(user_input)
                    if 1 <= question_num <= 10:
                        question = await self._get_question_by_number(question_num)
                        if question:
                            self.console.print(f"\n[bold green]Выбран вопрос #{question_num}:[/bold green]")
                            self.console.print(f"[italic]{question}[/italic]\n")
                            await self.process_single_query(question)
                        else:
                            self.console.print("❌ Ошибка получения вопроса", style="red")
                    else:
                        self.console.print("❌ Введите номер от 1 до 10", style="red")
                    continue
                
                elif len(user_input.strip()) < 3:
                    self.console.print("❌ Слишком короткий запрос. Введите номер вопроса (1-10) или свой вопрос.", style="red")
                    continue
                
                # Обрабатываем произвольный вопрос
                await self.process_single_query(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n👋 Прерывание работы. До свидания!", style="yellow")
                break
            except Exception as e:
                self.console.print(f"❌ Неожиданная ошибка: {e}", style="red")

    async def _get_question_by_number(self, number: int) -> str:
        """Получить вопрос по номеру"""
        try:
            # Список вопросов напрямую здесь, чтобы убедиться в синхронизации
            questions = [
                "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте, по сравнению с другими способами оплаты?",
                "Как распределяется доход фрилансеров в зависимости от региона проживания?", 
                "Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?",
                "Какая средняя стоимость проекта у фрилансеров с разным уровнем опыта?",
                "В каких регионах фрилансеры зарабатывают больше всего и меньше всего?",
                "Как количество завершенных проектов влияет на средний доход фрилансера?",
                "Какой процент фрилансеров использует каждый способ оплаты?",
                "Есть ли разница в доходах между новичками и экспертами в разных регионах?",
                "Какая средняя почасовая ставка у фрилансеров с разными способами оплаты?",  # ИСПРАВЛЕННЫЙ вопрос 9
                "Сколько в среднем проектов выполняют фрилансеры разного уровня квалификации?"
            ]
            
            if 1 <= number <= len(questions):
                return questions[number - 1]
            return None
        except Exception as e:
            self.console.print(f"Ошибка получения вопроса: {e}", style="red")
            return None

async def main():
    """Главная функция CLI"""
    parser = argparse.ArgumentParser(description="Система анализа данных фрилансеров")
    parser.add_argument("--query", "-q", type=str, help="Одиночный запрос для обработки")
    parser.add_argument("--info", action="store_true", help="Показать информацию о датасете")
    parser.add_argument("--examples", action="store_true", help="Показать примеры запросов")
    
    args = parser.parse_args()
    
    cli = FreelancerAnalysisCLI()
    
    try:
        cli.show_welcome()
        await cli.initialize()
        
        if args.info:
            cli.show_dataset_info()
        elif args.examples:
            await cli.show_examples()
        elif args.query:
            await cli.process_single_query(args.query)
        else:
            await cli.interactive_mode()
            
    except KeyboardInterrupt:
        cli.console.print("\n👋 Работа прервана пользователем", style="yellow")
    except Exception as e:
        cli.console.print(f"❌ Критическая ошибка: {e}", style="red bold")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())