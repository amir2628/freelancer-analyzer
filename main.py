"""
Главное приложение FastAPI для анализа данных фрилансеров
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging

from src.data_processor import DataProcessor
from src.query_analyzer import QueryAnalyzer
from src.llm_service import LLMService

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Система анализа данных фрилансеров",
    description="API для анализа статистических данных о доходах фрилансеров с поддержкой запросов на естественном языке",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    """Модель запроса от пользователя"""
    question: str
    language: Optional[str] = "ru"

class QueryResponse(BaseModel):
    """Модель ответа системы"""
    question: str
    answer: str
    data_summary: Optional[dict] = None
    confidence: Optional[float] = None

# Инициализация компонентов системы
data_processor = None
query_analyzer = None
llm_service = None

@app.on_event("startup")
async def startup_event():
    """Инициализация системы при запуске"""
    global data_processor, query_analyzer, llm_service
    
    try:
        logger.info("Запуск системы анализа данных фрилансеров...")
        
        # Инициализация обработчика данных
        data_processor = DataProcessor()
        await data_processor.initialize()
        
        # Инициализация LLM сервиса
        llm_service = LLMService()
        await llm_service.initialize()
        
        # Инициализация анализатора запросов
        query_analyzer = QueryAnalyzer(data_processor, llm_service)
        
        logger.info("Система успешно инициализирована")
        
    except Exception as e:
        logger.error(f"Ошибка при инициализации системы: {e}")
        raise

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Добро пожаловать в систему анализа данных фрилансеров",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Проверка состояния системы"""
    try:
        # Проверяем доступность всех компонентов
        data_status = data_processor is not None and data_processor.is_ready()
        llm_status = llm_service is not None and llm_service.is_ready()
        
        return {
            "status": "healthy" if data_status and llm_status else "degraded",
            "components": {
                "data_processor": "ready" if data_status else "not_ready",
                "llm_service": "ready" if llm_status else "not_ready"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка проверки состояния: {str(e)}")

@app.get("/dataset/info")
async def get_dataset_info():
    """Получение информации о датасете"""
    try:
        if not data_processor:
            raise HTTPException(status_code=503, detail="Система не инициализирована")
        
        info = data_processor.get_dataset_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения информации о датасете: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Обработка запроса на естественном языке"""
    try:
        if not query_analyzer:
            raise HTTPException(status_code=503, detail="Система не инициализирована")
        
        logger.info(f"Получен запрос: {request.question}")
        
        # Обработка запроса
        result = await query_analyzer.process_query(request.question, request.language)
        
        response = QueryResponse(
            question=request.question,
            answer=result["answer"],
            data_summary=result.get("data_summary"),
            confidence=result.get("confidence")
        )
        
        logger.info(f"Запрос обработан успешно")
        return response
        
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

@app.get("/dataset/debug")
async def debug_dataset():
    """Отладочная информация о датасете"""
    try:
        if not data_processor:
            raise HTTPException(status_code=503, detail="Система не инициализирована")
        
        if not data_processor.is_ready():
            raise HTTPException(status_code=503, detail="Данные не готовы")
        
        df = data_processor.df
        
        debug_info = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "sample_data": df.head(3).to_dict('records'),
            "unique_payment_methods": df['payment_method'].dropna().unique().tolist() if 'payment_method' in df.columns else [],
            "unique_skill_levels": df['skill_level'].dropna().unique().tolist() if 'skill_level' in df.columns else [],
            "unique_locations": df['location'].dropna().unique().tolist()[:10] if 'location' in df.columns else [],  # Первые 10
            "earnings_stats": {
                "min": float(df['earnings'].min()) if 'earnings' in df.columns else None,
                "max": float(df['earnings'].max()) if 'earnings' in df.columns else None,
                "mean": float(df['earnings'].mean()) if 'earnings' in df.columns else None,
                "count": int(df['earnings'].count()) if 'earnings' in df.columns else None
            }
        }
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения отладочной информации: {str(e)}")

@app.post("/test/llm")
async def test_llm(request: dict):
    """Тестирование LLM с произвольным текстом"""
    try:
        if not llm_service:
            raise HTTPException(status_code=503, detail="LLM сервис не инициализирован")
        
        test_prompt = request.get("prompt", "Привет! Как дела?")
        
        if llm_service.pipeline:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: llm_service.pipeline(test_prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
                )
                
                return {
                    "llm_available": True,
                    "model_name": llm_service.model_name,
                    "device": llm_service.device,
                    "prompt": test_prompt,
                    "response": result[0]['generated_text'] if result else "Нет ответа",
                    "status": "success"
                }
            except Exception as e:
                return {
                    "llm_available": False,
                    "error": str(e),
                    "status": "error"
                }
        else:
            return {
                "llm_available": False,
                "message": "Pipeline не инициализирован",
                "status": "not_available"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка тестирования LLM: {str(e)}")

@app.get("/device/info")
async def get_device_info():
    """Получение информации об используемом устройстве"""
    try:
        if not llm_service:
            raise HTTPException(status_code=503, detail="LLM сервис не инициализирован")
        
        device_info = llm_service.get_device_info()
        return device_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения информации об устройстве: {str(e)}")

@app.get("/examples")
async def get_example_queries():
    """Получение примеров запросов"""
    examples = [
        "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте, по сравнению с другими способами оплаты?",
        "Как распределяется доход фрилансеров в зависимости от региона проживания?",
        "Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?",
        "Какая средняя стоимость проекта у фрилансеров с опытом работы более 5 лет?",
        "В каких областях деятельности фрилансеры зарабатывают больше всего?",
        "Как количество завершенных проектов влияет на средний доход фрилансера?",
        "Какой процент фрилансеров работает полный рабочий день?"
    ]
    
    return {
        "examples": examples,
        "description": "Примеры вопросов, которые может обработать система"
    }



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )