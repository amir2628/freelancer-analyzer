"""
Скрипт для диагностики GPU и проверки совместимости
"""

import torch
import sys
from pathlib import Path

def check_gpu_availability():
    """Проверка доступности GPU"""
    print("🔍 Проверка GPU...")
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA версия: {torch.version.cuda}")
        print(f"cuDNN версия: {torch.backends.cudnn.version()}")
        print(f"Количество GPU: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"\n📱 GPU {i}: {gpu_props.name}")
            print(f"   Память: {gpu_props.total_memory / (1024**3):.1f} GB")
            print(f"   Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"   Мультипроцессоры: {gpu_props.multi_processor_count}")
        
        # Тест простой операции на GPU
        try:
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print(f"\n✅ Тест GPU операций: Успешно")
            print(f"   Размер тестового тензора: {z.shape}")
            
            # Проверяем использование памяти
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**2)
            print(f"   Использовано памяти: {memory_allocated:.1f} MB")
            print(f"   Зарезервировано памяти: {memory_reserved:.1f} MB")
            
        except Exception as e:
            print(f"❌ Ошибка при тестировании GPU: {e}")
            
    else:
        print("❌ GPU недоступно")
        print("\n🔧 Возможные решения:")
        print("1. Установите CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("2. Установите PyTorch с поддержкой CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Проверьте драйверы NVIDIA")

def check_transformers_gpu():
    """Проверка работы transformers с GPU"""
    print("\n🤖 Проверка Transformers с GPU...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        model_name = "microsoft/DialoGPT-small"  # Используем маленькую модель для теста
        print(f"Загружаем тестовую модель: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if torch.cuda.is_available():
            print("Загружаем модель на GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0,
                max_length=50
            )
            
            # Тестовая генерация
            test_input = "Hello, how are you"
            result = pipe(test_input, max_length=30, num_return_sequences=1)
            print(f"✅ Тест генерации на GPU успешен")
            print(f"   Входной текст: {test_input}")
            print(f"   Результат: {result[0]['generated_text']}")
            
            # Проверяем память после загрузки модели
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"   Память после загрузки модели: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            
        else:
            print("⚠️ GPU недоступно, тестируем на CPU...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1,
                max_length=50
            )
            
            test_input = "Hello"
            result = pipe(test_input, max_length=20, num_return_sequences=1)
            print(f"✅ Тест генерации на CPU успешен")
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании Transformers: {e}")
        import traceback
        traceback.print_exc()

def check_memory_requirements():
    """Проверка требований к памяти"""
    print("\n💾 Проверка требований к памяти...")
    
    models_memory = {
        "microsoft/DialoGPT-small": 0.5,
        "microsoft/DialoGPT-medium": 1.5,
        "microsoft/DialoGPT-large": 3.0,
        "facebook/blenderbot-400M-distill": 1.2
    }
    
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Доступная GPU память: {available_memory:.1f} GB")
        
        print("\nРекомендации по моделям:")
        for model, required_memory in models_memory.items():
            if available_memory >= required_memory + 1:  # +1GB для буфера
                status = "✅ Рекомендуется"
            elif available_memory >= required_memory:
                status = "⚠️ Возможно (мало буфера)"
            else:
                status = "❌ Недостаточно памяти"
                
            print(f"   {model}: {required_memory}GB требуется - {status}")
    else:
        print("GPU недоступно - все модели будут работать на CPU")

def main():
    """Главная функция диагностики"""
    print("🚀 Диагностика GPU для системы анализа фрилансеров\n")
    
    check_gpu_availability()
    check_transformers_gpu()
    check_memory_requirements()
    
    print("\n📋 Рекомендации:")
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if available_memory >= 4:
            print("✅ Ваша система отлично подходит для работы с GPU")
            print("   Рекомендуем использовать DialoGPT-medium или DialoGPT-large")
        elif available_memory >= 2:
            print("✅ Ваша система подходит для работы с GPU")
            print("   Рекомендуем использовать DialoGPT-medium")
        else:
            print("⚠️ Ограниченная GPU память")
            print("   Рекомендуем использовать DialoGPT-small или работать на CPU")
    else:
        print("ℹ️ Система будет работать на CPU")
        print("   Это нормально, но может быть медленнее")
        print("   Для ускорения рассмотрите установку CUDA")

if __name__ == "__main__":
    main()