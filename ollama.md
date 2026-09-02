# Локальний запуск DeepEval через Ollama

Цей варіант дає змогу запускати навчальне демо без API-ключів Anthropic і OpenAI.

Ollama запускає мовну модель локально на комп'ютері, а DeepEval використовує її як модель-суддю.

## 1. Встановити Ollama

Для macOS через Homebrew:

```bash
brew install --cask ollama
```

Або завантажити застосунок з [ollama.com/download](https://ollama.com/download).

Перевірити встановлення:

```bash
ollama --version
```

Запусти застосунок Ollama. Якщо сервер не запустився автоматично, виконай:

```bash
ollama serve
```

## 2. Завантажити локальну модель

Для комп'ютера з 8 GB RAM:

```bash
ollama pull qwen2.5:3b
```

Для комп'ютера з 16 GB RAM або більше краще використовувати більшу модель:

```bash
ollama pull qwen2.5:7b
```

Перевірити модель:

```bash
ollama run qwen2.5:3b
```

Введи, наприклад:

```text
Привіт
```

Вийти з режиму можна через `Ctrl+D`.

## 3. Встановити інтеграцію LangChain

У корені проєкту виконай:

```bash
V=/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python
"$V" -m pip install langchain-ollama
```

Також додай до `requirements.txt`:

```text
langchain-ollama
```

## 4. Перевести `agent_plain.py` на Ollama

Зараз файл використовує Anthropic:

```python
from langchain_anthropic import ChatAnthropic
```

Заміни імпорт на:

```python
from langchain_ollama import ChatOllama
```

Заміни створення моделі:

```python
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
```

на:

```python
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)
```

Якщо завантажено модель 7b:

```python
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)
```

Запуск:

```bash
V=/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python
"$V" /Users/romanmakarenko/Documents/Python/deepeval-agent-demo/agent_plain.py
```

У тестовому прикладі використовується `ORD-1099`, але такого замовлення немає у словнику `ORDERS`. Тому відповідь на кшталт цієї є нормальною:

```text
No order found with ID ORD-1099.
```

Для перевірки знайденого замовлення використай `ORD-1042`.

## 5. Налаштувати DeepEval на локальну модель

Створи у корені проєкту файл `local_models.py`:

```python
from deepeval.models import OllamaModel

judge_model = OllamaModel(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0,
)
```

Для моделі 7b зміни назву:

```python
from deepeval.models import OllamaModel

judge_model = OllamaModel(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0,
)
```

У тесті було:

```python
metric = TaskCompletionMetric(
    threshold=0.7,
    model="gpt-4o",
)
```

Заміни на:

```python
from local_models import judge_model

metric = TaskCompletionMetric(
    threshold=0.7,
    model=judge_model,
)
```

Аналогічно заміни `model="gpt-4o"` на `model=judge_model` для таких метрик:

- `TaskCompletionMetric`;
- `AnswerRelevancyMetric`;
- `PromptAlignmentMetric`;
- `StepEfficiencyMetric`;
- `GEval`;
- `FaithfulnessMetric`;
- `ContextualPrecisionMetric`;
- `ContextualRecallMetric`;
- `BiasMetric`;
- `ToxicityMetric`;
- `PIILeakageMetric`;
- інших метрик, які використовують LLM-суддю.

`ToolCorrectnessMetric` зазвичай не потребує LLM-судді.

## 6. Перевести RAG на локальні embeddings

У `rag_agent.py` зараз використовуються OpenAI embeddings:

```python
from langchain_openai import OpenAIEmbeddings
```

і:

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

Це також потребує `OPENAI_API_KEY`.

Завантаж локальну embedding-модель:

```bash
ollama pull nomic-embed-text
```

Заміни імпорт:

```python
from langchain_ollama import OllamaEmbeddings
```

і створення embeddings:

```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)
```

Після цього RAG-агент також не використовує OpenAI.

## 7. Про `chatbot.py`

`chatbot.py` напряму використовує OpenAI:

```python
from openai import OpenAI
```

і модель:

```python
model="gpt-4o"
```

Тому цей файл залишатиметься залежним від `OPENAI_API_KEY`, доки його окремо не перевести на Ollama.

Для першого локального демо можна запускати:

- `agent_plain.py` після заміни моделі;
- `agent_instrumented.py` після заміни моделі;
- тести DeepEval після заміни LLM-судді;
- `rag_agent.py` після заміни моделі та embeddings.

## 8. Фінальна перевірка

Перевірити залежності:

```bash
V=/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python
"$V" -m pip check
```

Перевірити імпорти:

```bash
V=/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python
"$V" -c '
from deepeval.models import OllamaModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

judge = OllamaModel(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0,
)

print("Local Ollama integration: OK")
'
```

Переконайся, що Ollama запущена і модель доступна:

```bash
ollama list
```

## `.env`

Для локального варіанту ключі можуть залишатися порожніми:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
CONFIDENT_API_KEY=
```

## Обмеження локального варіанту

- перше завантаження моделі може зайняти кілька хвилин;
- локальна модель працює повільніше за хмарні API;
- маленькі моделі іноді неправильно формують JSON, потрібний DeepEval;
- для якіснішого tool calling і оцінювання краще використовувати `qwen2.5:7b`, якщо комп'ютер має достатньо RAM;
- локальний запуск не потребує оплати за токени, але використовує CPU/RAM/GPU комп'ютера.

## Офіційна документація

- [DeepEval — Ollama integration](https://deepeval.com/integrations/models/ollama)
- [Ollama — API documentation](https://docs.ollama.com/api)
- [Ollama — download](https://ollama.com/download)
