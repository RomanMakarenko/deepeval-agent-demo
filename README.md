# DeepEval Agentic Testing Demo

Навчальний проєкт для оцінювання LangChain-агентів за допомогою DeepEval.
За замовчуванням judge-модель і RAG runtime працюють локально через Ollama;
комерційні провайдери можна ввімкнути конфігурацією.

## Що демонструє проєкт

- інструментальний LangChain-агент із `TaskCompletionMetric` і
  `ToolCorrectnessMetric`;
- RAG-агент із пошуком по політиках та оцінюванням
  `FaithfulnessMetric`, `ContextualPrecisionMetric` і
  `ContextualRecallMetric`;
- DeepEval-трейсинг через `CallbackHandler`, `@observe` і
  `update_current_trace`;
- перемикання між локальним Ollama та хмарним RAG runtime.

## Встановлення

```bash
cd /Users/romanmakarenko/Documents/Python/deepeval-agent-demo
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

## Запуск локального тесту

### Етап 1 — запустити Ollama

У першому терміналі запусти сервер і залиш його працювати:

```bash
ollama serve
```

У другому терміналі завантаж моделі, необхідні для judge та RAG:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### Етап 2 — запустити Task Completion test

Команду потрібно виконувати з кореня проєкту:

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_TaskCompletion
```

Тест запускає два кейси та перевіряє їх метриками `Task Completion` і
`Tool Correctness`. Успішний запуск покаже окремий результат для кожного кейсу
та aggregate metrics.

![Результати DeepEval Task Completion](<Screenshot 2026-09-11 at 23.07.55.png>)

На прикладі вище обидва кейси пройшли обидві метрики: середній бал
`Task Completion` — `0.85`, а `Tool Correctness` — `1.00`.

## Інші команди

```bash
# Перевірити агента без eval
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python agent_instrumented.py

# Запустити RAG-тест
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_rag_agent

# Запустити тест компонентного трейсингу
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_TracingComponentsTest
```

## Конфігурація провайдерів

`.env.example` містить локальні значення за замовчуванням:

```env
DEEPEVAL_JUDGE_PROVIDER=ollama
DEEPEVAL_JUDGE_MODEL=qwen2.5:3b

RAG_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Для RAG можна окремо ввімкнути хмарний runtime:

```env
RAG_PROVIDER=cloud
RAG_CLOUD_CHAT_MODEL=claude-sonnet-4-6
RAG_CLOUD_EMBEDDING_MODEL=text-embedding-3-small
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

Для хмарного DeepEval judge замість Ollama:

```env
DEEPEVAL_JUDGE_PROVIDER=openai
DEEPEVAL_JUDGE_MODEL=gpt-4o
OPENAI_API_KEY=...
```

`RAG_PROVIDER` і `DEEPEVAL_JUDGE_PROVIDER` незалежні: наприклад, RAG може
працювати локально через Ollama, а judge — через OpenAI.

## Основні файли

| Файл | Призначення |
|---|---|
| `agent_plain.py` | Базовий агент без DeepEval-інструментації. |
| `agent_instrumented.py` | Агент із callback-трейсингом DeepEval. |
| `rag_agent.py` | RAG-агент із локальним Ollama runtime за замовчуванням. |
| `local_models.py` | Спільна конфігурація judge та runtime-моделей. |
| `evals/test_TaskCompletion.py` | Task Completion і Tool Correctness. |
| `evals/test_rag_agent.py` | Метрики якості RAG-відповідей. |
| `ollama.md` | Розширена інструкція з локального запуску Ollama. |

`deepeval login` необов’язковий. Він потрібен лише для надсилання трейсів у
Confident AI dashboard.
