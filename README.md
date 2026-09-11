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

## Усі тести з `evals/`

Усі команди запускаються з кореня проєкту після запуску Ollama:

```bash
cd /Users/romanmakarenko/Documents/Python/deepeval-agent-demo
```

### `test_TaskCompletion.py`

Перевіряє завершення задачі та правильність виклику інструментів для двох
запитів: статус замовлення і політика повернення.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_TaskCompletion
```

### `test_TracingComponentsTest.py`

Перевіряє component-level tracing: `TaskCompletionMetric` оцінює весь trace,
а `ToolCorrectnessMetric` — правильність викликаного tool.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_TracingComponentsTest
```

### `test_rag_agent.py`

Перевіряє RAG-відповіді за метриками `ContextualPrecisionMetric`,
`ContextualRecallMetric`, `AnswerRelevancyMetric` і `FaithfulnessMetric`.
Потрібні моделі `qwen2.5:3b` і `nomic-embed-text`.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_rag_agent
```

### `test_multipleEvalsTest.py`

Запускає для трьох запитів одночасно `PromptAlignmentMetric`,
`StepEfficiencyMetric` і `AnswerRelevancyMetric`.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_multipleEvalsTest
```

### `test_customMetricEvals.py`

Перевіряє відповідність фактичної відповіді очікуваній за допомогою `GEval`
і `SingleTurnParams`.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_customMetricEvals
```

### `test_chatbot.py`

Запускає багатокрокову розмову з chatbot і перевіряє релевантність ходів,
утримання знань та повноту розмови.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_chatbot
```

> `chatbot.py` використовує OpenAI-compatible endpoint Ollama, якщо
> `DEEPEVAL_JUDGE_PROVIDER=ollama`, тому окремий OpenAI API key для локального
> режиму не потрібен.

### `test_chatbot_customreqd.py`

Перевіряє багатокрокову розмову за допомогою `ConversationalGEval`: чи вирішив
чатбот проблему клієнта, чи використовував tools і чи надав точну відповідь.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_chatbot_customreqd
```

### `test_agent_synthesized.py`

Генерує goldens із `policies.txt`, використовуючи локальні Ollama judge та
embeddings, а потім перевіряє їх за `BiasMetric`, `ToxicityMetric` і
`PIILeakageMetric`.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_agent_synthesized
```

Для цього тесту обов’язково встанови embedding-модель:

```bash
ollama pull nomic-embed-text
```

### Швидкий запуск усіх тестів

```bash
for test in \
  test_TaskCompletion \
  test_TracingComponentsTest \
  test_rag_agent \
  test_multipleEvalsTest \
  test_customMetricEvals \
  test_chatbot \
  test_chatbot_customreqd \
  test_agent_synthesized; do
  /Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m "evals.$test"
done
```

> Повний цикл може тривати довго, оскільки кожен тест виконує локальні запити
> до Ollama. Якщо потрібен лише базовий smoke test, почни з
> `test_TaskCompletion`.

### Інші команди

```bash
# Перевірити агента без eval
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python agent_instrumented.py

# Перевірити RAG-агента без eval
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python rag_agent.py
```

---

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
