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

## Запуск локальних тестів

### Етап 1 — запустити Ollama

У першому терміналі запусти сервер і залиш його працювати:

```bash
ollama serve
```

У другому терміналі завантаж моделі:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### Етап 2 — запуск тестів із `evals/`

Усі команди нижче запускаються з кореня проєкту:

```bash
cd /Users/romanmakarenko/Documents/Python/deepeval-agent-demo
```

#### `test_TaskCompletion.py`

Перевіряє завершення задачі та правильність виклику інструментів для запитів про
статус замовлення і політику повернення.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_TaskCompletion
```

#### `test_TracingComponentsTest.py`

Перевіряє component-level tracing: `TaskCompletionMetric` оцінює весь trace,
а `ToolCorrectnessMetric` — правильність викликаного tool.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_TracingComponentsTest
```

#### `test_rag_agent.py`

Перевіряє RAG-відповіді за допомогою `ContextualPrecisionMetric`,
`ContextualRecallMetric`, `AnswerRelevancyMetric` і `FaithfulnessMetric`.
Потрібні `qwen2.5:3b` та `nomic-embed-text`.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_rag_agent
```

#### `test_multipleEvalsTest.py`

Запускає для трьох запитів `PromptAlignmentMetric`, `StepEfficiencyMetric` і
`AnswerRelevancyMetric`.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_multipleEvalsTest
```

#### `test_customMetricEvals.py`

Перевіряє відповідність фактичної відповіді очікуваній за допомогою `GEval` і
`SingleTurnParams`.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_customMetricEvals
```

#### `test_chatbot.py`

Запускає багатокрокову розмову та перевіряє релевантність ходів, утримання
знань і повноту розмови.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_chatbot
```

У локальному режимі `chatbot.py` використовує OpenAI-compatible endpoint Ollama,
тому окремий OpenAI API key не потрібен.

#### `test_chatbot_customreqd.py`

Перевіряє багатокрокову розмову за допомогою `ConversationalGEval`: чи вирішив
чатбот проблему клієнта, чи використовував tools і чи надав точну відповідь.

```bash
/Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_chatbot_customreqd
```

#### `test_agent_synthesized.py`

Генерує goldens із `policies.txt`, використовуючи локальні Ollama judge та
embeddings, а потім перевіряє їх за `BiasMetric`, `ToxicityMetric` і
`PIILeakageMetric`.

> **Важливо:** для `test_agent_synthesized.py` потрібна ще embedding-модель.
> Якщо вона ще не встановлена, запусти:
>
> ```bash
> ollama pull nomic-embed-text
> ```
>
> Після завершення завантаження повтори запуск тесту:
>
> ```bash
> /Users/romanmakarenko/Documents/Python/deepeval-agent-demo/.venv/bin/python -m evals.test_agent_synthesized
> ```

Перевірити встановлені моделі можна так:

```bash
ollama list
```

У списку мають бути `qwen2.5:3b` і `nomic-embed-text`. Якщо Ollama не
запущена, спочатку виконай `ollama serve` в іншому терміналі.

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

Повний цикл може тривати довго, оскільки кожен тест виконує локальні запити до
Ollama. Якщо потрібен лише базовий smoke test, почни з `test_TaskCompletion`.

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
| `evals/` | Вісім тестів DeepEval, описаних вище. |
| `ollama.md` | Розширена інструкція з локального запуску Ollama. |

`deepeval login` необов’язковий. Він потрібен лише для надсилання трейсів у
Confident AI dashboard.
