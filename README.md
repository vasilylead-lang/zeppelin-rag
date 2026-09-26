# zeppelin-rag

Вопросно-ответная система по базе знаний о цеппелинах и дирижаблях. Ответы генерирует
Claude, поиск и самокоррекция собраны в граф на **LangGraph**, качество оценивается
через **RAGAS**.

*Corrective RAG over a Russian-language zeppelin knowledge base (LangGraph + Claude), evaluated with RAGAS.*

## Как это работает

```mermaid
flowchart LR
    Q([Вопрос]) --> R[retrieve<br/>гибридный поиск]
    R --> G{grade<br/>Claude Haiku}
    G -- есть релевантные --> A[generate<br/>Claude Opus]
    G -- ничего не нашлось --> W[rewrite<br/>переформулировать запрос]
    W --> R
    A --> E([Ответ со ссылками])
```

1. **retrieve.** Гибридный поиск по 70 разделам базы знаний. Используются локальные
   мультиязычные эмбеддинги (fastembed, ONNX, без API) и BM25 по основам слов.
   Лексическая часть ловит редкие термины вроде «блау-газ» и «оперение», которые
   маленькая модель эмбеддингов пропускает.
2. **grade.** Быстрая модель отбирает фрагменты, которые действительно отвечают на
   вопрос (structured output).
3. **rewrite.** Если релевантных фрагментов нет, запрос переформулируется и поиск
   повторяется, не более `max_rewrites` раз.
4. **generate.** Основная модель отвечает только по найденным фрагментам и ставит
   ссылки `[n]`. Если данных нет, она так и говорит.

## Почему RAGAS, а не Langfuse

Задача проекта — измерить качество RAG-пайплайна. RAGAS считает метрики офлайн, одной
командой, без сервера и аккаунта, поэтому оценку легко воспроизвести. Langfuse силён в
трассировке и мониторинге продакшена. Ему нужен запущенный сервер или облачный проект,
а для небольшого демо это лишняя инфраструктура.

| Метрика RAGAS | Что проверяет |
|---|---|
| `faithfulness` | утверждения ответа подтверждаются найденным контекстом (нет галлюцинаций) |
| `answer_relevancy` | ответ отвечает именно на заданный вопрос |
| `context_recall` | найденный контекст покрывает эталонный ответ |
| `context_precision_with_reference` | релевантные фрагменты стоят в выдаче первыми |
| `factual_correctness` | факты ответа совпадают с эталоном (F1 по утверждениям) |

В роли судьи выступает Claude Sonnet. Тестовый набор — 17 вопросов с эталонными
ответами: [data/eval/testset.jsonl](data/eval/testset.jsonl).

## База знаний

9 документов на русском в [data/knowledge/](data/knowledge/):

| Файл | Содержание |
|---|---|
| `01_history.md` | граф Цеппелин, LZ 1, DELAG, Первая мировая, «Граф Цеппелин», конец эпохи, Zeppelin NT |
| `02_types_construction.md` | жёсткие, полужёсткие и мягкие дирижабли; каркас, дюралюминий, газовые ячейки, бодрюш, обшивка, оперение |
| `03_form_factor_aerodynamics.md` | форма корпуса, удлинение, составляющие сопротивления, формула D = ½ρv²C_DV·V^(2/3), расчёт для «Гинденбурга», число Рейнольдса, P ∝ v³, закон квадрата-куба, присоединённая масса |
| `04_buoyancy_gases.md` | закон Архимеда, водород и гелий, влияние высоты, высота давления, перегрев газа, балласт, блау-газ, динамический подъём, баллонеты |
| `05_propulsion_performance.md` | двигатели Maybach и Daimler-Benz, винты, скорости, дальность, влияние ветра, энергоэффективность |
| `06_stability_control_operations.md` | устойчивость, момент Мунка, органы управления, прочность, мачты, эллинги, погода |
| `07_famous_airships.md` | характеристики LZ 127, LZ 129, LZ 130, Akron/Macon, Shenandoah, R100/R101, «Норвегия», «Италия», «СССР В-6» |
| `08_disasters_safety.md` | катастрофы «Гинденбурга», R101, Akron, Macon, Shenandoah и их уроки |
| `09_modern_airships.md` | Zeppelin NT, Airlander 10, Pathfinder 1, Flying Whales, современные ниши |

Каждый раздел `## ...` — отдельный фрагмент для поиска. Чтобы расширить базу, достаточно
добавить Markdown-файл с такими разделами.

## Запуск

Нужны Python 3.12+ и [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # вписать ANTHROPIC_API_KEY
```

Задать вопрос:

```bash
uv run zeppelin-ask "Почему гелий поднимает меньше водорода?" --trace
```

Прогнать оценку RAGAS (отчёты попадут в `reports/summary.md` и `reports/ragas_scores.csv`):

```bash
uv run zeppelin-eval            # весь тестовый набор
uv run zeppelin-eval --limit 3  # быстрый прогон
```

При первом запуске fastembed скачивает модель эмбеддингов (~220 МБ).

Если проект лежит в папке, которая синхронизируется с iCloud (например, `~/Documents`),
iCloud помечает файлы `.venv` скрытыми. Python 3.12.12+ пропускает скрытые `.pth`, и
команды падают с `ModuleNotFoundError: No module named 'zeppelin_rag'`. Держите окружение
в папке, которую iCloud не синхронизирует:

```bash
rm -rf .venv && mkdir .venv.nosync && ln -s .venv.nosync .venv && uv sync
```

## Настройки

Всё задаётся переменными окружения, полный список — в [.env.example](.env.example).

| Переменная | По умолчанию | Роль |
|---|---|---|
| `ZEPPELIN_ANSWER_MODEL` | `claude-opus-5` | генерация ответа |
| `ZEPPELIN_FAST_MODEL` | `claude-haiku-4-5` | оценка фрагментов и переформулировка |
| `ZEPPELIN_JUDGE_MODEL` | `claude-sonnet-5` | судья RAGAS |
| `ZEPPELIN_TOP_K` | `4` | сколько фрагментов достаётся из поиска |
| `ZEPPELIN_LEXICAL_WEIGHT` | `0.5` | вес BM25 в гибридном поиске (0 — только эмбеддинги) |
| `ZEPPELIN_MAX_REWRITES` | `2` | лимит переформулировок запроса |

## Разработка

```bash
uv run pytest        # тесты не ходят в API: фейковые эмбеддинги и LLM
uv run ruff check .
```

CI (GitHub Actions) запускает ruff и pytest на каждый push и pull request.

## Структура

```
src/zeppelin_rag/
  knowledge.py   # Markdown -> фрагменты по разделам
  retriever.py   # гибридный поиск: fastembed + BM25
  llm.py         # вызовы Claude (Anthropic SDK)
  graph.py       # граф LangGraph: retrieve -> grade -> rewrite/generate
  evaluate.py    # оценка RAGAS
  cli.py         # zeppelin-ask
data/knowledge/  # база знаний
data/eval/       # тестовый набор
```
