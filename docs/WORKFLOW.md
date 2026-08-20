# Рабочий процесс

В корне проекта оставлены короткие команды, которые нужны каждый день.

## 1. Скачать новые реплеи

```bash
python3 download_replays.py
```

Новые JSON появятся в `data/replays/`.

## 2. Пересобрать все датасеты

```bash
python3 build_dataset.py
```

Все производные файлы появятся в `data/processed/`. Одна команда строит
переходы, признаки, Worker Dataset, Economic Dataset и Value Dataset.

Чужие публичные матчи складываются отдельно в `data/teacher_replays/`:

```bash
python3 build_teacher_dataset.py
```

Результат появится в `data/teacher_processed/`. Флаг `--winner-only` оставляет
только победителя каждого матча; без него сохраняются обе стороны.

## 3. Обучить production-политику

```bash
.venv/bin/python train_model.py
```

Скрипт сравнивает варианты экономики в полных локальных играх. Новый вариант
попадает в `artifacts/models/promoted_economic_config.json` только если
обыгрывает baseline на проверочных seed.

## 4. Проверить агента

```bash
.venv/bin/python evaluate_agents.py --agent-a main.py --agent-b agents/baseline.py --games 8
```

## 5. Собрать Kaggle submission

```bash
python3 package_submission.py
```

Готовый файл: `artifacts/submission.tar.gz`.

## Структура

- `main.py` — единственная обязательная точка входа Kaggle.
- `src/kaggriculture/core/` — правила, форматы действий, State Parser.
- `src/kaggriculture/planning/` — Task, Worker и Economic Planner.
- `src/kaggriculture/data/` — парсинг реплеев и построение датасетов.
- `src/kaggriculture/learning/` — обучение и policy search.
- `agents/` — контрольные агенты для локальных матчей.
- `data/replays/` — исходные реплеи.
- `data/processed/` — построенные датасеты.
- `artifacts/models/` — принятые модели и конфигурации.
- `artifacts/reports/` — отчёты турниров и экспериментов.
- `experiments/` — модели, которые пока не допущены в production.
- `tests/` — автоматические проверки.
