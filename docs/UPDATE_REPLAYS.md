# Обновление данных

Рабочий процесс состоит ровно из двух команд. Обе команды запускаются из папки
проекта.

## 1. Скачать новые replay

```bash
python3 download_replays.py
```

Команда спрашивает Kaggle о завершенных играх отправки `55562698`, сравнивает их
с папкой `replays` и скачивает только отсутствующие файлы. Уже скачанные replay
она не трогает. OAuth-вход повторять не нужно.

## 2. Собрать датасет

```bash
python3 build_dataset.py
```

Команда берет все JSON-файлы из папки `replays` и полностью пересобирает:

- `data/processed/transitions.jsonl.gz` — состояния, действия и следующие
  состояния;
- `data/processed/episodes.csv` — краткие результаты каждой игры;
- `data/processed/features.jsonl.gz` — числовые признаки для нейросети;
- `data/processed/worker_dataset.jsonl.gz` — признаки и правильная команда для
  каждого работника;
- `action_schema.json` и `worker_feature_schema.json` — словари команд и
  признаков работников;
- файлы `manifest.json`, `feature_manifest.json` и `feature_schema.json`.

## Всегда один и тот же порядок

```bash
python3 download_replays.py
python3 build_dataset.py
```

Первая команда работает с Kaggle и скачивает replay. Вторая команда не обращается
к Kaggle, а только превращает локальные replay в датасет.

## Одноразовая установка Kaggle CLI

OAuth уже сохранен. Если после перезагрузки появится сообщение
`Kaggle CLI was not found`, один раз выполните:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U kaggle
```

После этого две основные команды по-прежнему запускаются через обычный
`python3` и не меняются.
