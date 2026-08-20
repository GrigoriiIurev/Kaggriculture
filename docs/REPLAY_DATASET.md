# Replay Dataset

`src/kaggriculture/data/outcome_logger.py` превращает загруженные replay в два удобных набора данных.

## Запуск

```bash
python3 build_dataset.py
```

Имя нашей команды определяется автоматически: это имя, которое встречается в
файлах чаще всех. При необходимости его можно указать явно:

```bash
python3 build_dataset.py
```

## Результат

- `data/processed/transitions.jsonl.gz` — обучающие примеры. Каждая строка содержит состояние
  `observation`, сделанное агентом `action`, следующее состояние
  `next_observation` и результат всей игры.
- `data/processed/episodes.csv` — одна понятная строка на каждую нашу игровую траекторию:
  победа или поражение, итоговые деньги, купленные семена и животные,
  неиспользованные семена, примерное число потерянных животных и счетчики
  действий.
- `data/processed/manifest.json` — сколько файлов, игр, траекторий и ходов вошло в набор.

Архив `.gz` читается напрямую из Python:

```python
import gzip
import json

with gzip.open("data/processed/transitions.jsonl.gz", "rt") as source:
    first_example = json.loads(next(source))

print(first_example["observation"])
print(first_example["action"])
print(first_example["outcome"])
```

## Почему действие берется из следующего кадра

В формате Kaggle действие, принятое по состоянию в кадре `t`, сохраняется в
кадре `t + 1`. Парсер поэтому собирает пример так:

```text
steps[t].observation -> steps[t + 1].action -> steps[t + 1].observation
```

Служебное поле `remainingOverageTime` удаляется. Для второго игрока поле `step`
в replay отсутствует, поэтому оно восстанавливается как `day * turnsPerDay + hour`.
В строку попадает только наблюдение выбранного игрока: закрытый приватный
инвентарь противника к нему не подмешивается.
