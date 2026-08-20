# Behavior Cloning

Behavior Cloning учит модель повторять команды текущего rule-based агента.

## Обучение

После сборки датасета запустите:

```bash
.venv/bin/python -m src.kaggriculture.learning.train_behavior_cloning
```

Скрипт читает Worker Dataset небольшими частями, поэтому не пытается загрузить
все данные в память. Все игры из `holdout` используются только для итоговой
проверки.

Результат:

- `experiments/behavior_cloning/artifacts/worker_bc.npz` — веса модели;
- `experiments/behavior_cloning/artifacts/worker_bc_report.json` — точность и метрики по каждому действию.
- `experiments/behavior_cloning/artifacts/worker_bc_policy_report.json` — точность полной политики после маски
  допустимых действий.

Модель имеет две головы:

1. `operation` выбирает действие: движение, полив, кормление и так далее;
2. `argument` выбирает предмет или культуру для `PICKUP`, `PLACE` и `PLANT`.

Количество для команд работников во всех текущих replay равно единице, поэтому
отдельная модель количества пока не нужна.

## Использование

`src/kaggriculture/learning/behavior_model.py` загружает экспортированные веса и предсказывает команду для
каждого работника. `experiments/behavior_cloning/behavior_agent.py` является гибридным агентом:

- работниками управляет обученная модель;
- рынком пока управляет существующий `Economic Planner`;
- `Action Decoder` превращает числовой ответ в команду Kaggle.

`src/kaggriculture/core/legal_actions.py` вычеркивает физически невозможные ответы модели. Например,
при пустом складе модель не сможет выбрать `PICKUP`, даже если этот класс получил
самый высокий балл.

Это исследовательская модель, а не production-агент. Она учится копировать
текущего агента и не попадает в submission без победы в полном турнире.
