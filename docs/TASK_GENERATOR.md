# Task Generator

`TaskGenerator` получает готовый `GameState` и отвечает на вопрос: «Какие
полевые работы сейчас существуют?»

```python
from state_parser import parse_observation
from task_generator import TaskGenerator

generator = TaskGenerator()


def agent(obs):
    state = parse_observation(obs)
    tasks = generator.generate(state)

    for task in tasks:
        print(task.priority, task.operation, task.target, task.reason)

    return {"farmer": ["PASS"], "hands": [], "market": []}
```

Пример результата:

```text
1321 FEED    Position(x=4, y=2) animal_will_escape
1271 WATER   Position(x=2, y=3) plant_will_become_weed
 704 HARVEST Position(x=4, y=2) animal_has_harvest
 321 CARE    Position(x=4, y=2) animal_needs_daily_care
 100 DIG     Position(x=1, y=1) clear_weed
```

## Поля задачи

- `operation` — действие на целевой клетке: `WATER`, `FEED`, `HARVEST`,
  `CARE`, `COLLECT_FERTILIZER` или `DIG`;
- `target` — координаты клетки;
- `priority` — чем больше число, тем раньше нужно заняться задачей;
- `category` — `SURVIVAL`, `PRODUCTION` или `MAINTENANCE`;
- `reason` — короткая причина для отладки;
- `required_item` — предмет, который должен нести работник, например `WHEAT`;
- `deadline_step` — последний или уже пропущенный безопасный ход;
- `critical` — грозит ли растению, животному или урожаю потеря;
- `action` — готовая команда, которую можно выполнить после прибытия на клетку.

## Порядок приоритетов

По умолчанию работы располагаются так:

1. животное может сбежать;
2. растение может превратиться в сорняк;
3. урожай уже начал стареть;
4. обычное ежедневное кормление;
5. обычный ежедневный полив;
6. сбор продукции;
7. сбор удобрения;
8. уход за животным;
9. удаление сорняка.

Чем ближе конец дня, тем выше приоритет ежедневного полива, кормления и ухода.
Значения находятся в `TaskPriorities`, поэтому позднее их можно менять и
сравнивать в симуляциях.

## Что генератор пока не делает

Он не выбирает работника, не строит маршрут и не выполняет задачу. Одна и та же
работа будет генерироваться каждый ход, пока реальное состояние клетки не
изменится.

Он также пока не создаёт задачи для посадки, строительства, покупок и продаж.
Такие решения зависят от экономической стратегии, которой у агента ещё нет.
