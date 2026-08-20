# State Parser

`OBS` от Kaggle является большим набором вложенных словарей. `src/kaggriculture/core/state_parser.py`
один раз разбирает его и возвращает объект `GameState` с понятными именами.

## Подключение

```python
from state_parser import parse_observation


def agent(obs, configuration=None):
    state = parse_observation(obs, configuration)

    print(state.step)                  # текущий ход
    print(state.me.money)              # наши деньги
    print(state.opponent.money)        # деньги противника
    print(state.private.shed)          # наш сарай
    print(state.market.prices)         # текущие цены
    print(state.town.unlocked_shops)   # открытые магазины

    return {"farmer": ["PASS"], "hands": [], "market": []}
```

## Клетки фермы

```python
for plant in state.me.plants:
    print(plant.crop, plant.position, plant.yield_units)

for animal in state.me.animals:
    print(animal.animal, animal.position, animal.yield_units)

for weed in state.me.weeds:
    print("Сорняк:", weed.position)
```

У клетки есть готовые проверки:

```python
plant.needs_water
animal.needs_feed
tile.is_empty
tile.is_locked
tile.has_animal
```

Конкретную клетку можно получить по координатам:

```python
from state_parser import Position

tile = state.me.tile_at(Position(x=3, y=4))
```

## Работники

`state.units[0]` всегда является главным фермером. Остальные элементы являются
нанятыми работниками в том же порядке, в котором для них нужно вернуть команды.

```python
for unit in state.units:
    print(unit.index, unit.position, unit.inventory)
```

## Производные значения

Parser уже вычисляет некоторые часто используемые значения:

```python
state.private.shed_used    # сколько места занято в сарае
state.private.shed_free    # сколько места осталось
state.remaining_steps      # сколько ходов осталось вместе с текущим
state.is_last_step         # последний ли это ход
state.town.shop_counts     # количество магазинов каждого типа
```

Parser не выбирает действия и не является стратегией. Его задача только в том,
чтобы превратить внешний формат Kaggle в надёжное внутреннее состояние агента.
