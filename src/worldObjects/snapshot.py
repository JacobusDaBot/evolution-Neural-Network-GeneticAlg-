from dataclasses import dataclass

@dataclass(frozen=True)
class ObjectView:
    kind: str        # "creatureA", "creatureB", "plant", "water", "corpse"
    x: float
    y: float
    radius: float = 1.0

@dataclass(frozen=True)
class CreatureView:
    id: int
    kind: str          # "creatureA" / "creatureB"
    health: float
    hunger: float
    thirst: float
    numChildren: int
    decision: tuple    # plain floats, empty before the creature's first tick

@dataclass(frozen=True)
class WorldSnapshot:
    tick: int
    width: float
    height: float
    objects: tuple     # tuple[ObjectView, ...]
    creatures: tuple   # tuple[CreatureView, ...]