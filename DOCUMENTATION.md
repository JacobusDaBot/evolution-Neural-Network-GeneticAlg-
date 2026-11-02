# Evolution Simulation: Code Documentation

*Snapshot of the project at `D:\Coding_\evolution`, re-checked against the code on 2 November 2026 (second revision).*

---

## 1. Overview

This project is the start of an **artificial-life simulation** in which two populations of creatures live in a 2-D world containing plants, water and corpses. The long-term goal (stated in the code comments) is:

1. Every creature has a **neural network "brain"** that reads what it can see and remembers, and outputs a decision (move, turn, attack, eat, drink, procreate).
2. When creatures die, their **lifetime and weights are recorded**.
3. A **genetic algorithm** takes the creatures and world and evolves the networks, so that later generations survive longer.

This is the same neuroevolution idea as in `neuroevolution_car_summary.md` (network = driver, genetic algorithm = trainer, genome = all weights and biases flattened into one vector $\boldsymbol{\theta}$), applied to survival instead of lane-keeping.

### 1.1 Current status

| Area | State |
|---|---|
| World data model (creatures, plants, water, corpses) | Written, working |
| Vision and memory (cone of sight, remembering sightings) | **Working** (verified: creatures now remember plants and water) |
| Per-tick survival rules (hunger, thirst, death) | **Working** (no crash) |
| Creature actions (move, turn, attack, eat, drink) | Written, but nothing calls them yet (`act()` is empty) |
| Neural network (`neuralNetStuff/brain.py`) | **Not started** (comment only) |
| Genetic algorithm (`neuralNetStuff/geneticAlg.py`) | **Not started** (comment only) |
| Reproduction (`createChild`) | **Stub** |
| Rendering / drawing | **Not started** (`#TODO draw`) |
| Entry point (`main.py`) | **Runs**, but loops forever and shows nothing |

Verified by running: `World(1)` builds, `tick()` runs 50 times without error, and all 10 herbivores and 10 carnivores are still alive (hunger has fallen from 50 to 45, as expected from $50 - 50 \times 0.1$).

### 1.2 What changed since the previous version of this document

| Change | Where |
|---|---|
| `brain.py` and `geneticAlg.py` moved into a new folder `src/neuralNetStuff/` | folder structure |
| `main.py`: `"main"` corrected to `"__main__"`, and `World()` now called as `World(1)` | `main.py:8-9` |
| `lifespan+=1` corrected to `self.lifespan+=1` | `creatures.py:75` |
| `self.type` is now stored (`type=-1` class default, `self.type=type` in `__init__`) | `creatures.py:49, 58` |
| `World.tick` now calls the module function `update_memory(...)` | `world.py:108-110` |
| `getBrain` removed; `__str__` now returns an empty string (with a TODO) | `creatures.py:67-69` |
| Defaults changed: `maxSpeed = 1`, `fieldOfView = 1`, all memory capacities `5`, `memorySize = 25` | `creatures.py:13, 21, 25, 28-32` |
| `attackRange` now set per species in `__init__` (herbivore 5, carnivore 10) | `creatures.py:62, 66` |
| `gender` commented out | `creatures.py:20` |
| `tick()` locals (`creaturesAPoints` etc.) no longer stored on `self` | `world.py:97-101` |
| New TODO comments: memory budget, load start positions from a file, bound creature movement | see Section 6 |

---

## 2. Project structure

**Re-scan result: the structure has changed.** The two placeholder files moved from `src/` into `src/neuralNetStuff/`; everything else is as before.

```
evolution/
├── DOCUMENTATION.md                  this file
├── neuroevolution_car_summary.md     theory notes (not code)
├── weights/                          empty; intended for saved brains
└── src/
    ├── main.py                       entry point (simulation loop)
    ├── neuralNetStuff/               (NEW location)
    │   ├── brain.py                  neural net            [placeholder]
    │   └── geneticAlg.py             evolution of networks [placeholder]
    └── worldObjects/
        ├── world.py                  World class + vision/memory helpers
        ├── creatures.py              Creature class
        └── plants.py                 Corpse, Water, Plant classes
```

Previous layout, for comparison: `src/brain.py` and `src/geneticAlg.py` sat directly under `src/`, and `neuralNetStuff/` was an empty folder.

There are no `__init__.py` files. This works because Python 3.3+ treats the folders as namespace packages, as long as the program is started from the `evolution` folder (for example `python -m src.main`). When you import the new modules, use `src.neuralNetStuff.brain` and `src.neuralNetStuff.geneticAlg`.

**Dependencies:** `numpy` (used in `world.py`) and the standard `math` module.

---

## 3. Architecture and data flow

### 3.1 Objects

| Object | File | Role |
|---|---|---|
| `World` | `world.py` | Owns every entity, runs the simulation clock (`tick`) |
| `Creature` | `creatures.py` | An animal: stats, senses, memory, actions |
| `Plant` | `plants.py` | Static food that regrows |
| `Water` | `plants.py` | Static drinking spot |
| `Corpse` | `plants.py` | Meat left by a dead creature; rots away |

### 3.2 Two species

`World.__init__` creates two groups, each seeing the other as an enemy:

* `creaturesA`: 10 **herbivores** (`type = 0`)
* `creaturesB`: 10 **carnivores** (`type = 1`)

### 3.3 One simulation tick (`World.tick`)

| Step | What happens | Status |
|---|---|---|
| 1 | Snapshot all positions into local point lists | Working |
| 2 | Update each creature's memory of enemies, plants and water | **Working** |
| 3 | `tick()` every creature, corpse and plant (hunger, thirst, rot, regrowth) | **Working** |
| 4 | `act()` every creature | Stub (`pass`), so creatures never move or eat |
| 5 | Remove dead creatures and rotted corpses | Working |
| 6 | Draw | Not written |

The "think" step (feed senses to the neural network, get `decision`) is marked `#get NN output->think` inside `Creature.tick` and is not implemented.

### 3.4 Vision geometry

A creature at $(x, y)$ facing angle $\theta$ sees a point $(p_x, p_y)$ if it lies inside a **cone** (a wedge) in front of it. Let $\Delta x = p_x - x$ and $\Delta y = p_y - y$. Rotate into the creature's frame so that its facing direction becomes the local $x$-axis:

$$f = \Delta x \cos\theta + \Delta y \sin\theta \qquad\text{(forward distance)}$$

$$\ell = -\Delta x \sin\theta + \Delta y \cos\theta \qquad\text{(sideways offset)}$$

The point is visible when all three conditions hold:

$$f > 0, \qquad |\ell| \le m\,f, \qquad d = \sqrt{\Delta x^2 + \Delta y^2} \le R$$

Here $m$ is `fieldOfView` and is a **slope** (the cone is $y = \pm m x$ in the local frame), so the half-angle is $\alpha = \arctan m$. $R$ is `viewDistance`. The same test is used for attacking, with $R$ replaced by `attackRange`.

With the new defaults $m = 1$, so $\alpha = \arctan 1 = 45^\circ$ (a 90° wide cone), and $R = \sqrt{10 + 10 \cdot 1} = \sqrt{20} \approx 4.47$.

---

## 4. Module reference

### 4.1 `src/main.py`: entry point

```python
from src.worldObjects import world

if __name__=="__main__":
    w= world.World(1)

    while True:
        w.tick()
```

Builds a world with id 1 and ticks it forever. There are no functions. It now starts correctly (verified), but it has no stop condition, no output and no drawing, so it just spins until you interrupt it.

### 4.2 `src/neuralNetStuff/brain.py`: neural network

Contains one comment: `## is the neural net`. Nothing is implemented.

**Planned role:** map a creature's sensory vector to the `decision` output vector. In line with the neuroevolution notes, the network's weights and biases should be exposed as one flat genome vector $\boldsymbol{\theta}$ so the genetic algorithm can cross and mutate them. The `Creature.brain` attribute (commented "this is the neural net weights") is where it will plug in.

### 4.3 `src/neuralNetStuff/geneticAlg.py`: evolution

Contains one comment: `## takes all the creatures and the world and does genetic stuff to their networks`. Nothing is implemented.

**Planned role:** evaluate fitness (for example lifetime, which the creature file says must be recorded on death), select parents, apply crossover and mutation to the genomes, and produce the next generation.

### 4.4 `src/worldObjects/world.py`: the world

#### Module-level helper functions

##### `merge_memory(new, old, max_n)`

| | |
|---|---|
| **Purpose** | Combine freshly seen positions with older remembered ones, keeping a bounded memory. |
| **Parameters** | `new`: list of `(x, y)` tuples seen this tick. `old`: list of positions remembered from earlier ticks. `max_n`: memory capacity. |
| **Returns** | A new list: all of `new` first, then the entries of `old` that are not already in `new`, truncated to `max_n`. |

Behaviour: `seen = set(new)`; an old entry is skipped if `tuple(p)` is in `seen` (a re-sighting of the same spot). Because `new` comes first, current sightings always outrank old ones, and the oldest memories fall off the end first. This is how a creature "remembers things after it stops seeing them".

Verified example: `merge_memory([(1,1)], [(1,1),(2,2),(3,3)], 2)` returns `[(1, 1), (2, 2)]`.

Note: duplicates *inside* `old` are not removed, and memories are never invalidated (a remembered plant stays remembered after it is eaten, and a remembered enemy stays where it was last seen).

##### `cone_candidates(creatures, pts, mem_counts, chunk=512)`

| | |
|---|---|
| **Purpose** | For every creature, find which of a set of points it can currently see. |
| **Parameters** | `creatures`: list of `Creature`. `pts`: list of `(x, y)` points (plants, enemies, water...). `mem_counts`: per-creature maximum number of points to return (index-aligned with `creatures`). `chunk`: how many creatures to process per numpy batch. |
| **Returns** | A list (one entry per creature) of lists of `(x, y)` tuples, **nearest first**, capped at that creature's `mem_counts[i]`. |

Algorithm (vectorised with numpy):

1. If `pts` is empty, return an empty list for every creature.
2. For each chunk of up to 512 creatures, build column vectors of `posX`, `posY`, `direction`, `fieldOfView` and `viewDistance`.
3. Compute $\Delta x$, $\Delta y$, forward $f$, lateral $\ell$ and distance $d$ as matrices of shape (creatures in chunk) $\times$ (points).
4. Build the boolean mask `ok = (f > 0) & (|l| <= fov*f) & (d <= view)`.
5. Set `dist[~ok] = inf`, then `argsort` each row so visible points come first, nearest first.
6. Keep the first `n = mem_counts[...]` indices and drop any whose distance is still `inf` (not visible).

Cost per chunk is $O(512 \cdot P)$ memory and $O(512 \cdot P \log P)$ time for $P$ points; chunking stops the matrices from growing with the whole population. A creature never sees itself, because a point at its own position has $f = 0$ and the test needs $f > 0$.

Verified behaviour: with `fieldOfView = 1`, facing along $+x$, points at $(1,0)$, $(5,0)$ and $(0.5,0.5)$ are seen when the view distance allows, while $(-1,0)$ (behind) and $(3,3.5)$ (too far to the side) are not.

Observed in the live world: after 50 ticks a herbivore starting at $(0,0)$ remembers the five nearest plants `(1,0), (1,1), (2,0), (2,1), (2,2)` and two water sources `(1,0), (1,1)`. Its enemy memory is empty, because every creature starts at the same point, so every enemy has $f = 0$ (see Issue 15).

##### `update_memory(group, pts, mem_attr, last_attr)`

| | |
|---|---|
| **Purpose** | Refresh one kind of memory for a whole group of creatures. |
| **Parameters** | `group`: list of creatures. `pts`: points to look for. `mem_attr`: name of the capacity attribute (for example `"numberOfPlantMemory"`). `last_attr`: name of the memory list attribute (for example `"lastPlantsSeen"`). |
| **Returns** | Nothing; it writes to each creature with `setattr`. |

Steps: read each creature's capacity via `getattr(c, mem_attr)`, call `cone_candidates`, then for each creature store `merge_memory(new, old, capacity)` back into `last_attr`. Using `setattr` creates a fresh per-instance list, so creatures do not share memory lists.

Capacities now default to 5, so memory is no longer empty.

#### Class `World`

**Class attributes:** `creaturesA`, `creaturesB`, `plants`, `waterSources`, `corpses` (all lists), `width = 500`, `height = 500`.

`width` and `height` are not used anywhere yet, so nothing is bounded or drawn (there is now a TODO for this).

##### `World.__init__(self, id)`

Stores `id`, then creates:

* 10 herbivores `Creature(i, 0)` and 10 carnivores `Creature(i, 1)` for $i = 0..9$ (ids repeat across the two groups),
* 25 plants on the integer grid $x, y \in \{0,\dots,4\}$,
* 4 water sources at $x, y \in \{0, 1\}$.

All creatures start at $(0, 0)$ facing $\theta = 0$, and everything sits in the bottom-left corner of the $500 \times 500$ area. `self.corpses` is not created here (see Issue 11).

##### `World.makeCorpse(self, posX, posY)`

Appends `Corpse(0, posX, posY)` to the corpse list. The id is always 0. Nothing calls this yet, so dead creatures leave no corpse.

##### `World.createChild(self, type, brain1, brain2)`

Intended to create a child of species `type` from two parents' brains (crossover plus mutation). The body is `pass`, so it does nothing and returns `None`. It is called from `Creature.Procreate`.

##### `World.tick(self)`

Advances the world by one step, as in Section 3.3. In detail:

1. Builds the local lists `creaturesAPoints`, `creaturesBPoints`, `plantsPoints`, `waterSourcesPoints`, `corpsesPoints` (no longer stored on `self`; `corpsesPoints` is built but not used).
2. For each of `(creaturesA, B's points)` and `(creaturesB, A's points)`, updates three memories: enemies, plants and water. Allies and corpses are **not** updated.
3. Calls `tick()` on all creatures, corpses and plants.
4. Calls `act()` on all creatures.
5. Rebuilds the creature and corpse lists, dropping anything with `remove == True`. Plants are never removed.

### 4.5 `src/worldObjects/creatures.py`: the creature

Module-level: `memorySize = 25` ("how many things it can remember at once"). It is still unused; a TODO (line 27) says the five memory capacities should be constrained so their sum stays below it. Currently $5 \times 5 = 25$ exactly equals it.

#### Class `Creature`: attributes

**Permanent stats** (meant to be set once, probably from genes):

| Attribute | Default | Meaning |
|---|---|---|
| `id` | 0 | Identifier |
| `type` | -1 (set to 0 or 1 in `__init__`) | Species: 0 herbivore, 1 carnivore |
| `maxSpeed` | **1** (was 0) | Distance moved per `move()` |
| `foodFromPlant` | 0 | Hunger gained per bite of plant (see Issue 6) |
| `foodFromMeat` | 0 | Hunger gained from a corpse (see Issue 6) |
| `fieldOfView` | **1** (was 0) | Cone slope $m$ |
| `viewDistance` | $\sqrt{20}\approx 4.472$ (fixed at class creation; see Issue 7) | Sight range $R$ |
| `numberOfAlliesMemory`, `numberOfEnemiesMemory`, `numberOfPlantMemory`, `numberOfWaterMemory`, `numberOfCorpseMemory` | **5** each (were 0) | Memory capacities |
| `attackPower` | 0 | Damage dealt per attack (still 0, so attacks do nothing) |
| `attackRange` | 0 default; **5 herbivore, 10 carnivore** after `__init__` | Reach of attacks and of eating/drinking |

The `gender` attribute is now commented out.

**Per-tick state:**

| Attribute | Default | Meaning |
|---|---|---|
| `remove` | `False` | True when the creature is dead |
| `lifespan` | 0 | Ticks survived |
| `hunger`, `thirst`, `health` | 50 (set in `__init__`) | Out of 100. **Note:** hunger and thirst count *down* to death, so a high number is good. |
| `posX`, `posY` | 0 | Position (a TODO says to load these from a file) |
| `direction` | 0 | Facing angle $\theta$ in radians |
| `lastEnemiesSeen`, `lastAlliesSeen`, `lastPlantsSeen`, `lastWaterSeen`, `lastCorpsesSeen` | `[]` | Remembered `(x, y)` sightings |
| `brain` | `[]` | The neural network weights (future) |
| `decision` | `[]` | The network's output vector (future) |

#### Class `Creature`: methods

##### `__init__(self, id, type)`

Sets `id`, `type`, and `health`, `hunger`, `thirst` to 50. Then by `type`:

| `type` | Species | plant food | meat food | attack range |
|---|---|---|---|---|
| 0 | herbivore | 1 | 0.1 | 5 |
| 1 | carnivore | 0.6 | 10 | 10 |

The food values are still assigned to `FoodFromPlant`/`FoodFromMeat` with a capital F, so the lower-case attributes the rest of the code reads stay 0 (Issue 6). The parameter name `type` still shadows the Python built-in.

##### `__str__(self)`

Currently returns an empty string (`f""`), with a `#TODO complete`. The old `getBrain` helper that it called has been removed.

##### `tick(self)`: think and live one step

1. If `health <= 0`, set `remove = True` and return.
2. Increment `self.lifespan`.
3. Decrease hunger and thirst:

$$h_{t+1} = h_t - 0.1, \qquad w_{t+1} = w_t - 0.1$$

4. If hunger or thirst is $\le 0$, set `health = 0` and `remove = True`.
5. The thinking step (run network, process output) is a comment only.

From the starting value of 50, a creature that never eats or drinks dies after about $50 / 0.1 = 500$ ticks. Because of Issue 6 and the empty `act()`, that is currently the fate of every creature.

##### `act(self)`

Placeholder (`pass`). It should use `decision` to call one of the action methods below.

##### `damage(self, amount)`

`health -= amount`. Health is not clamped, and death is processed on the *next* `tick`.

##### `move(self)`

$$x \leftarrow x + v_{\max}\cos\theta, \qquad y \leftarrow y + v_{\max}\sin\theta$$

Moves at full speed ($v_{\max} = 1$ by default) along the facing direction. There is no world boundary check yet (a TODO in `world.py` covers this).

##### `turnL(self)` and `turnR(self)`

Change the facing angle by a fixed step:

$$\theta \leftarrow \theta \mp \frac{\pi}{9} \quad(\text{20 degrees})$$

`turnL` subtracts and `turnR` adds. Neither wraps $\theta$ into $[0, 2\pi)$, which does not matter for $\sin$/$\cos$.

##### `attack(self, enemies)`

Damages every enemy inside the attack cone. For each enemy `e` it computes forward $f$ and lateral $\ell$ exactly as in Section 3.4, skips it if $f \le 0$ or $|\ell| > m f$ or the distance exceeds `attackRange`, and otherwise calls `e.damage(self.attackPower)`.

It hits **all** qualifying enemies in one call, and it does not check whether the target is already dead. With `attackPower = 0` it currently deals no damage.

##### `eatPlant(self, plants)`

Finds the **first** plant within `attackRange` and, for that one plant, does:

$$\text{hunger} \leftarrow \min\!\big(100,\ \text{hunger} + \texttt{plant.tryEat}(\texttt{foodFromPlant})\big)$$

then stops (`break`). The plant's `tryEat` returns how much was actually available. Verified: because `foodFromPlant` is 0, hunger does not rise.

##### `eatCorpse(self, corpses)`

Finds the first corpse within `attackRange` and adds the flat `foodFromMeat` to hunger (capped at 100). It does **not** call `Corpse.tryEat`, so the corpse is never depleted (Issue 10). The inline comment still says "plant" (copy-paste).

##### `drinkWater(self, waterSources)`

Finds the first water source within `attackRange` and adds 2 to thirst, capped at 100. Water is never depleted.

##### `Procreate(self, world, allies)`

"Easy mode": finds the first ally within `attackRange` and calls `world.createChild(self.type, self.brain, p.brain)`, then stops. A commented-out "advanced" version (which used `gender`) restricts mating to male-to-female and damages the partner.

`self.type` now exists, so the call no longer crashes on that. But `createChild` is a stub, and if `allies` contains the creature itself (distance 0) it can pair with itself (Issue 14).

### 4.6 `src/worldObjects/plants.py`: plants, water and corpses

Despite the filename, this file holds three classes. (Only a comment was removed here since the last version; behaviour is unchanged.)

#### Class `Corpse`

Fields: `id`, `posX`, `posY`, `health` (default 0), `remove` (default `False`).

* `__init__(self, id, posX, posY)`: stores the id and position. Health is **not** set, so it stays 0 (Issue 9).
* `tick(self)`: `health -= 1`; when `health <= 0`, sets `remove = True` (rotting).
* `tryEat(self, damage)`: takes up to `damage` from the corpse.
  * If `health > damage`: reduce health by `damage`, return `damage`.
  * Otherwise: mark `remove = True`, return the remaining health, set health to 0.

#### Class `Water`

Fields: `posX`, `posY`.

* `__init__(self, posX, posY)`: stores the position. Static and infinite.

#### Class `Plant`

Fields: `posX`, `posY`, `health` (set to 50 on creation).

* `__init__(self, posX, posY)`: stores the position, `health = 50`.
* `tick(self)`: regrows, $H_{t+1} = H_t + 0.5$. There is no upper cap, so a plant that is never eaten grows without limit.
* `tryEat(self, damage)`: like `Corpse.tryEat` but the plant is never removed. If `health > damage` it loses `damage` and returns it, otherwise it returns its remaining health and drops to 0 (then regrows).

---

## 5. Default parameters at a glance

| Quantity | Value | Where |
|---|---|---|
| World size | $500 \times 500$ (unused) | `World` |
| Herbivores / carnivores | 10 / 10 | `World.__init__` |
| Plants / water sources | 25 / 4 | `World.__init__` |
| Starting position and direction | all creatures at $(0,0)$, $\theta = 0$ | class defaults |
| Starting health, hunger, thirst | 50 | `Creature.__init__` |
| Hunger and thirst decay | $0.1$ per tick | `Creature.tick` |
| Max speed | 1 | `Creature` |
| Field of view slope $m$ | 1 (45° half-angle) | `Creature` |
| View distance $R$ | $\sqrt{20}\approx 4.47$ | `Creature` |
| Memory slots per type | 5 (total budget `memorySize = 25`, unenforced) | `Creature` |
| Attack range | herbivore 5, carnivore 10 | `Creature.__init__` |
| Attack power | 0 | `Creature` |
| Turn step | $\pi/9$ (20°) | `turnL`, `turnR` |
| Plant regrowth | $0.5$ per tick | `Plant.tick` |
| Corpse rot | 1 per tick | `Corpse.tick` |
| Water gained per drink | 2 | `drinkWater` |
| Food per bite (intended) | herbivore 1 plant / 0.1 meat; carnivore 0.6 plant / 10 meat | `Creature.__init__` |
| Vision chunk size | 512 creatures | `cone_candidates` |

---

## 6. TODO list

### 6.1 TODO comments written in the code

| # | File and line | TODO |
|---|---|---|
| 1 | `world.py:12` | Load a template of creatures, plants and water so every generation has the same start. |
| 2 | `world.py:73` | Actually restrict creature movement to the world bounds ($0..500$). |
| 3 | `world.py:114` | All creatures think (creatures A). |
| 4 | `world.py:116` | All creatures think (creatures B). |
| 5 | `world.py:123` | All creatures act. |
| 6 | `world.py:128` | Keep old (dead) creatures so they can be mutated and evolved. |
| 7 | `world.py:132` | Draw the world. |
| 8 | `creatures.py:27` | Implement `memorySize` so that the sum of all memory types must be less than it. |
| 9 | `creatures.py:57` | Initialise `posX`, `posY` and `direction` from a file. |
| 10 | `creatures.py:68` | Complete `__str__`. |
| 11 | `creatures.py:84` | Get the neural-network output and think (`#get NN output->think`). |
| 12 | `creatures.py:88` | `act()`: use `decision` to choose a function to run. |
| 13 | `creatures.py:5` (header) | On death, record the lifetime and weights of the creature. |
| 14 | `creatures.py:161-169` | The "advanced" `Procreate` (male/female mating) is commented out. |

### 6.2 Unfinished features (implied by the code, no TODO written)

| # | Item | Notes |
|---|---|---|
| 15 | **Neural network** (`neuralNetStuff/brain.py`) | Empty. Needs input vector, forward pass, flat genome get/set. |
| 16 | **Genetic algorithm** (`neuralNetStuff/geneticAlg.py`) | Empty. Needs fitness, selection, crossover, mutation, elitism. |
| 17 | `World.createChild` | Stub (`pass`). |
| 18 | Dead creatures do not become corpses | `makeCorpse` is never called; `Corpse` also starts with health 0. |
| 19 | `act()` is empty | So `move`, `turnL`, `turnR`, `attack`, `eatPlant`, `eatCorpse`, `drinkWater` and `Procreate` are never used. |
| 20 | Allies and corpse memory not updated | `World.tick` only handles enemies, plants and water. |
| 21 | `main.py` has no stop condition, output or drawing | Loops forever (verified: still running after a 5 second timeout). |
| 22 | `weights/` folder is empty | Nothing saves or loads brains yet. |
| 23 | `attackPower` is 0 | Attacks are inert until it is set per species or from genes. |
| 24 | Mixed species behaviour | Carnivores and herbivores differ only in the food and range constants. |

---

## 7. Issues: status after re-check

Each item from the previous version was re-tested against the current code. **Fixed: 8. Still open: 6. New: 3.** (Issues 1 to 14 keep their original numbers.)

### 7.1 Fixed

| # | Issue | How it was fixed |
|---|---|---|
| 1 | `__name__=="main"` in `main.py` | Now `"__main__"`. Verified: the simulation starts. |
| 2 | `World()` called without its `id` | Now `World(1)`. (`World()` with no argument still raises `TypeError`, which is fine as long as callers pass an id.) |
| 3 | `lifespan+=1` crash | Now `self.lifespan+=1` (`creatures.py:75`). Verified: `tick` increments it. |
| 4 | `self.type` never stored | `type=-1` class default and `self.type=type` in `__init__`. Verified: herbivore has `type == 0`. |
| 5 | `self._update_memory` does not exist | `World.tick` now calls `update_memory(...)`. Verified: 50 ticks run cleanly. |
| 8 | `getBrain` used missing attributes | `getBrain` removed and `__str__` returns `""`. Resolved, but see New Issue 17. |
| 12 | Zero-width cone (`fieldOfView = 0`) | Default is now 1 (45° half-angle). Verified: creatures see plants and water. |
| 13 | Memory capacities all 0 | Now 5 each. Verified: memories fill up. |

### 7.2 Still open

| # | File and line | Problem | Effect | Suggested fix |
|---|---|---|---|---|
| 6 | `creatures.py:60-61, 64-65` | Sets `FoodFromPlant`/`FoodFromMeat` (capital F), but `eatPlant`/`eatCorpse` read `foodFromPlant`/`foodFromMeat` | Food stays 0. Verified: `foodFromPlant == 0` while `FoodFromPlant == 1`, and hunger does not rise when eating. | Rename to lower-case. |
| 7 | `creatures.py:26` | `viewDistance=sqrt(10+10*fieldOfView)` is evaluated once, when the class is defined | Now a constant $\sqrt{20}$ (matches the default $m = 1$ but will not follow a creature's own $m$ if genes change it). | Compute in `__init__` or as a property: $R = \sqrt{10 + 10m}$. |
| 9 | `plants.py:7` | `Corpse.health` is never set (0) | A corpse is flagged for removal on its first tick. Verified: health becomes $-1$, `remove = True`. | Pass or derive a starting health. |
| 10 | `creatures.py:143` | `eatCorpse` adds a flat amount and never calls `Corpse.tryEat` | A corpse is an unlimited food source until it rots. | `hunger += p.tryEat(self.foodFromMeat)`. |
| 11 | `world.py:68-72, 83-84` | Lists are class attributes; `corpses` is not created in `__init__`, so `makeCorpse` appends to a list shared by all `World` objects until `tick` reassigns it | Verified: a corpse made on one `World` appears in `World.corpses`. Hidden shared state. | Create every list in `__init__`. |
| 14 | `creatures.py:171-174` | `Procreate` can pair a creature with itself if it is in `allies` | Self-breeding. | Skip `p is self`. |

### 7.3 New issues found in this re-check

| # | File and line | Problem | Effect | Suggested fix |
|---|---|---|---|---|
| 15 | `creatures.py:41-43`, `world.py:78-79` | Every creature starts at $(0,0)$ facing $\theta = 0$ | Every enemy has forward distance $f = 0$, so enemies are never seen. Verified: `lastEnemiesSeen` is empty after 50 ticks. | Spread start positions and directions (the TODO at `creatures.py:57` already plans a file). |
| 16 | `creatures.py:87-89` with 6 and 23 | `act()` is empty and `attackPower = 0` | Nobody moves, eats, drinks or fights, so every creature starves at about tick 500. | Implement `act()` and set `attackPower`. |
| 17 | `creatures.py:67-69` | `__str__` returns an empty string, and the sensory-vector builder (`getBrain`) was removed | `print(creature)` shows nothing, and there is no function yet that builds the network input. | Write `__str__`, and add an input-vector method for the brain. |

### 7.4 Design observations (not crashes, unchanged)

* Mutable class-level lists on `Creature` (`lastEnemiesSeen`, `brain`, `decision`, ...) are shared between instances if anything ever mutates them in place. Verified: `Creature.lastAlliesSeen is Creature(3,0).lastAlliesSeen` is `True`. `update_memory` is safe because it reassigns.
* `Creature.attack` and `cone_candidates` repeat the same cone geometry; a shared helper would remove the duplication.
* `numpy` is still imported twice in `world.py`, and `creatures.py` imports `sqrt` and also `math`.
* Creature ids repeat across groups A and B (both use 0-9).
* `memorySize` is defined but unused; `corpsesPoints` is built in `World.tick` but unused; `width` and `height` are unused.

---

## 8. Suggested route to a working version

1. **Fix the open basics:** Issues 6, 9, 10, 11, 14 (small edits each).
2. **Spread the creatures out** (Issue 15) so enemies can actually be seen.
3. **Write `neuralNetStuff/brain.py`:** a small feed-forward network with a `forward(inputs)` method and `get_genome()` / `set_genome(theta)` so the weights flatten to a vector $\boldsymbol{\theta}$ of length $P = \sum_l n_{l+1}(n_l + 1)$.
4. **Build the input vector** from state and memories, run the brain into `decision`, and implement `act()` to pick an action.
5. **Record deaths:** on removal, store `(lifespan, genome)` (the `weights/` folder is ready for this) and spawn a `Corpse` with a real health value.
6. **Write `neuralNetStuff/geneticAlg.py`:** fitness (for example lifespan, possibly plus food eaten), parent selection, crossover, Gaussian mutation and elitism, then refill the world from the new genomes.
7. **Add rendering** and a generation counter in `main.py`, with a stop condition instead of `while True`.

---

## 9. Quick-start

```bash
cd D:\Coding_\evolution
pip install numpy
python -m src.main      # runs forever; stop with Ctrl+C
```

```python
from src.worldObjects import world

w = world.World(1)
for _ in range(1000):
    w.tick()
print(len(w.creaturesA), len(w.creaturesB))   # survivors per species
```

With the current code, that second snippet prints `0 0` after 1000 ticks, because nobody can eat or drink yet (see Issues 6 and 16): every creature starves at about tick 500.
