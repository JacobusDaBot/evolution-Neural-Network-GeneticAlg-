# Evolution Simulation: Code Documentation

*Snapshot of the project at `D:\Coding_\evolution`, re-scanned against the code on 5 November 2026 (third revision, updated the same day for memory time limits and the Life sidebar column). Bugs are deliberately not analysed in this revision; Section 9 only lists them briefly so they are not forgotten.*

---

## 1. Overview

This is an **artificial-life / neuroevolution simulation**. Two species of creatures (herbivores and carnivores) live in a $500 \times 500$ world containing plants, water sources and corpses.

1. Every creature has a **feed-forward neural network "brain"** that reads its own state plus what it remembers seeing, and outputs an 8-number **decision** vector (turn, move, and one of attack / eat plant / eat corpse / drink / procreate).
2. When creatures die they are kept in a "dead" list together with their brain, lifespan and number of children.
3. When a world ends, a **genetic algorithm** scores the dead creatures, selects parents and breeds a new set of brains, which seed the next world (the next generation).
4. A **pygame window** shows the world live, with a sidebar listing every creature's stats and network outputs.

This is the same idea as in `neuroevolution_car_summary.md` (network = driver, genetic algorithm = trainer, genome = all weights and biases flattened into one vector $\boldsymbol{\theta}$), applied to survival instead of lane-keeping.

### 1.1 Current status

| Area | State |
|---|---|
| World data model (creatures, plants, water, corpses) | Working |
| Vision and memory (enemies, plants, water, corpses) | Working |
| Neural network (`neuralNetStuff/brain.py`) | **Implemented** |
| Sensory input vector (`Creature.inputVector`) | **Implemented** (58 inputs) |
| Think step (network runs every tick) | **Implemented** |
| `act()` (decision → actions) | **Implemented** |
| Reproduction (`procreate` → `World.createChild`) | **Implemented** |
| Death → corpse, dead creatures recorded | **Implemented** |
| Genetic algorithm (`neuralNetStuff/geneticAlg.py`) | **Implemented** (strategy B: one world, unique creatures) |
| Generation loop | **Implemented** in `main.py` (30 worlds) |
| Rendering | **Implemented** with pygame, on its own thread |
| Saving brains to `weights/` | Not started |
| Ally memory | Not updated (memory slots exist, stay empty) |
| World bounds | Not enforced |

Verified by running headless (without the UI, random seed 0): `World(0)` builds 20 herbivores and 20 carnivores, the network has 58 inputs and a genome of 7155 numbers, creatures move, fight, eat and breed (27 children in total), and the world goes extinct at tick 2501. `hasEnded()` then returns two lists of 20 new brains.

### 1.2 What changed since the previous version of this document

The code has grown from a skeleton into a full simulation loop. In summary:

| Change | Where |
|---|---|
| `NeuralNetwork` class written (genome ↔ weight matrices, `tanh` forward pass) | `neuralNetStuff/brain.py` |
| Genetic algorithm written: `fitness`, `makeChildBrain`, `getBest` | `neuralNetStuff/geneticAlg.py` |
| New file `snapshot.py`: frozen dataclasses passed from simulation to UI | `worldObjects/snapshot.py` |
| `main.py` rewritten: pygame UI, simulation on a worker thread, 30-generation loop | `main.py` |
| `Creature.__init__` rewritten: takes `world`, `brain`, position, health, water, food; random start positions; all fields are now instance attributes | `creatures.py:33-97` |
| Network shape and input size defined as module constants | `creatures.py:19-30` |
| `inputVector()` and `_flatten_memory()` added | `creatures.py:99-124` |
| `isDead()` added; `tick()` now runs the network; hunger/thirst decay lowered to 0.01 | `creatures.py:131-152` |
| `act()` implemented | `creatures.py:154-170` |
| Action methods no longer take arguments; they read lists from `self.world` (or memory, for water) | `creatures.py:188-270` |
| `attackPower` set per species (10 / 20); `fieldOfView` and `viewDistance` set per species | `creatures.py:83-96` |
| `Procreate` renamed `procreate`; "advanced" version is active, with health/hunger/thirst costs | `creatures.py:250-279` |
| `World.__init__` takes `brainsA`, `brainsB`, `initialCount`; lists are now instance attributes | `world.py:72-97` |
| Plants (81) and water (49) spread on grids across the whole world | `world.py:89-92` |
| `World.snapshot()`, `createChild()` (real), `makeCorpse()` (uses creature health), `hasEnded()` added | `world.py:99-194` |
| `World.tick()` reordered: removes the dead first, creates corpses, adds newborns; corpse memory now updated | `world.py:135-179` |
| `Corpse.__init__` takes a `health` argument | `plants.py:4-10` |
| Imports changed to `from worldObjects ...` / `from neuralNetStuff ...`, so the program is run from inside `src/` | all files |
| The project is now a git repository (4 commits) | `.git/` |

---

## 2. Project structure

```
evolution/
├── DOCUMENTATION.md                  this file
├── neuroevolution_car_summary.md     theory notes (not code)
├── weights/                          empty; intended for saved brains
└── src/
    ├── main.py                       entry point: UI thread + simulation thread + generation loop
    ├── neuralNetStuff/
    │   ├── brain.py                  NeuralNetwork class
    │   └── geneticAlg.py             fitness, crossover/mutation, selection
    └── worldObjects/
        ├── world.py                  World class + vision/memory helpers
        ├── creatures.py              Creature class, network shape constants
        ├── plants.py                 Corpse, Water, Plant classes
        └── snapshot.py               ObjectView, CreatureView, WorldSnapshot (NEW)
```

`__pycache__` folders show the code is being run with **Python 3.14**.

**Imports.** All imports are now relative to `src/` (for example `from worldObjects import world`, `from neuralNetStuff.brain import NeuralNetwork`), so the program must be started from inside `src`:

```bash
cd D:\Coding_\evolution\src
python main.py
```

**Dependencies:** `numpy`, `pygame`, and the standard library (`math`, `random`, `threading`, `time`, `dataclasses`, `typing`).

**Import graph:**

```
main.py ──► worldObjects.world ──► worldObjects.creatures ──► neuralNetStuff.brain
                 │                         ▲
                 ├──► worldObjects.plants  │
                 ├──► worldObjects.snapshot│
                 └──► neuralNetStuff.geneticAlg ──┘ (imports Creature and NeuralNetwork)
```

`creatures.py` imports `World` only under `TYPE_CHECKING`, to avoid a circular import.

**Version control.** The folder is a git repository on branch `main` with a remote `origin`. Commits so far: "first commit", "stuff", "made neural net , ui, started genetic", "try fix the bug-unsuccessful".

---

## 3. Architecture and data flow

### 3.1 Objects

| Object | File | Role |
|---|---|---|
| `World` | `world.py` | Owns every entity, runs the clock (`tick`), makes snapshots, runs the GA at the end |
| `Creature` | `creatures.py` | An animal: stats, senses, memory, brain, actions |
| `NeuralNetwork` | `brain.py` | The brain: flat genome → layers → forward pass |
| `Plant` | `plants.py` | Static food that regrows |
| `Water` | `plants.py` | Static, infinite drinking spot |
| `Corpse` | `plants.py` | Meat left by a dead creature; rots away |
| `ObjectView`, `CreatureView`, `WorldSnapshot` | `snapshot.py` | Read-only copies of the world for the UI |
| `SharedState` | `main.py` | Thread-safe mailbox between simulation and UI |

### 3.2 Two species

| | Group | `type` | Colour in UI | Sidebar tag |
|---|---|---|---|---|
| Herbivores | `creaturesA` | 0 | red | `A<id>` |
| Carnivores | `creaturesB` | 1 | blue | `B<id>` |

Each group sees the other as enemies (for memory).

### 3.3 Program flow (`main.py`)

```
for generation i = 0 .. 29:
    build World i         (i == 1: 50 fresh random creatures per species;
                           otherwise: brains from the previous generation, or random if None)
    start sim_loop(world) on a daemon thread
    ui_loop() on the main thread  ── draws snapshots at 60 fps
        stops when the window is closed or no creatures are left
    stop and join the sim thread
    brains1, brains2 = world.hasEnded()   (the GA's new brains)
```

### 3.4 Threads

| Thread | Function | Does |
|---|---|---|
| Worker (daemon) | `sim_loop` | `w.tick()` in a loop, sleeping 0.01 s per tick (so at most about 100 ticks/s). Publishes a `WorldSnapshot` at most 60 times a second. Pauses while `paused` is set. On stop, stores `w.hasEnded()` in `shared.result`. |
| Main | `ui_loop` | Handles window events, reads the latest snapshot, draws it. **Space** toggles pause; closing the window sets `stop`. Also sets `stop` when the snapshot has no living creatures. |

The UI never touches live `Creature` objects; it only reads frozen snapshots, which is why `snapshot.py` exists.

### 3.5 One simulation tick (`World.tick`)

| Step | What happens |
|---|---|
| 1 | `tick_count += 1` |
| 2 | Every creature flagged `remove` is appended to `deadCreaturesA`/`deadCreaturesB`, and a corpse is made at its position |
| 3 | Rotted/eaten corpses are dropped; dead creatures are dropped and newborns (`newCreaturesA/B`) are added to the living lists |
| 4 | Position lists are built for both groups, plants, water and corpses |
| 5 | Memories updated for each group: enemies, plants, water, corpses (not allies) |
| 6 | `tick()` every creature (hunger/thirst decay, death check, **network forward pass**), corpse (rot) and plant (regrow) |
| 7 | `act()` every creature (turn, move, one action) |

Note the order: a creature that dies during step 6 or 7 is only moved to the dead list at step 2 of the *next* tick.

### 3.6 The creature's "think → act" cycle

$$\underbrace{\text{state} + \text{memories}}_{\mathbf{x}\,\in\,\mathbb{R}^{58}} \;\xrightarrow{\;\text{NeuralNetwork.evaluate}\;}\; \underbrace{\mathbf{d}}_{\text{decision}\,\in\,\mathbb{R}^{8}} \;\xrightarrow{\;\text{act()}\;}\; \text{turn, move, one action}$$

### 3.7 Vision geometry

A creature at $(x, y)$ facing angle $\theta$ sees a point $(p_x, p_y)$ if it lies inside a cone in front of it. With $\Delta x = p_x - x$, $\Delta y = p_y - y$, rotate into the creature's frame:

$$f = \Delta x \cos\theta + \Delta y \sin\theta \qquad\text{(forward distance)}$$

$$\ell = -\Delta x \sin\theta + \Delta y \cos\theta \qquad\text{(sideways offset)}$$

The point is visible when

$$f > 0, \qquad |\ell| \le m\,f, \qquad d = \sqrt{\Delta x^2 + \Delta y^2} \le R,$$

where $m$ is `fieldOfView` (a slope, so the half-angle is $\alpha = \arctan m$) and $R$ is `viewDistance`. The same cone test (with $R$ replaced by `attackRange`) is used by `attack`.

View distance is now derived per creature as

$$R = \sqrt{70 + 70\,m},$$

which gives:

| Species | $m$ | Half-angle $\alpha = \arctan m$ | Full cone | $R$ |
|---|---|---|---|---|
| Herbivore | 0.2 | $\approx 11.3^\circ$ | $\approx 22.6^\circ$ | $\sqrt{84} \approx 9.17$ |
| Carnivore | 1 | $45^\circ$ | $90^\circ$ | $\sqrt{140} \approx 11.83$ |

So herbivores currently see a narrower wedge than carnivores, but only slightly less far.

---

## 4. Module reference

### 4.1 `src/main.py`: entry point, UI and generation loop

#### Class `SharedState`

| Member | Meaning |
|---|---|
| `_lock` | `threading.Lock` guarding `_snapshot` |
| `_snapshot` | Latest `WorldSnapshot` (or `None`) |
| `stop` | `threading.Event`; set = shut down |
| `paused` | `threading.Event`; set = paused |
| `result` | Filled by `sim_loop` with `w.hasEnded()`, i.e. `[brainsA, brainsB]` |
| `publish(snap)` | Store a new snapshot (locked) |
| `latest()` | Read the newest snapshot (locked) |

#### `sim_loop(w, shared, publish_hz=60.0)`

Publishes an initial snapshot, then until `stop` is set: sleeps 0.01 s while paused, otherwise sleeps 0.01 s, calls `w.tick()`, and publishes a snapshot if at least $1/60$ s has passed since the last one. After the loop: `shared.result = w.hasEnded()`.

#### Drawing constants

| Name | Value |
|---|---|
| `KIND_COLOURS` | default grey, creatureA red, creatureB blue, plant green, water dark blue, corpse brown |
| `SIDEBAR_W` | 690 px (widened for the Life column) |
| `DECISION_LABELS` | `["R", "L", "Move", "Atk", "Plant", "Corps", "Drink", "Mate"]` (the 8 outputs) |
| `STAT_LABELS` | `["HP", "Hun", "Thi", "Kids", "Life"]` (Life = lifespan in ticks) |
| `ACTION_START` | 3 (index of the first exclusive action in `decision`) |

#### `stat_colour(value, low=25)`

Returns red if a 0-100 stat is below `low`, otherwise light grey.

#### `draw_sidebar(screen, snap, font, x0, height)`

Draws a table on the right: one row per living creature with its tag, HP / hunger / thirst / kids / lifespan, and all 8 decision values (formatted `+0.00`). The action that `act()` will pick, $\arg\max_{k}\, d_{3+k}$, is highlighted in yellow. Rows that do not fit are summarised as `... +N more`.

#### `ui_loop(shared, window_size=(1300, 700))`

Opens a resizable window titled "World". Each frame: handle events (quit, Space = pause), fill the background, scale the $500 \times 500$ world to fit the area left of the sidebar (keeping aspect ratio and centring it), draw the world border, draw every object as a circle (radius scaled, minimum 2 px), draw the HUD line (`tick`, object count, creature count, fps, `[PAUSED]`), draw the sidebar, and stop when there are no creatures. Runs at 60 fps.

#### Generation loop (`if __name__ == "__main__":`)

Runs 30 worlds (`i = 0 .. 29`):

| `i` | World created |
|---|---|
| 0 | `World(0, None, None)` → 20 random creatures per species |
| 1 | `World(1, initialCount=50)` → 50 random creatures per species (the GA result of world 0 is not used) |
| $\ge 2$ | `World(i, brains1, brains2)` → one creature per brain from the previous world |

After each world, `brains1, brains2 = shared.result`. Since `getBest` is called with `initialCount` (default 20), generations from $i = 2$ onwards have 20 creatures per species, while world 1 has 50.

### 4.2 `src/neuralNetStuff/brain.py`: `NeuralNetwork`

#### `__init__(self, layer_sizes, genome=None)`

`layer_sizes` is a list $[n_0, n_1, \dots, n_L]$. The genome length is

$$P = \sum_{l=0}^{L-1} (n_l + 1)\, n_{l+1},$$

(weights $n_l n_{l+1}$ plus biases $n_{l+1}$ per layer). If `genome` is `None`, it is drawn uniformly, $\theta_j \sim \mathcal{U}(-1, 1)$; otherwise its length must equal $P$ or a `ValueError` is raised. The genome is stored as a float numpy array and unpacked.

#### `_unpack_genome(self)`

Walks through the flat genome layer by layer: the first $n_l n_{l+1}$ numbers become the weight matrix $W_l \in \mathbb{R}^{n_l \times n_{l+1}}$ (reshaped row-major), the next $n_{l+1}$ numbers become the bias vector $\mathbf{b}_l$. The matrices are numpy **views** into the genome, so they share memory with it.

#### `evaluate(self, inputs)`

Forward pass with row-vector convention:

$$\mathbf{h}_0 = \mathbf{x}, \qquad \mathbf{h}_{l+1} = \begin{cases} \tanh\!\big(\mathbf{h}_l W_l + \mathbf{b}_l\big) & l < L-1 \quad\text{(hidden layers)}\\[2pt] \mathbf{h}_l W_l + \mathbf{b}_l & l = L-1 \quad\text{(output layer, linear)} \end{cases}$$

Returns $\mathbf{h}_L$ as a numpy array.

#### The network used by creatures

`SHAPE = [58, 58, 40, 25, 10, 8]` (defined in `creatures.py`), so there are 4 hidden `tanh` layers and a linear output of 8.

| Layer | $n_l \to n_{l+1}$ | Parameters $(n_l+1)\,n_{l+1}$ |
|---|---|---|
| 1 | $58 \to 58$ | 3422 |
| 2 | $58 \to 40$ | 2360 |
| 3 | $40 \to 25$ | 1025 |
| 4 | $25 \to 10$ | 260 |
| 5 | $10 \to 8$ | 88 |
| **Total** | | **7155** (verified) |

### 4.3 `src/neuralNetStuff/geneticAlg.py`: evolution

The file header lists four strategies being considered:

| | Strategy |
|---|---|
| A | Many worlds per simulation; all creatures of a type in a world are identical; score the whole world; mutate the single creature design. |
| B | One world per simulation; every creature unique; score each creature on lifespan and children; next world = children of the best. |
| C | Hybrid: start with B, later add world extinction time to fitness and run many worlds in parallel. |
| D | Hybrid: B for actions, plus mutating the "final" (permanent) stats of the best creatures. |

**Strategy B is implemented.**

#### `fitness(creature)`

$$F = \frac{\text{lifespan}}{50000} + 5\cdot\text{numChildren} + \text{hunger} + \text{health} + \text{thirst}$$

Values are read when the GA runs (after death), so health is 0 and hunger/thirst are whatever was left at death.

#### `makeChildBrain(brain1, brain2, blend=0.5, mutation_rate=0.02, mutation_strength=0.1)`

Builds a child genome of length $n$ from parent genomes $\boldsymbol{\theta}^{(1)}, \boldsymbol{\theta}^{(2)}$:

1. **Uniform crossover:** for each gene, pick a parent with probability $\tfrac12$:
$$\beta_j = \begin{cases}\theta^{(1)}_j & u_j < 0.5\\ \theta^{(2)}_j & \text{otherwise}\end{cases}, \qquad u_j \sim \mathcal{U}(0,1)$$
2. **Blend noise** proportional to how much the parents disagree:
$$\nu_j = \big|\theta^{(1)}_j - \theta^{(2)}_j\big| \cdot \text{blend} \cdot v_j, \qquad v_j \sim \mathcal{U}(-1, 1)$$
3. **Sparse Gaussian mutation:** each gene mutates with probability `mutation_rate`:
$$\mu_j = \mathbb{1}[r_j < \text{rate}] \cdot z_j, \qquad z_j \sim \mathcal{N}(0, \sigma^2),\ \sigma = \text{mutation\_strength}$$

Child genome: $\theta^{\text{child}}_j = \beta_j + \nu_j + \mu_j$, returned as a new `NeuralNetwork` with the parents' layer sizes.

#### `getBest(creatures, num_children=None, survivor_frac=0.3, wildcard_frac=0.15, elite_frac=0.05, immigrant_frac=0.05, tournament_size=3, mutation_rate=0.02, mutation_strength=0.1)`

Returns a list of `num_children` brains (default: as many as there are creatures; `[]` if `creatures` is empty). With $N$ creatures and $C$ children:

| Step | What happens | Count |
|---|---|---|
| 1. Score | Sort `(fitness, brain)` pairs, best first | $N$ |
| 2. Parent pool | Top survivors + random "wildcards" from the rest | $\max(2, \lfloor 0.3N \rfloor)$ + $\min(\text{rest}, \lfloor 0.15N \rfloor)$ |
| 3. Elites | Best brains copied unchanged | $\min(C, \max(1, \lfloor 0.05C \rfloor))$ |
| 4. Immigrants | Brand-new random brains | $\lfloor 0.05C \rfloor$ (capped by space left) |
| 5. Selection | Tournament: sample 3 from the pool, keep the fittest | per parent |
| 6. Breeding | Two parents (retry up to 5 times if both are the same brain), per-child mutation rate $\times\,\mathcal{U}(0.5, 3)$ and strength $\times\,\mathcal{U}(0.5, 2)$, then `makeChildBrain` | until $C$ brains |

Example with $N = 36$ dead creatures and $C = 20$: pool $= 10 + 5 = 15$, elites $= 1$, immigrants $= 1$, bred $= 18$.

### 4.4 `src/worldObjects/world.py`: the world

#### Module-level helpers (unchanged)

* **`merge_memory(new, old, max_n, now, max_age)`**: stamps each new sighting as `(x, y, now)`, keeps the `old` entries that were not re-sighted **and** are younger than the time limit, and truncates to `max_n`. An entry seen at tick $t_s$ is kept while
$$\text{now} - t_s < \text{max\_age},$$
so it is forgotten exactly `max_age` ticks after the creature last saw it. Re-seeing a thing refreshes its stamp. Current sightings outrank old ones.
* **`cone_candidates(creatures, pts, mem_counts, chunk=512)`**: vectorised cone test (Section 3.7) for a batch of creatures against a set of points; returns, per creature, visible points nearest-first, capped at `mem_counts[i]`. Processes 512 creatures per numpy batch.
* **`update_memory(group, pts, mem_attr, last_attr, time_attr, now)`**: runs `cone_candidates` for a group and writes `merge_memory(..., now, getattr(c, time_attr))` back to each creature's `last_attr` list with `setattr`. `World.tick` passes `now = self.tick_count`.

#### Class `World`

**Class attributes:** `width = 500`, `height = 500` (now used for spawning and drawing).

##### `__init__(self, id, brainsA=None, brainsB=None, initialCount=20)`

| Attribute | Value |
|---|---|
| `id`, `tick_count = 0`, `initialCount` | as given |
| `creaturesA` | one `Creature(i, 0, self, b)` per brain in `brainsA`, or `initialCount` random ones |
| `creaturesB` | the same with type 1 and `brainsB` |
| `creaturesACount`, `creaturesBCount` | starting count; incremented to give newborns ids |
| `plants` | 81 plants on a $9 \times 9$ grid: $x, y \in \{50, 100, \dots, 450\}$ |
| `waterSources` | 49 sources on a $7 \times 7$ grid: $x, y \in \{62.5, 125, \dots, 437.5\}$ |
| `corpses`, `newCreaturesA/B`, `deadCreaturesA/B` | empty lists |

All lists are now instance attributes. Creatures start at uniformly random positions in $[0, 500]^2$, facing $\theta = 0$.

##### `snapshot(self) -> WorldSnapshot`

Builds `ObjectView`s for water (radius 5), plants (2), corpses (2), and both creature groups (3), plus a `CreatureView` per living creature (id, kind, health, hunger, thirst, children, decision as plain floats). Returns a frozen `WorldSnapshot`. Should only be called from the simulation thread between ticks.

##### `makeCorpse(self, posX, posY, creature)`

Appends `Corpse(0, posX, posY, creature.health)`.

##### `createChild(self, type, brain1, brain2, posX, posY, health=50, food=10, water=10)`

Makes a child brain with `makeChildBrain(brain1, brain2)` (default GA settings), increments the relevant count, and appends a new `Creature` at `(posX, posY)` to `newCreaturesA` (type 0) or `newCreaturesB` (otherwise). Newborns join the living list at the start of the next tick.

##### `tick(self)`

See Section 3.5.

##### `hasEnded(self)`

Runs the GA: `brainsA = getBest(deadCreaturesA, initialCount)`, `brainsB = getBest(deadCreaturesB, initialCount)`, and returns `[brainsA, brainsB]`. Only dead creatures are scored (creatures still alive when the window is closed are not). A commented-out block prints dead creatures' lifespans. A TODO says to save the brains.

### 4.5 `src/worldObjects/creatures.py`: the creature

#### Module constants

| Name | Value | Meaning |
|---|---|---|
| `memorySize` | 25 | Intended total memory budget (unused) |
| `NUM_ENEMIES_MEM`, `NUM_ALLIES_MEM`, `NUM_PLANT_MEM`, `NUM_WATER_MEM`, `NUM_CORPSE_MEM` | 5 each | Memory slots used to size the input |
| `ENEMY_MEM_TIME`, `ALLY_MEM_TIME` | 30 | Ticks before a moving creature's position is forgotten |
| `PLANT_MEM_TIME` | 300 | Ticks before a plant memory is forgotten |
| `WATER_MEM_TIME` | 600 | Ticks before a water memory is forgotten |
| `CORPSE_MEM_TIME` | 100 | Ticks before a corpse memory is forgotten |
| `BASE_INPUTS` | 8 | Own-state inputs |
| `MEMORY_SLOTS` | 25 | Sum of the five above |
| `INPUT_SIZE` | $8 + 25 \times 2 = 58$ | Network input length |
| `SHAPE` | `[58, 58, 40, 25, 10, 8]` | Network layer sizes |

#### `__init__(self, id, type, world, brain=None, posX=-1, posY=-1, health=50, water=25, food=25)`

| Attribute | Value |
|---|---|
| `world`, `id`, `type` | as given (0 herbivore, 1 carnivore) |
| `remove`, `lifespan`, `numChildren` | `False`, 0, 0 |
| `maxSpeed` | 1 |
| `enemyMemoryTime`, `allyMemoryTime`, `plantMemoryTime`, `waterMemoryTime`, `corpseMemoryTime` | from the `..._MEM_TIME` constants (per creature, so they could later be evolved) |
| `numberOf...Memory` (allies, enemies, plants, water, corpses) | 5 each |
| `health`, `hunger`, `thirst` | from `health`, `food`, `water` (as floats). Hunger and thirst count *down*: high is good. |
| `posX`, `posY` | given, or random in $[0, 500]$ if left at $-1$ |
| `direction` | 0 |
| `last...Seen` (five lists) | `[]`; entries are `(x, y, tickSeen)` |
| `brain` | given `NeuralNetwork`, or a new random one with `SHAPE` |
| `decision` | first forward pass on the starting input vector |

Per-species stats:

| `type` | Species | `foodFromPlant` | `foodFromMeat` | `attackRange` | `attackPower` | `fieldOfView` $m$ | `viewDistance` |
|---|---|---|---|---|---|---|---|
| 0 | herbivore | 1 | 0.1 | 5 | 10 | 0.2 | $\sqrt{84}\approx 9.17$ |
| 1 | carnivore | 0.6 | 10 | 10 | 20 | 1 | $\sqrt{140}\approx 11.83$ |

#### `_flatten_memory(self, points, max_len)`

Turns `[(x, y, tickSeen), ...]` (only $x$ and $y$ are used) into exactly `2 * max_len` floats $[x_1, y_1, x_2, y_2, \dots]$, truncating or padding with zeros.

#### `inputVector(self)`: the 58 network inputs

| Indices | Content |
|---|---|
| 0-7 | `lifespan`, `numChildren`, `health`, `hunger`, `thirst`, `posX`, `posY`, `direction` |
| 8-17 | 5 remembered enemy positions $(x, y)$ |
| 18-27 | 5 remembered ally positions (always zeros for now) |
| 28-37 | 5 remembered plant positions |
| 38-47 | 5 remembered water positions |
| 48-57 | 5 remembered corpse positions |

Returned as a `float32` numpy array. Positions are absolute world coordinates (not relative to the creature) and inputs are not normalised.

#### `__str__(self)`

Still returns `""` (TODO).

#### `isDead(self)`

If health, hunger or thirst is $\le 0$: set `remove = True`, `health = 0`, return `True`. Otherwise `False`.

#### `tick(self)`: live and think

1. If alive: $h \leftarrow h - 0.01$, $w \leftarrow w - 0.01$ (hunger and thirst).
2. If still alive: `lifespan += 1` and `decision = brain.evaluate(inputVector())`.

With the default starting hunger and thirst of 25, a creature that never eats or drinks dies after

$$\frac{25}{0.01} = 2500 \text{ ticks},$$

which matches the observed extinction at tick 2501.

#### `act(self)`: decision → actions

With $\mathbf{d} = (d_0, \dots, d_7)$ = `decision`, if the creature is alive:

1. **Turn** (optional): if $d_0 > 0$ or $d_1 > 0$, then `turnL()` if $d_1 > d_0$, else `turnR()`.
2. **Move** (optional): if $d_2 > 0$, `move()`.
3. **Exactly one action:** $k = \arg\max_{k \in \{0,\dots,4\}} d_{3+k}$, then run

| $k$ | output index | action |
|---|---|---|
| 0 | 3 | `attack()` |
| 1 | 4 | `eatPlant()` |
| 2 | 5 | `eatCorpse()` |
| 3 | 6 | `drinkWater()` |
| 4 | 7 | `procreate()` |

#### Movement

* `move()`: $x \leftarrow x + v_{\max}\cos\theta$, $y \leftarrow y + v_{\max}\sin\theta$ with $v_{\max} = 1$. No bounds check.
* `turnL()`: $\theta \leftarrow \theta - \pi/9$; `turnR()`: $\theta \leftarrow \theta + \pi/9$ (20°).
* `damage(amount)`: `health -= amount`, then `isDead()`.

#### `attack(self)`

Picks a target list from `self.world` by `type` (type 0 → `creaturesA`, type 1 → `creaturesB`), then damages, by `attackPower`, **every** creature in that list that is inside the attack cone (same $m$ as vision, range `attackRange`).

#### `eatPlant(self)`

Takes the first plant in `world.plants` within `attackRange` (Euclidean distance, no cone), adds `plant.tryEat(foodFromPlant)` to hunger, caps at 100.

#### `eatCorpse(self)`

Same, over `world.corpses`, using `corpse.tryEat(foodFromMeat)`. The corpse is now depleted.

#### `drinkWater(self)`

Looks through the creature's **memory** `lastWaterSeen` (water never moves, so memory is accurate) for a source within `attackRange`; if found, thirst $+2$, capped at 100.

#### `procreate(self)` ("advanced" mode, active)

Allies are `world.creaturesA` for type 0, `world.creaturesB` for type 1. For the first ally within `attackRange` that is alive:

1. Partner takes `attackPower` damage; self takes `health / 2` damage.
2. Both lose 10 hunger and 5 thirst.
3. If both are still alive: `world.createChild(type, self.brain, p.brain, posX, posY, attackPower + health/2, 20, 10)` and `numChildren += 1`.

The old "easy mode" (free child, no costs) is kept as a commented string.

### 4.6 `src/worldObjects/plants.py`: plants, water and corpses

| Class | Fields | Methods |
|---|---|---|
| `Corpse(id, posX, posY, health)` | `id`, `posX`, `posY`, `health` (now passed in), `remove` | `tick()`: health $-1$, remove at $\le 0$. `tryEat(d)`: gives up to `d` health; when exhausted, marks `remove`. |
| `Water(posX, posY)` | `posX`, `posY` | none (static, infinite) |
| `Plant(posX, posY)` | `posX`, `posY`, `health = 50` | `tick()`: health $+0.5$ (no cap). `tryEat(d)`: gives up to `d`; never removed. |

### 4.7 `src/worldObjects/snapshot.py`: UI data (new)

Three `@dataclass(frozen=True)` classes, so the UI thread cannot change simulation data:

| Class | Fields |
|---|---|
| `ObjectView` | `kind` (`"creatureA"`, `"creatureB"`, `"plant"`, `"water"`, `"corpse"`), `x`, `y`, `radius = 1.0` |
| `CreatureView` | `id`, `kind`, `health`, `hunger`, `thirst`, `numChildren`, `lifespan`, `decision` (tuple of floats) |
| `WorldSnapshot` | `tick`, `width`, `height`, `objects` (tuple of `ObjectView`), `creatures` (tuple of `CreatureView`) |

---

## 5. Default parameters at a glance

| Quantity | Value | Where |
|---|---|---|
| World size | $500 \times 500$ | `World` |
| Creatures per species | 20 (world 1: 50) | `World.__init__`, `main.py` |
| Generations | 30 | `main.py` |
| Plants / water sources | 81 ($9\times9$ grid) / 49 ($7\times7$ grid) | `World.__init__` |
| Start position / direction | uniform random in $[0,500]^2$ / $\theta = 0$ | `Creature.__init__` |
| Starting health / hunger / thirst | 50 / 25 / 25 | `Creature.__init__` |
| Newborn health / hunger / thirst arguments | $\text{attackPower} + \text{health}/2$ / 20 / 10 | `procreate` |
| Hunger and thirst decay | 0.01 per tick | `Creature.tick` |
| Max speed | 1 per tick | `Creature` |
| Turn step | $\pi/9$ (20°) | `turnL`, `turnR` |
| Field of view $m$ | herbivore 0.2, carnivore 1 | `Creature.__init__` |
| View distance $R = \sqrt{70+70m}$ | herbivore 9.17, carnivore 11.83 | `Creature.__init__` |
| Attack range / power | herbivore 5 / 10, carnivore 10 / 20 | `Creature.__init__` |
| Food per bite | herbivore 1 plant / 0.1 meat; carnivore 0.6 plant / 10 meat | `Creature.__init__` |
| Water per drink | 2 | `drinkWater` |
| Procreation cost | 10 hunger, 5 thirst each; self loses half its health; partner takes `attackPower` | `procreate` |
| Memory slots | 5 per kind (25 total) | `Creature` |
| Memory time limits | enemies/allies 30, plants 300, water 600, corpses 100 ticks | `creatures.py` |
| Network | `[58, 58, 40, 25, 10, 8]`, `tanh` hidden, linear out, 7155 parameters | `creatures.py`, `brain.py` |
| Initial weights | $\mathcal{U}(-1, 1)$ | `NeuralNetwork` |
| GA fractions | survivors 0.30, wildcards 0.15, elites 0.05, immigrants 0.05 | `getBest` |
| Tournament size | 3 | `getBest` |
| Mutation | rate 0.02, $\sigma = 0.1$ (scaled per child by $\mathcal{U}(0.5,3)$ and $\mathcal{U}(0.5,2)$); blend 0.5 | `getBest`, `makeChildBrain` |
| Plant regrowth / corpse rot | $+0.5$ / $-1$ per tick | `plants.py` |
| Sim speed | 0.01 s sleep per tick (≤ ~100 ticks/s) | `sim_loop` |
| Snapshot rate / UI fps | 60 / 60 | `sim_loop`, `ui_loop` |
| Vision chunk size | 512 creatures | `cone_candidates` |

---

## 6. Controls

| Input | Effect |
|---|---|
| **Space** | Pause / resume the simulation |
| Close window | Ends the current world; the GA runs and the next generation's window opens |
| (automatic) | A world also ends when no creatures are left alive |

Note that closing the window only ends the *current* world; to stop all 30 generations you have to close it repeatedly or interrupt the program (Ctrl+C in the terminal).

---

## 7. TODO list

### 7.1 TODO comments in the code

| # | File and line | TODO |
|---|---|---|
| 1 | `world.py:11` | Load a template of creatures, plants and water so each generation has the same start. |
| 2 | `world.py:68` | Restrict creature movement to the world bounds. |
| 3 | `world.py:181` | Save dead stuff to mutate later (now partly done via `deadCreaturesA/B`). |
| 4 | `world.py:185` | `hasEnded`: save the brains → genetic stuff → make new brains (the GA part is done; saving is not). |
| 5 | `creatures.py:52` | Enforce `memorySize` so the memory capacities sum to less than it. |
| 6 | `creatures.py:128` | Complete `__str__`. |
| 7 | `creatures.py:5` (header) | On death, record the lifetime and weights (recorded in memory, not on disk). |

### 7.2 Unfinished features (no TODO written)

| # | Item |
|---|---|
| 8 | Saving / loading brains in `weights/` (folder is still empty). |
| 9 | Ally memory is never updated, so input slots 18-27 are always zero. |
| 10 | Strategies A, C and D from the `geneticAlg.py` header are not implemented. |
| 11 | No per-generation statistics (best/mean fitness, lifespans) are printed or plotted. |
| 12 | Permanent stats (speed, field of view, attack, memory sizes) are fixed per species and not evolved (strategy D). |
| 13 | Creatures always start facing $\theta = 0$. |

---

## 8. Suggested next steps (features, not bug fixes)

1. **Save brains** after each generation: `np.save("weights/genA_<i>.npy", np.stack([b.genome for b in brainsA]))`, and a matching loader, so runs can be resumed.
2. **Log generation statistics** (max / mean fitness, mean lifespan, extinction tick) to see whether evolution is actually improving anything.
3. **Normalise the inputs** (for example divide positions by 500, stats by 100) and consider giving memories **relative** positions $(\Delta x, \Delta y)$ or (distance, angle). With $\tanh$ hidden layers, raw inputs in the hundreds saturate the first layer.
4. **Fill ally memory** (one more `update_memory` call with each group's own positions).
5. **Headless fast mode**: a flag to run generations without pygame and without the 0.01 s sleep, for faster training.
6. Later: **evolve the permanent stats** (strategy D) by appending them to the genome.

---

## 9. Bugs noted but deferred

As requested, these are only listed, not analysed or fixed.

* `attack()` targets the creature's **own** group (type 0 attacks `creaturesA`).
* `createChild` gives carnivore children `type = 0` (verified: 16 of the 36 dead in group B had type 0).
* `createChild` passes `health, food, water` but `Creature.__init__` expects `health, water, food` (the two are swapped).
* Corpses get `creature.health`, which `isDead()` has just set to 0, so corpses disappear on their first tick.
* World 1 ignores world 0's GA result and starts with 50 fresh random creatures.
* `procreate` can pick the creature itself as partner (distance 0).
* `hasEnded()` only scores dead creatures; survivors are lost if the window is closed early.
* Fitness is measured after death, so the `health` term is always 0.
* Movement is unbounded (creatures can leave the $500 \times 500$ area).

---

## 10. Quick-start

```bash
cd D:\Coding_\evolution\src
pip install numpy pygame
python main.py          # opens the window; Space pauses; close it to go to the next generation
```

Headless (no window), from inside `src`:

```python
from worldObjects import world

w = world.World(0)
while w.creaturesA or w.creaturesB:
    w.tick()
print(w.tick_count)                       # extinction tick (about 2500)
brainsA, brainsB = w.hasEnded()           # GA output: 20 new brains per species
w2 = world.World(1, brainsA, brainsB)     # next generation
```
