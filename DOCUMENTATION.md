# Evolution Simulation: Code Documentation

*Snapshot of the project at `D:\Coding_\evolution`, re-scanned against the code on 6 November 2026 (fourth revision). Every constant in this document was read from the source or checked by running the code. Section 9 lists known problems briefly; they are not analysed in depth.*

---

## 1. Overview

This is an **artificial-life / neuroevolution simulation**. Two species of creatures (herbivores and carnivores) live in a $500 \times 500$ world containing plants, water sources and corpses.

1. Every creature has a **feed-forward neural network "brain"** that reads its own state plus what it remembers seeing, and outputs an 8-number **decision** vector (turn, move, and one of attack / eat plant / eat corpse / drink / procreate).
2. When creatures die they are kept in a "dead" list together with their brain, lifespan and number of children.
3. When a world ends, the dead creatures' brains are **saved to disk** (`weights/<generation>/`), and a **genetic algorithm** scores them, selects parents and breeds a new set of brains, which seed the next world (the next generation).
4. A **pygame window** shows the world live, with a sidebar listing every creature's stats and network outputs.
5. A second entry point, `continueMain.py`, **reloads a saved generation** and carries on evolving from it.

This is the same idea as in `neuroevolution_car_summary.md` (network = driver, genetic algorithm = trainer, genome = all weights and biases flattened into one vector $\boldsymbol{\theta}$), applied to survival instead of lane-keeping.

### 1.1 Current status

| Area | State |
|---|---|
| World data model (creatures, plants, water, corpses) | Working |
| Vision and memory (enemies, plants, water, corpses) | Working |
| Neural network (`neuralNetStuff/brain.py`) | Implemented |
| Sensory input vector (`Creature.inputVector`) | Implemented (**63** inputs, normalised, memories relative to the creature) |
| Think step (network runs every tick) | Implemented |
| `act()` (decision → actions) | Implemented |
| Reproduction (`procreate` → `World.createChild`) | Implemented |
| Death → corpse, dead creatures recorded | Implemented |
| Genetic algorithm (`neuralNetStuff/geneticAlg.py`) | Implemented (strategy B: one world, unique creatures) |
| Generation loop | Implemented in `main.py` (100 worlds) |
| Rendering | Implemented with pygame, on its own thread |
| **Saving brains to `weights/`** | **Implemented** (`World.saveBrains`, 38 generations saved so far) |
| **Resuming from saved brains** | **Implemented** (`continueMain.py`) |
| Ally memory | Not updated (memory slots exist, stay empty) |
| World bounds | Not enforced (but the position inputs are clipped to $[0,1]$) |

Verified by running headless (without the UI, random seed 0, weights redirected to a temporary folder): `World(0)` builds 20 herbivores and 20 carnivores, the network has 63 inputs and a genome of 13030 numbers, creatures move, fight, eat and breed (2 herbivore and 7 carnivore children), and the world is empty at tick 8028 (about 6 seconds without the UI sleep). `hasEnded()` then saves the brains and returns two lists of 20 new brains.

### 1.2 What changed since the previous version of this document

| Change | Where |
|---|---|
| Brains are saved after every world: `saveBrains()` writes `weights/<id>/genomesA.npz` and `genomesB.npz`; `hasEnded()` calls it first | `world.py:209-243` |
| New `WEIGHTS_DIR` constant (`evolution/weights`, found relative to `world.py`) | `world.py:14` |
| New file `continueMain.py`: loads the newest (or a chosen) saved generation, breeds from it and keeps going | `src/continueMain.py` |
| `main.py`: simulation sleep removed (runs as fast as the CPU allows); 100 generations; world 1 now has 55 creatures per species | `main.py:39, 181-201` |
| Network is deeper: `SHAPE = [63, 63, 50, 45, 40, 25, 15, 10, 8]` (7 weight layers, 13030 parameters) | `creatures.py:56` |
| Memory sizes changed to 10 enemies, 10 allies, 3 plants, 3 water, 2 corpses (28 slots); memory times changed | `creatures.py:20-31` |
| Inputs are now **normalised** to $[0,1]$; **lifespan is no longer an input**; remembered positions are now **relative to the creature**, in its own frame, and distance-weighted | `creatures.py:33-161` |
| Hunger/thirst decay is now 0.012 per tick; health regenerates and drains; creatures die of old age at 15000 ticks | `creatures.py:174-189` |
| Carnivores are faster (speed 2) and see further; view distance is now a fixed number (100 / 150) | `creatures.py:118-133` |
| Turning step is now 10° (not 20°), and direction is wrapped into $[0, 2\pi)$ | `creatures.py:224-230` |
| `attack()` now targets the *other* species; `procreate()` skips the creature itself; `createChild` uses keyword arguments and gives carnivore children the right type | `creatures.py:232-316`, `world.py:150-157` |
| Fitness changed to $\text{lifespan}/100 + 10\cdot\text{children} + \text{hunger} + \text{thirst}$; blend noise removed from crossover; new GA defaults | `geneticAlg.py` |
| 49 plants ($7\times7$) and 25 water sources ($5\times5$), plants regrow $+2$ per tick, corpses rot $-0.1$ per tick, corpse food $=$ hunger $+$ thirst at death | `world.py:112-115`, `plants.py` |
| `getBest` prints the best creature of each species every generation | `geneticAlg.py:84` |

---

## 2. Project structure

```
evolution/
├── DOCUMENTATION.md                  this file
├── neuroevolution_car_summary.md     theory notes (not code)
├── weights/                          saved brains: weights/<generation>/genomesA.npz, genomesB.npz
│   ├── 0/ ... 37/                    38 generations saved so far
└── src/
    ├── main.py                       entry point: UI thread + simulation thread + generation loop (fresh start)
    ├── continueMain.py               entry point: resume from saved weights  (NEW)
    ├── neuralNetStuff/
    │   ├── brain.py                  NeuralNetwork class
    │   └── geneticAlg.py             fitness, crossover/mutation, selection
    └── worldObjects/
        ├── world.py                  World class, vision/memory helpers, saveBrains
        ├── creatures.py              Creature class, network shape and input constants
        ├── plants.py                 Corpse, Water, Plant classes
        └── snapshot.py               ObjectView, CreatureView, WorldSnapshot
```

`__pycache__` folders show the code is being run with **Python 3.14**.

**Imports.** All imports are relative to `src/` (for example `from worldObjects import world`, `from neuralNetStuff.brain import NeuralNetwork`), so both entry points must be started from inside `src`:

```bash
cd D:\Coding_\evolution\src
python main.py                    # fresh run
python continueMain.py            # resume from the newest saved generation
```

**Dependencies:** `numpy`, `pygame`, and the standard library (`math`, `random`, `threading`, `time`, `dataclasses`, `typing`, `pathlib`, `argparse`, `types`).

**Import graph:**

```
main.py ──► worldObjects.world ──► worldObjects.creatures ──► neuralNetStuff.brain
   ▲             │                         ▲
   │             ├──► worldObjects.plants  │
continueMain.py  ├──► worldObjects.snapshot│
 (also imports   └──► neuralNetStuff.geneticAlg ──┘ (imports Creature and NeuralNetwork)
  SHAPE, WEIGHTS_DIR,
  NeuralNetwork, getBest)
```

`creatures.py` imports `World` only under `TYPE_CHECKING`, to avoid a circular import.

**Version control.** The folder is a git repository on branch `main` with a remote `origin`.

---

## 3. Architecture and data flow

### 3.1 Objects

| Object | File | Role |
|---|---|---|
| `World` | `world.py` | Owns every entity, runs the clock (`tick`), makes snapshots, saves brains and runs the GA at the end |
| `Creature` | `creatures.py` | An animal: stats, senses, memory, brain, actions |
| `NeuralNetwork` | `brain.py` | The brain: flat genome → layers → forward pass |
| `Plant` | `plants.py` | Static food that regrows |
| `Water` | `plants.py` | Static, infinite drinking spot |
| `Corpse` | `plants.py` | Meat left by a dead creature; rots away |
| `ObjectView`, `CreatureView`, `WorldSnapshot` | `snapshot.py` | Read-only copies of the world for the UI |
| `SharedState` | `main.py` | Thread-safe mailbox between simulation and UI |
| `ContinueMain` | `continueMain.py` | Loads saved genomes and runs the generation loop from them |

### 3.2 Two species

| | Group | `type` | Colour in UI | Sidebar tag |
|---|---|---|---|---|
| Herbivores | `creaturesA` | 0 | red | `A<id>` |
| Carnivores | `creaturesB` | 1 | blue | `B<id>` |

Each group sees the other as enemies (for memory), and `attack()` now hits the *other* group.

### 3.3 Program flow (`main.py`)

```
brains1, brains2 = None, None
for generation i = 0 .. 99:
    build World i         (i == 1: 55 fresh random creatures per species, ignoring the GA result of world 0;
                           otherwise: one creature per brain from the previous world, or 20 random ones if None)
    start sim_loop(world) on a daemon thread
    ui_loop() on the main thread  ── draws snapshots at 60 fps
        stops when the window is closed or no creatures are left
    stop and join the sim thread (timeout 2 s)
    sim_loop's last line runs  world.hasEnded()  ->  saves weights/<i>/  and returns the GA's new brains
    brains1, brains2 = shared.result
```

`continueMain.py` runs the same loop, but its first brains come from disk (Section 4.2).

### 3.4 Threads

| Thread | Function | Does |
|---|---|---|
| Worker (daemon) | `sim_loop` | `w.tick()` in a loop with **no sleep** (the `time.sleep(0.01)` line is commented out), so the simulation runs as fast as the CPU allows. Publishes a `WorldSnapshot` at most 60 times a second. Pauses while `paused` is set. On stop, stores `w.hasEnded()` in `shared.result`. |
| Main | `ui_loop` | Handles window events, reads the latest snapshot, draws it. **Space** toggles pause; closing the window sets `stop`. Also sets `stop` when the snapshot has no living creatures. |

The UI never touches live `Creature` objects; it only reads frozen snapshots, which is why `snapshot.py` exists.

### 3.5 One simulation tick (`World.tick`)

| Step | What happens |
|---|---|
| 1 | `tick_count += 1` |
| 2 | Every creature flagged `remove` is appended to `deadCreaturesA`/`deadCreaturesB`, and a corpse is made at its position |
| 3 | Rotted/eaten corpses are dropped; dead creatures are dropped and newborns (`newCreaturesA/B`) are added to the living lists |
| 4 | Position lists are built for both groups, plants, water and corpses |
| 5 | Memories updated for each group: enemies, plants, water, corpses (**not allies**) |
| 6 | `tick()` every creature (hunger/thirst decay, health, death check, **network forward pass**), corpse (rot) and plant (regrow) |
| 7 | `act()` every creature (turn, move, one action) |

A creature that dies during step 6 or 7 is only moved to the dead list at step 2 of the *next* tick.

### 3.6 The creature's "think → act" cycle

$$\underbrace{\text{state} + \text{memories}}_{\mathbf{x}\,\in\,[0,1]^{63}\text{ (memories in }[-1.05,1.05])} \;\xrightarrow{\;\text{NeuralNetwork.evaluate}\;}\; \underbrace{\mathbf{d}}_{\text{decision}\,\in\,\mathbb{R}^{8}} \;\xrightarrow{\;\text{act()}\;}\; \text{turn, move, one action}$$

### 3.7 Vision geometry

A creature at $(x, y)$ facing angle $\theta$ sees a point $(p_x, p_y)$ if it lies inside a cone in front of it. With $\Delta x = p_x - x$, $\Delta y = p_y - y$, rotate into the creature's frame:

$$f = \Delta x \cos\theta + \Delta y \sin\theta \qquad\text{(forward distance)}$$

$$\ell = -\Delta x \sin\theta + \Delta y \cos\theta \qquad\text{(sideways offset)}$$

The point is visible when

$$f > 0, \qquad |\ell| \le m\,f, \qquad d = \sqrt{\Delta x^2 + \Delta y^2} \le R,$$

where $m$ is `fieldOfView` (a slope, so the half-angle is $\alpha = \arctan m$) and $R$ is `viewDistance`. The same cone test (with $R$ replaced by `attackRange`) is used by `attack`.

`viewDistance` is now a fixed number per species (the old formula $\sqrt{70+70m}$ is left as a comment):

| Species | $m$ | Half-angle $\alpha = \arctan m$ | Full cone | $R$ |
|---|---|---|---|---|
| Herbivore | 0.2 | $\approx 11.3^\circ$ | $\approx 22.6^\circ$ | 100 |
| Carnivore | 1 | $45^\circ$ | $90^\circ$ | 150 |

Creatures can now see a long way (a fifth to a third of the world), but through a narrow wedge, especially the herbivores.

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

Publishes an initial snapshot, then until `stop` is set: sleeps 0.01 s while paused, otherwise calls `w.tick()` straight away and publishes a snapshot if at least $1/60$ s has passed since the last one. After the loop: `shared.result = w.hasEnded()`.

#### Drawing constants

| Name | Value |
|---|---|
| `KIND_COLOURS` | default grey, creatureA red, creatureB blue, plant green, water dark blue, corpse brown |
| `SIDEBAR_W` | 690 px |
| `DECISION_LABELS` | `["R", "L", "Move", "Atk", "Plant", "Corps", "Drink", "Mate"]` (the 8 outputs) |
| `STAT_LABELS` | `["HP", "Hun", "Thi", "Kids", "Life"]` (Life = lifespan in ticks) |
| `ACTION_START` | 3 (index of the first exclusive action in `decision`) |

#### `stat_colour(value, low=25)`

Returns red if a stat is below `low`, otherwise light grey.

#### `draw_sidebar(screen, snap, font, x0, height)`

Draws a table on the right: one row per living creature with its tag, HP / hunger / thirst / kids / lifespan, and all 8 decision values (formatted `+0.00`). The action that `act()` will pick, $\arg\max_{k}\, d_{3+k}$, is highlighted in yellow. Rows that do not fit are summarised as `... +N more`.

#### `ui_loop(shared, window_size=(1300, 700))`

Opens a resizable window titled "World". Each frame: handle events (quit, Space = pause), fill the background, scale the $500 \times 500$ world to fit the area left of the sidebar (keeping aspect ratio and centring it), draw the world border, draw every object as a circle (radius scaled, minimum 2 px), draw the HUD line (`tick`, object count, creature count, fps, `[PAUSED]`), draw the sidebar, and stop when there are no creatures. Runs at 60 fps.

#### Generation loop (`if __name__ == "__main__":`)

Runs 100 worlds (`i = 0 .. 99`), printing `world : i` before each:

| `i` | World created |
|---|---|
| 0 | `World(0, None, None)` → 20 random creatures per species |
| 1 | `World(1, initialCount=55)` → 55 fresh random creatures per species (the GA result of world 0 is discarded) |
| $\ge 2$ | `World(i, brains1, brains2)` → one creature per brain from the previous world |

The `initialCount` used by the GA (`hasEnded`) is the world's own `initialCount`: 55 for world 1 (so world 2 starts with 55 creatures per species), 20 for every other world (so worlds 3, 4, ... start with 20).

> **Warning.** Running `main.py` again starts at world 0 and `saveBrains()` uses `np.savez`, so it **overwrites** `weights/0`, `weights/1`, ... as it goes. Use `continueMain.py` to keep the saved generations.

### 4.2 `src/continueMain.py`: resume from saved weights

Same UI and simulation thread as `main.py` (it imports `SharedState`, `sim_loop`, `ui_loop`). The header comment lists the intended command-line options (`--gens`, `--start`), but **`argparse` is imported and never used**: the `__main__` block just calls `ContinueMain(initial_count=60).run(100)`.

#### Class `ContinueMain(weights_dir=WEIGHTS_DIR, start_gen=None, initial_count=20)`

| Method | What it does |
|---|---|
| `savedGenerations()` | Sorted list of all numeric folders in `weights/` that contain a `genomes*.npz` file |
| `latestGeneration()` | Highest saved generation number, or `None` |
| `loadSpecies(species)` | Loads `weights/<gen>/genomes<A or B>.npz` and wraps each row in a lightweight stand-in "dead creature" (`SimpleNamespace` with `brain`, `lifespan`, `numChildren`, and zeros for `health`, `hunger`, `thirst`). If this generation has no file for the species, falls back to the newest earlier generation that has one. Raises `ValueError` if the saved `layer_sizes` differ from the current `SHAPE`. Returns `None` if nothing is found. |
| `loadNextBrains()` | For each species, breeds the next generation with `getBest(dead, initial_count, elite_frac=0.1, survivor_frac=0.5)` (so `immigrant_frac` takes its default of 0.1), or `None` (random creatures) if nothing was loaded |
| `run(generations=100)` | Builds `World(i, brains1, brains2, initial_count)` for `i = start_gen + 1, ...` so existing folders are not overwritten, runs the UI, and feeds each world's GA output into the next |

Because the stand-in creatures have `hunger = thirst = 0`, the fitness used when resuming is $\text{lifespan}/100 + 10\cdot\text{children}$ only; the hunger and thirst terms (which the real GA sees) are missing. Selection after a resume therefore differs slightly from selection inside a normal run.

### 4.3 `src/neuralNetStuff/brain.py`: `NeuralNetwork` (unchanged)

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

`SHAPE = [63, 63, 50, 45, 40, 25, 15, 10, 8]` (defined in `creatures.py`), so there are 7 hidden `tanh` layers and a linear output of 8.

| Layer | $n_l \to n_{l+1}$ | Parameters $(n_l+1)\,n_{l+1}$ |
|---|---|---|
| 1 | $63 \to 63$ | 4032 |
| 2 | $63 \to 50$ | 3200 |
| 3 | $50 \to 45$ | 2295 |
| 4 | $45 \to 40$ | 1840 |
| 5 | $40 \to 25$ | 1025 |
| 6 | $25 \to 15$ | 390 |
| 7 | $15 \to 10$ | 160 |
| 8 | $10 \to 8$ | 88 |
| **Total** | | **13030** (verified, and equal to the genome length stored in `weights/`) |

(That is 8 weight matrices, 7 of which are followed by `tanh`.)

### 4.4 `src/neuralNetStuff/geneticAlg.py`: evolution

The file header lists four strategies being considered:

| | Strategy |
|---|---|
| A | Many worlds per simulation; all creatures of a type in a world are identical; score the whole world; mutate the single creature design. |
| B | One world per simulation; every creature unique; score each creature on lifespan and children; next world = children of the best. |
| C | Hybrid: start with B, later add world extinction time to fitness and run many worlds in parallel. |
| D | Hybrid: B for actions, plus mutating the "final" (permanent) stats of the best creatures. |

**Strategy B is implemented.**

#### `fitness(creature)`

$$F = \frac{\text{lifespan}}{100} + 10\cdot\text{numChildren} + \max(h, 0) + \max(t, 0),$$

where $h$ and $t$ are the hunger and thirst left at death (only counted when positive). The old alternative formula after the first `return` is unreachable. Under this formula the largest possible score is about $150 + 100 + 100 + 10\cdot\text{children}$ (lifespan is capped at 15000 ticks).

#### `makeChildBrain(brain1, brain2, blend=0.2, mutation_rate=0.02, mutation_strength=0.1)`

Builds a child genome of length $n$ from parent genomes $\boldsymbol{\theta}^{(1)}, \boldsymbol{\theta}^{(2)}$:

1. **Uniform crossover:** for each gene, pick a parent with probability $\tfrac12$:
$$\beta_j = \begin{cases}\theta^{(1)}_j & u_j < 0.5\\ \theta^{(2)}_j & \text{otherwise}\end{cases}, \qquad u_j \sim \mathcal{U}(0,1)$$
2. **Sparse Gaussian mutation:** each gene mutates with probability `mutation_rate`:
$$\mu_j = \mathbb{1}[r_j < \text{rate}] \cdot z_j, \qquad z_j \sim \mathcal{N}(0, \sigma^2),\ \sigma = \text{mutation\_strength}$$

Child genome: $\theta^{\text{child}}_j = \beta_j + \mu_j$, returned as a new `NeuralNetwork` with the parents' layer sizes. The earlier "blend noise" step has been removed, so the `blend` argument is now unused.

#### `getBest(creatures, num_children=None, survivor_frac=0.3, wildcard_frac=0.15, elite_frac=0.05, immigrant_frac=0.1, tournament_size=10, mutation_rate=0.04, mutation_strength=0.2)`

Returns a list of `num_children` brains (default: as many as there are creatures; `[]` if `creatures` is empty). It also prints one line with the best creature's score, lifespan, health, hunger, thirst and children. With $N$ creatures and $C$ children:

| Step | What happens | Count |
|---|---|---|
| 1. Score | Sort `(fitness, brain, ...)` tuples, best first | $N$ |
| 2. Parent pool | Top survivors + random "wildcards" from the rest | $\max(2, \lfloor \text{survivor\_frac}\cdot N \rfloor)$ + $\min(\text{rest}, \lfloor 0.15N \rfloor)$ |
| 3. Elites | Best brains copied unchanged | $\min(C, \max(1, \lfloor \text{elite\_frac}\cdot C \rfloor))$ |
| 4. Immigrants | Brand-new random brains | $\lfloor \text{immigrant\_frac}\cdot C \rfloor$ (capped by space left) |
| 5. Selection | Tournament: sample `tournament_size` (at most the pool size) from the pool, keep the fittest | per parent |
| 6. Breeding | Two parents (retry up to 5 times if both are the same brain), per-child mutation rate $\times\,\mathcal{U}(0.5, 3)$ and strength $\times\,\mathcal{U}(0.5, 2)$, then `makeChildBrain` | until $C$ brains |

`World.hasEnded` and `ContinueMain` both call it with `survivor_frac=0.5` and `elite_frac=0.1` (so the other fractions stay at their defaults). Example from the headless run: herbivores had $N = 22$ dead creatures and $C = 20$: top $= 11$, wildcards $= 3$, pool $= 14$, elites $= 2$, immigrants $= 2$, bred $= 16$. The tournament size of 10 out of a pool of 14 means almost every parent is one of the very best few, which is strong selection pressure and loses diversity quickly.

### 4.5 `src/worldObjects/world.py`: the world

#### Module-level helpers (unchanged)

* **`merge_memory(new, old, max_n, now, max_age)`**: stamps each new sighting as `(x, y, now)`, keeps the `old` entries that were not re-sighted **and** are younger than the time limit, and truncates to `max_n`. An entry seen at tick $t_s$ is kept while
$$\text{now} - t_s < \text{max\_age},$$
so it is forgotten exactly `max_age` ticks after the creature last saw it. Re-seeing a thing refreshes its stamp. Current sightings outrank old ones.
* **`cone_candidates(creatures, pts, mem_counts, chunk=512)`**: vectorised cone test (Section 3.7) for a batch of creatures against a set of points; returns, per creature, visible points nearest-first, capped at `mem_counts[i]`. Processes 512 creatures per numpy batch.
* **`update_memory(group, pts, mem_attr, last_attr, time_attr, now)`**: runs `cone_candidates` for a group and writes `merge_memory(..., now, getattr(c, time_attr))` back to each creature's `last_attr` list with `setattr`. `World.tick` passes `now = self.tick_count`.

#### `WEIGHTS_DIR`

`Path(__file__).resolve().parents[2] / "weights"`, i.e. `evolution/weights`, wherever the program is started from.

#### Class `World`

**Class attributes:** `width = 500`, `height = 500` (used for spawning, drawing and input normalisation).

##### `__init__(self, id, brainsA=None, brainsB=None, initialCount=20)`

| Attribute | Value |
|---|---|
| `id`, `tick_count = 0`, `initialCount` | as given |
| `creaturesA` | one `Creature(i, 0, self, b)` per brain in `brainsA`, or `initialCount` random ones if `brainsA` is `None` |
| `creaturesB` | the same with type 1 and `brainsB` |
| `creaturesACount`, `creaturesBCount` | starting count; incremented to give newborns ids |
| `plants` | 49 plants on a $7 \times 7$ grid: $x, y \in \{62.5, 125, \dots, 437.5\}$ (`numPlants = 8`, `range(1, 8)`) |
| `waterSources` | 25 sources on a $5 \times 5$ grid: $x, y \in \{83.3, 166.7, \dots, 416.7\}$ (`numWater = 6`, `range(1, 6)`) |
| `corpses`, `newCreaturesA/B`, `deadCreaturesA/B` | empty lists |

Creatures start at uniformly random positions in $[0, 500]^2$, facing $\theta = 0$. Note that an *empty list* of brains is not the same as `None`: it builds a world with no creatures (see Section 9).

##### `snapshot(self) -> WorldSnapshot`

Builds `ObjectView`s for water (radius 5), plants (2), corpses (2), and both creature groups (3), plus a `CreatureView` per living creature (id, kind, health, hunger, thirst, children, lifespan, decision as plain floats). Returns a frozen `WorldSnapshot`. Should only be called from the simulation thread between ticks.

##### `makeCorpse(self, posX, posY, creature)`

Appends `Corpse(0, posX, posY, creature.hunger + creature.thirst)`: the meat on a corpse is the hunger plus thirst the creature still had when it died.

##### `createChild(self, type, brain1, brain2, posX, posY, health=50, food=10, water=10)`

Makes a child brain with `makeChildBrain(brain1, brain2)` (its own defaults: rate 0.02, strength 0.1, not the GA's), increments the relevant count, and appends a new `Creature` (type 0 or type 1 to match `type`, built with keyword arguments) at `(posX, posY)` to `newCreaturesA` or `newCreaturesB`. Newborns join the living list at the start of the next tick.

##### `tick(self)`

See Section 3.5.

##### `saveBrains(self)`

Creates `weights/<world id>/` and, for each species that has at least one dead creature, writes one compressed-less `.npz` (`np.savez`) file:

| Key | Shape | Content |
|---|---|---|
| `genomes` | $(n, P)$ | one genome per dead creature, $P = 13030$ |
| `layer_sizes` | $(9,)$ | the network shape, so a loader can rebuild `NeuralNetwork` |
| `lifespan` | $(n,)$ | ticks survived |
| `numChildren` | $(n,)$ | children |
| `fitness` | $(n,)$ | `fitness(c)` at the time of saving |
| `ids` | $(n,)$ | creature ids |

Files are 8 to 22 MB each. A species with no dead creatures is skipped (no file).

##### `hasEnded(self)`

Calls `saveBrains()`, then runs the GA: `brainsA = getBest(deadCreaturesA, initialCount, elite_frac=0.1, survivor_frac=0.5, immigrant_frac=0.1)` (same for B), and returns `[brainsA, brainsB]`. Only dead creatures are scored and saved, so creatures still alive when the window is closed are not.

### 4.6 `src/worldObjects/creatures.py`: the creature

#### Module constants

| Name | Value | Meaning |
|---|---|---|
| `memorySize` | 25 | Intended total memory budget (**unused**) |
| `NUM_ENEMIES_MEM`, `NUM_ALLIES_MEM` | 10, 10 | Memory slots |
| `NUM_PLANT_MEM`, `NUM_WATER_MEM` | 3, 3 | Memory slots |
| `NUM_CORPSE_MEM` | 2 | Memory slots |
| `ENEMY_MEM_TIME`, `ALLY_MEM_TIME` | 50, 60 | Ticks before a moving creature's position is forgotten |
| `PLANT_MEM_TIME` | 300 | Ticks before a plant memory is forgotten |
| `WATER_MEM_TIME` | 600 | Ticks before a water memory is forgotten |
| `CORPSE_MEM_TIME` | 200 | Ticks before a corpse memory is forgotten |
| `CHILDREN_RANGE` | $(0, 10)$ | Input scaling range |
| `HEALTH_RANGE` | $(0, 100)$ | Input scaling range |
| `HUNGER_RANGE`, `THIRST_RANGE` | $(-40, 100)$ | Input scaling ranges |
| `DIRECTION_RANGE` | $(0, 2\pi)$ | Input scaling range |
| `BASE_INPUTS` | 7 | Own-state inputs |
| `MEMORY_SLOTS` | 28 | Sum of the five slot counts |
| `INPUT_SIZE` | $7 + 28 \times 2 = 63$ | Network input length (the comment on that line still says 57) |
| `SHAPE` | `[63, 63, 50, 45, 40, 25, 15, 10, 8]` | Network layer sizes |

`normalise(value, lo, hi)` maps a value into $[0,1]$ and clips anything outside:

$$\operatorname{norm}(v) = \min\!\Big(1, \max\!\Big(0, \frac{v - lo}{hi - lo}\Big)\Big).$$

#### `__init__(self, id, type, world, brain=None, posX=-1, posY=-1, health=50, water=25, food=25)`

| Attribute | Value |
|---|---|
| `world`, `id`, `type` | as given (0 herbivore, 1 carnivore) |
| `remove`, `lifespan`, `numChildren` | `False`, 0, 0 |
| `enemyMemoryTime`, `allyMemoryTime`, `plantMemoryTime`, `waterMemoryTime`, `corpseMemoryTime` | from the `..._MEM_TIME` constants (per creature, so they could later be evolved) |
| `numberOf...Memory` (allies, enemies, plants, water, corpses) | 10, 10, 3, 3, 2 |
| `health`, `hunger`, `thirst` | from `health`, `food`, `water` (as floats). Hunger and thirst count *down*: high is good. |
| `posX`, `posY` | given, or random in $[0, 500]$ if left at $-1$ |
| `direction` | 0 |
| `last...Seen` (five lists) | `[]`; entries are `(x, y, tickSeen)` |
| `brain` | given `NeuralNetwork`, or a new random one with `SHAPE` |
| `decision` | first forward pass on the starting input vector |

Per-species stats:

| `type` | Species | `maxSpeed` | `foodFromPlant` | `foodFromMeat` | `attackRange` | `attackPower` | `fieldOfView` $m$ | `viewDistance` |
|---|---|---|---|---|---|---|---|---|
| 0 | herbivore | 1 | 1 | 0.1 | 5 | 10 | 0.2 | 100 |
| 1 | carnivore | 2 | 0.5 | 10 | 10 | 20 | 1 | 150 |

#### `_flatten_memory(self, points, max_len)`

Turns the remembered `(x, y, tickSeen)` points (at most `max_len` of them) into exactly `2 * max_len` floats in the **creature's own frame**. For a remembered point at offset $(\Delta x, \Delta y)$ from the creature, with distance $d = \sqrt{\Delta x^2 + \Delta y^2}$ and facing $\theta$:

$$w = 1 - \frac{\min(d, 100)}{100} + 0.05, \qquad \big(a, b\big) = w\left(\frac{\Delta x\cos\theta + \Delta y\sin\theta}{d},\ \frac{-\Delta x\sin\theta + \Delta y\cos\theta}{d}\right)$$

So $a$ is "how far straight ahead" and $b$ "how far to the side", as a unit direction scaled by a weight that is $\approx 1.05$ for things right next to the creature and $0.05$ for things 100 or more away. Missing slots are padded with zeros. Because everything is relative, the network no longer needs to know absolute map coordinates to steer toward food.

#### `inputVector(self)`: the 63 network inputs

| Indices | Content |
|---|---|
| 0 | `numChildren`, normalised over $(0, 10)$ |
| 1 | `health`, normalised over $(0, 100)$ |
| 2 | `hunger`, normalised over $(-40, 100)$ |
| 3 | `thirst`, normalised over $(-40, 100)$ |
| 4-5 | `posX`, `posY`, normalised over the world size |
| 6 | `direction`, normalised over $(0, 2\pi)$ |
| 7-26 | 10 remembered enemies $(a, b)$ |
| 27-46 | 10 remembered allies $(a, b)$ (**always zeros for now**) |
| 47-52 | 3 remembered plants |
| 53-58 | 3 remembered water sources |
| 59-62 | 2 remembered corpses |

Returned as a `float32` numpy array. **Lifespan is deliberately not an input any more.**

#### `__str__(self)`

Still returns `""` (TODO).

#### `isDead(self)`

If health $\le 0$ or `remove` is already set: set `remove = True`, return `True`. Otherwise `False`. (It no longer forces `health = 0`.)

#### `tick(self)`: live and think

If not dead:

1. $h \leftarrow h - 0.012$ and $w \leftarrow w - 0.012$ (hunger and thirst).
2. If hunger $> 60$ **or** thirst $> 60$ and health $\le 100$: health $+1$.
3. If thirst $\le 0$ or hunger $\le 0$: health $-0.5$.
4. If `lifespan > 15000`: `remove = True` (death of old age).

Then, if still alive: `lifespan += 1` and `decision = brain.evaluate(inputVector())`.

Starting hunger and thirst are 25, so a creature that never eats or drinks hits zero after

$$\frac{25}{0.012} \approx 2083 \text{ ticks},$$

then loses 0.5 health per tick from 50, giving a lifespan of roughly 2180 ticks. The longest possible life is 15001 ticks (seen in the saved data).

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

* `move()`: $x \leftarrow x + v_{\max}\cos\theta$, $y \leftarrow y + v_{\max}\sin\theta$ with $v_{\max} = 1$ (herbivore) or $2$ (carnivore). No bounds check.
* `turnL()`: $\theta \leftarrow \theta - \pi/18$; `turnR()`: $\theta \leftarrow \theta + \pi/18$ (10°); then $\theta \leftarrow \theta \bmod 2\pi$.
* `damage(amount)`: `health -= amount`, then `isDead()`.

#### `attack(self)`

Type 0 (herbivore) targets `world.creaturesB`; type 1 (carnivore) targets `world.creaturesA`. It damages, by `attackPower`, **every** creature in that list that is inside the attack cone (same $m$ as vision, range `attackRange`).

#### `eatPlant(self)`

Takes the first plant in `world.plants` within `attackRange` (Euclidean distance, no cone), adds `plant.tryEat(foodFromPlant)` to hunger, caps at 100.

#### `eatCorpse(self)`

Same, over `world.corpses`, using `corpse.tryEat(foodFromMeat)`.

#### `drinkWater(self)`

Looks through the creature's **memory** `lastWaterSeen` (water never moves, so memory is accurate) for a source within `attackRange`; if found, thirst $+2$, capped at 100.

#### `procreate(self)` ("advanced" mode, active)

Allies are `world.creaturesA` for type 0, `world.creaturesB` for type 1. For the first ally (not the creature itself) within `attackRange`, if it is not dead:

1. Partner takes `attackPower` damage; self takes 20 damage.
2. Both lose 10 hunger and 5 thirst.
3. If both are still alive: `world.createChild(type, self.brain, p.brain, posX, posY, attackPower + 20, 40, 50)` and `numChildren += 1`. The child therefore starts with health $=$ `attackPower` $+ 20$ (30 for herbivores, 40 for carnivores), hunger 40 and thirst 50.

The old "easy mode" (free child, no costs) is kept as a commented string.

### 4.7 `src/worldObjects/plants.py`: plants, water and corpses

| Class | Fields | Methods |
|---|---|---|
| `Corpse(id, posX, posY, health)` | `id`, `posX`, `posY`, `health` (set to the dead creature's hunger + thirst), `remove` | `tick()`: health $-0.1$, remove at $\le 0$. `tryEat(d)`: gives up to `d` health; when exhausted, marks `remove`. |
| `Water(posX, posY)` | `posX`, `posY` | none (static, infinite) |
| `Plant(posX, posY)` | `posX`, `posY`, `health = 50` | `tick()`: health $+2$ (no cap). `tryEat(d)`: gives up to `d`; never removed. |

### 4.8 `src/worldObjects/snapshot.py`: UI data (unchanged)

Three `@dataclass(frozen=True)` classes, so the UI thread cannot change simulation data:

| Class | Fields |
|---|---|
| `ObjectView` | `kind` (`"creatureA"`, `"creatureB"`, `"plant"`, `"water"`, `"corpse"`), `x`, `y`, `radius = 1.0` |
| `CreatureView` | `id`, `kind`, `health`, `hunger`, `thirst`, `numChildren`, `lifespan`, `decision` (tuple of floats) |
| `WorldSnapshot` | `tick`, `width`, `height`, `objects` (tuple of `ObjectView`), `creatures` (tuple of `CreatureView`) |

### 4.9 `weights/`: saved generations

`weights/0` to `weights/37` exist (38 generations, written by the runs on 6 November). Each folder holds `genomesA.npz` and `genomesB.npz` in the format of Section 4.5. The two ends were inspected:

| Generation | Species | Dead creatures | Mean lifespan | Longest life | Total children |
|---|---|---|---|---|---|
| 0 | A (herbivore) | 83 | 1335 | 5194 | 23 |
| 0 | B (carnivore) | 142 | 1165 | 14145 | 82 |
| 37 | A (herbivore) | 77 | 1246 | 6954 | 17 |
| 37 | B (carnivore) | 150 | 1511 | 15001 (the old-age cap) | 90 |

All files use the current network shape (63 inputs, 13030 genes). Carnivores show a modest gain in mean lifespan; herbivores show none over these 38 generations. The stored `fitness` column was written by whichever fitness formula was current at the time: the generation 0 maximum (459) is higher than the current formula allows for that file (at most about 381), so stored fitness values are **not comparable across runs**. Compare `lifespan` and `numChildren` instead.

---

## 5. Default parameters at a glance

| Quantity | Value | Where |
|---|---|---|
| World size | $500 \times 500$ | `World` |
| Creatures per species | 20 (world 1: 55; `continueMain`: 60) | `World.__init__`, `main.py`, `continueMain.py` |
| Generations | 100 | `main.py`, `continueMain.py` |
| Plants / water sources | 49 ($7\times7$ grid) / 25 ($5\times5$ grid) | `World.__init__` |
| Start position / direction | uniform random in $[0,500]^2$ / $\theta = 0$ | `Creature.__init__` |
| Starting health / hunger / thirst | 50 / 25 / 25 | `Creature.__init__` |
| Newborn health / hunger / thirst | attackPower $+$ 20 / 40 / 50 | `procreate` |
| Hunger and thirst decay | 0.012 per tick | `Creature.tick` |
| Health regeneration / starvation damage | $+1$ if hunger or thirst $>60$ / $-0.5$ if either $\le 0$ | `Creature.tick` |
| Maximum age | 15000 ticks | `Creature.tick` |
| Max speed | herbivore 1, carnivore 2 | `Creature.__init__` |
| Turn step | $\pi/18$ (10°) | `turnL`, `turnR` |
| Field of view $m$ | herbivore 0.2, carnivore 1 | `Creature.__init__` |
| View distance | herbivore 100, carnivore 150 | `Creature.__init__` |
| Attack range / power | herbivore 5 / 10, carnivore 10 / 20 | `Creature.__init__` |
| Food per bite | herbivore 1 plant / 0.1 meat; carnivore 0.5 plant / 10 meat | `Creature.__init__` |
| Water per drink | 2 | `drinkWater` |
| Procreation cost | 10 hunger, 5 thirst each; self takes 20 damage; partner takes `attackPower` | `procreate` |
| Memory slots | enemies 10, allies 10, plants 3, water 3, corpses 2 (28 total) | `creatures.py` |
| Memory time limits | enemies 50, allies 60, plants 300, water 600, corpses 200 ticks | `creatures.py` |
| Network | `[63, 63, 50, 45, 40, 25, 15, 10, 8]`, `tanh` hidden, linear out, 13030 parameters | `creatures.py`, `brain.py` |
| Initial weights | $\mathcal{U}(-1, 1)$ | `NeuralNetwork` |
| Fitness | $\text{lifespan}/100 + 10\,\text{children} + \max(\text{hunger},0) + \max(\text{thirst},0)$ | `fitness` |
| GA fractions used by `hasEnded` | survivors 0.50, wildcards 0.15, elites 0.10, immigrants 0.10 | `World.hasEnded`, `getBest` |
| Tournament size | 10 | `getBest` |
| Mutation (GA breeding) | rate 0.04, $\sigma = 0.2$ (scaled per child by $\mathcal{U}(0.5,3)$ and $\mathcal{U}(0.5,2)$) | `getBest` |
| Mutation (in-world births) | rate 0.02, $\sigma = 0.1$ | `makeChildBrain` defaults via `createChild` |
| Plant regrowth / corpse rot | $+2$ / $-0.1$ per tick | `plants.py` |
| Sim speed | no sleep; as fast as the CPU allows | `sim_loop` |
| Snapshot rate / UI fps | 60 / 60 | `sim_loop`, `ui_loop` |
| Vision chunk size | 512 creatures | `cone_candidates` |

---

## 6. Controls

| Input | Effect |
|---|---|
| **Space** | Pause / resume the simulation |
| Close window | Ends the current world; the brains are saved, the GA runs and the next generation's window opens |
| (automatic) | A world also ends when no creatures are left alive |

Closing the window only ends the *current* world; to stop all 100 generations you have to close it repeatedly or interrupt the program (Ctrl+C in the terminal).

---

## 7. TODO list

### 7.1 TODO comments in the code

| # | File and line | TODO |
|---|---|---|
| 1 | `world.py:15` | Load a template of creatures, plants and water so each generation has the same start. |
| 2 | `world.py:91` | Restrict creature movement to the world bounds. |
| 3 | `world.py:206` | "Save dead stuff to mutate later" (stale: `saveBrains` now does this). |
| 4 | `creatures.py:78` | Enforce `memorySize` so the memory capacities sum to less than it. |
| 5 | `creatures.py:165` | Complete `__str__`. |
| 6 | `creatures.py:5` (header) | On death, record the lifetime and weights (now done through `saveBrains`; the header comment is stale). |

### 7.2 Unfinished features (no TODO written)

| # | Item |
|---|---|
| 7 | Ally memory is never updated, so input slots 27-46 are always zero (20 of the 63 inputs carry no information). |
| 8 | Strategies A, C and D from the `geneticAlg.py` header are not implemented. |
| 9 | No per-generation statistics beyond the one printed "best Score" line per species; nothing is plotted. |
| 10 | Permanent stats (speed, field of view, attack, memory sizes) are fixed per species and not evolved (strategy D). |
| 11 | Creatures always start facing $\theta = 0$. |
| 12 | `continueMain.py` command-line options (`--gens`, `--start`) are described in its header but not implemented. |

---

## 8. Suggested next steps (features, not bug fixes)

1. **Log generation statistics** (max / mean lifespan, children, extinction tick) to a CSV in `weights/` or a `logs/` folder, so progress can be plotted across all 38+ generations. Section 4.9 shows the herbivores are not obviously improving, and numbers are the quickest way to find out why.
2. **Fill ally memory** (one more `update_memory` call with each group's own positions), or shrink `NUM_ALLIES_MEM` to 0 until then so the network does not carry 20 dead inputs.
3. **Implement the `continueMain.py` arguments** with `argparse` (`--gens`, `--start`, `--count`) so the settings are not edited in the file.
4. **Give `continueMain.py` the real hunger/thirst** (save them in `saveBrains`) so resumed selection matches in-run selection.
5. **Headless fast mode**: a flag to run generations without pygame, since the simulation already runs without the sleep; saving brains after each world is already in place.
6. Later: **evolve the permanent stats** (strategy D) by appending them to the genome.

---

## 9. Known problems (listed, not analysed)

* World 1 in `main.py` ignores world 0's GA result and starts with 55 fresh random creatures per species.
* If a world ends with no dead creatures (for example the window is closed immediately), `getBest` returns `[]`, and `World(i, [], [])` then builds an **empty** world (an empty list is not `None`); every following world is empty too.
* `hasEnded()` only scores and saves dead creatures; survivors are lost if the window is closed early.
* Running `main.py` again overwrites `weights/0`, `weights/1`, ... (use `continueMain.py` to keep them).
* Resuming (`continueMain.py`) breeds from saved genomes with hunger and thirst treated as 0, so fitness differs slightly from an in-run GA.
* Fitness is measured after death, so the hunger and thirst terms reflect the state at death, not during life.
* The health regeneration test uses `or` (hunger $>60$ **or** thirst $>60$): a creature with a full stomach is healed $+1$ per tick even when its thirst is zero, which cancels the $-0.5$ starvation damage, so it cannot die of thirst while well fed. This may have been meant as `and`.
* A corpse made from a creature with negative hunger + thirst starts with negative meat; `tryEat` on it can return a negative amount.
* Movement is unbounded (creatures can leave the $500 \times 500$ area; the position inputs are clipped to $[0,1]$, so the network just sees "at the edge").
* `makeChildBrain` has an unused `blend` argument; `memorySize` is unused; the comment on `INPUT_SIZE` says 57 but the value is 63.
* Saved `fitness` values come from different fitness formulas in different runs (Section 4.9).

---

## 10. Quick-start

```bash
cd D:\Coding_\evolution\src
pip install numpy pygame
python main.py            # fresh run (overwrites weights/0, 1, ...); Space pauses; close the window to go to the next generation
python continueMain.py    # resume from the newest generation in weights/ (60 creatures per species, 100 more worlds)
```

Headless (no window), from inside `src`:

```python
from worldObjects import world

w = world.World(0)
while w.creaturesA or w.creaturesB or w.newCreaturesA or w.newCreaturesB:
    w.tick()
print(w.tick_count)                       # extinction tick (8028 with seed 0)
brainsA, brainsB = w.hasEnded()           # saves weights/0/ and returns 20 new brains per species
w2 = world.World(1, brainsA, brainsB)     # next generation
```

Note that `hasEnded()` writes into the real `weights/` folder, so change `world.WEIGHTS_DIR` first (for example to a temporary folder) when only experimenting.
