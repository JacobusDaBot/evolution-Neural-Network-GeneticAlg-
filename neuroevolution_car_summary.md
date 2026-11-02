# Neural Networks with Genetic Algorithms: Self-Driving Car Example

A summary of our conversation on how neuroevolution works, using the example of training a car to stay inside the lines of a road.

---

## 1. The big picture

The system has two separate layers:

- **The neural network is the driver.** It maps what the car senses to what the car does. It knows nothing about driving until it is trained.
- **The genetic algorithm (GA) is the trainer.** It never looks inside the network to understand it. It creates many drivers, lets each one try, keeps the best, and breeds variations of them.

Ordinary neural networks are trained with backpropagation, which needs a differentiable loss. "Did the car crash?" is not differentiable with respect to the weights, so instead we search over the weights by trial, selection and mutation.

**Mapping of the example:**

| Concept | In the car example |
|---|---|
| Individual | One car, with its own network |
| Genome | The network's weights and biases, flattened into one vector |
| Fitness / scoring function | How far and how well the car drove before crashing |
| Network inputs | Speed and where the road lines are relative to the car |
| Network outputs | Speed controls (throttle/brake) and left/right steering |
| Population | A group of cars (for example 100) run each generation |

---

## 2. The neural network (the "brain")

### Inputs

The original idea was to feed in $(x, y)$, current speed and the lines. The lines have to be turned into numbers, and a robust choice is distances to the lines measured by a few "whisker" sensors:

$$
\mathbf{x} = \begin{bmatrix} v & d_{\text{left}} & d_{\text{right}} & d_{\text{ahead-left}} & d_{\text{ahead-right}} \end{bmatrix}^\top
$$

**Warning on raw $(x,y)$:** absolute position lets the network memorise where the turns are on one track instead of learning to drive. Distances to the lines are relative quantities that mean the same thing on any road, so they generalise. Normalise inputs (for example $d\in[0,1]$) so no input dominates.

### Forward pass

For $n$ inputs, $m$ hidden neurons and $k$ outputs:

$$
\mathbf{h} = \tanh\!\left(W_1 \mathbf{x} + \mathbf{b}_1\right), \qquad
\mathbf{y} = \tanh\!\left(W_2 \mathbf{h} + \mathbf{b}_2\right)
$$

with $W_1 \in \mathbb{R}^{m \times n}$ and $W_2 \in \mathbb{R}^{k \times m}$. The $\tanh$ squashes each value into $[-1,1]$.

### Outputs

$$
a_t = a_{\max}\, y_1 \;\;(\text{throttle/brake}), \qquad s_t = y_2 \;\;(\text{steering: } <0 \text{ left},\ >0 \text{ right})
$$

---

## 3. What $W_1$ and $W_2$ are

Each matrix is a table of "how strongly does this thing influence that thing?". Every number is the strength of one connection. A large positive value pushes the target up, a large negative value pushes it down, and a value near zero means "mostly ignore it".

$$
\underbrace{\mathbf{x}}_{\text{5 sensors}}
\;\xrightarrow{\;W_1\;}\;
\underbrace{\mathbf{h}}_{\text{6 hidden neurons}}
\;\xrightarrow{\;W_2\;}\;
\underbrace{\mathbf{y}}_{\text{2 controls}}
$$

- $W_1$ connects the inputs to the hidden layer. It turns raw sensor readings into useful features.
- $W_2$ connects the hidden layer to the outputs. It turns features into actions.

**Shape rule: rows are the neurons receiving, columns are the neurons sending.**

- $W_1 \in \mathbb{R}^{6\times 5}$: row $r$ is the recipe for hidden neuron $r$ (how much it listens to each of the 5 inputs).
- $W_2 \in \mathbb{R}^{2\times 6}$: row 1 is the throttle recipe, row 2 is the steering recipe, each over the 6 hidden neurons.

A single neuron: take the previous layer's values, multiply each by a weight, sum them, add a bias, then squash with $\tanh$:

$$
h_r = \tanh\!\Big(\sum_{j} (W_1)_{rj}\,x_j + (\mathbf{b}_1)_r\Big)
$$

**Why a hidden layer?** It lets the network build intermediate ideas such as "left wall close and I am going fast". Without it, each control would be a plain weighted sum of sensors, which struggles with combined conditions.

**Tiny numeric example** (two sensors $(0.2, 0.8)$, one hidden neuron with weights $(-2, +2)$, steering weight $1.5$, zero biases):

$$
h = \tanh(-2(0.2) + 2(0.8)) = \tanh(1.2) \approx 0.834, \qquad
y = \tanh(1.5 \times 0.834) \approx 0.848
$$

Since $y>0$ means steer right, the car turns away from the close left wall.

---

## 4. The genome: $\boldsymbol{\theta}$ is all the weights, flattened

$\boldsymbol{\theta}^A$ is not just one matrix. It is **all** of network $A$'s learnable numbers (weights and biases) concatenated into a single flat vector.

Parameter count:

$$
P = m(n+1) + k(m+1)
$$

For $n=5$, $m=6$, $k=2$: $P = 6\cdot 6 + 2\cdot 7 = 50$.

| Piece | Shape | Values |
|---|---|---|
| $W_1$ | $6\times5$ | 30 |
| $\mathbf{b}_1$ | 6 | 6 |
| $W_2$ | $2\times6$ | 12 |
| $\mathbf{b}_2$ | 2 | 2 |
| **Total** | | **50** |

Layout: $\theta_{1:30}=W_1$, $\theta_{31:36}=\mathbf{b}_1$, $\theta_{37:48}=W_2$, $\theta_{49:50}=\mathbf{b}_2$. Before driving, the vector is cut back into matrices with the same fixed layout.

**Why flatten?** Crossover and mutation work gene by gene, and a flat vector makes that trivial. The GA does not need to know which entry belongs to which matrix.

**Two important details:**

1. The architecture ($n$, $m$, $k$, the $\tanh$ choice, the reshape layout) is fixed code shared by every car. The genome holds only the numbers, which is why two genomes can be crossed over.
2. Entry $i$ means the same *slot*, not the same *function*. Hidden neurons are interchangeable labels, so parent $A$'s neuron 3 may do the job of parent $B$'s neuron 5. Mixing them can make a child worse than both parents (the *competing conventions* or permutation problem). Methods like NEAT track gene history to align parents before crossover.

---

## 5. Running one car (the simulation)

For each time step $t$:

1. **Sense:** compute $\mathbf{x}_t$ from the car state and the road.
2. **Think:** $\mathbf{y}_t = f_{\boldsymbol{\theta}}(\mathbf{x}_t)$.
3. **Act:** update the physics, for example

$$
v_{t+1} = v_t + a_t \Delta t, \qquad \phi_{t+1} = \phi_t + s_t\, \omega_{\max} \Delta t
$$
$$
x_{t+1} = x_t + v_{t+1}\cos\phi_{t+1}\,\Delta t, \qquad y_{t+1} = y_t + v_{t+1}\sin\phi_{t+1}\,\Delta t
$$

4. **Check:** if the car touches a line, or the time limit $T_{\max}$ is reached, the run ends.

---

## 6. The scoring (fitness) function

Fitness is the only link between "how the car behaved" and "which brains survive". An example:

$$
F(\boldsymbol{\theta}) = w_1 D \;+\; w_2 \bar{v} \;-\; w_3 \bar{e}
$$

- $D$: distance travelled along the road (progress, not wandering)
- $\bar{v}$: average speed
- $\bar{e}$: mean normalised offset from the lane centre
- A crash ends the run, which caps the score

**Reward hacking pitfalls:**

| Bad fitness | What evolution finds |
|---|---|
| Reward only survival time | The car sits still or circles slowly |
| Reward only raw distance | The car drives a wide loop and ignores the lane |
| Reward only speed | The car crashes at top speed, because it was fast for the first second |

Evolution optimises exactly what you measure, not what you meant. Progress along the centreline (for example checkpoints passed) is more robust than raw distance.

---

## 7. The genetic algorithm (the "trainer")

Keep a population of $N$ genomes. One loop is one generation.

**A. Evaluate.** Run every car and record $F_i$.

**B. Select parents.** Better cars are more likely to reproduce.

- Fitness-proportional (roulette): $p_i = F_i / \sum_j F_j$. For $F=(10,40,5,45)$ the sum is 100, so $p=(0.10,0.40,0.05,0.45)$.
- Tournament: pick $k$ random cars and take the best. Simpler and less sensitive to the scale of $F$.

**C. Crossover.** Build a child gene by gene:

$$
\theta^{C}_i = \begin{cases} \theta^{A}_i & \text{prob. } 0.5 \\ \theta^{B}_i & \text{prob. } 0.5 \end{cases}
\qquad\text{or blend: } \boldsymbol{\theta}^{C} = \gamma\,\boldsymbol{\theta}^{A} + (1-\gamma)\,\boldsymbol{\theta}^{B}
$$

**D. Mutation.** With small probability $p_m$ per gene:

$$
\theta_i' = \theta_i + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2), \quad \sigma \approx 0.1\text{ to }0.3
$$

Mutation is the only source of genuinely new behaviour. Crossover only recombines what already exists.

**E. Elitism and next generation.** Copy the top $e$ cars unchanged, fill the remaining $N-e$ slots with mutated children, and repeat.

---

## 8. What happens over generations

- **Generation 0:** random weights. Almost every car crashes within a second.
- **Early generations:** by luck a few cars steer slightly toward the road centre. They get a higher $D$, get selected, and their weights are copied and nudged.
- **Later:** selection pressure accumulates. Cars learn to track lanes, then corners, then speed. Average and best fitness rise until they plateau.

## 9. Why it works

The GA performs a search over $\mathbb{R}^{P}$ on a fitness landscape $F(\boldsymbol{\theta})$. Selection pushes the population uphill, while mutation and crossover explore around good regions. No gradient $\nabla_{\boldsymbol{\theta}} F$ is needed, so the fitness can be any messy, discontinuous simulation result. The cost is sample efficiency: each generation requires simulating $N$ cars.

## 10. Tradeoffs and failure modes

- **Exploration vs exploitation:** too little mutation causes premature convergence (stuck in a local optimum), too much destroys good solutions.
- **Selection pressure:** too strong loses diversity, too weak makes progress very slow.
- **Overfitting to one track:** score each car on several (or randomised) tracks and average.
- **Noisy fitness:** average several runs so one lucky run does not mislead selection.
- **Network size:** more neurons give more expressive power but a larger search space. Start small.
- **Beyond fixed topology:** methods like NEAT also evolve the network structure.

---

## 11. The 6 inputs / 3 outputs / 4 layers network

We then drew a deeper network: input layer (6) $\to$ hidden 1 (5) $\to$ hidden 2 (4) $\to$ output layer (3). "4 layers" was read as input + 2 hidden + output.

```
 Input (6)      Hidden 1 (5)    Hidden 2 (4)    Output (3)
 speed             o               o              throttle
 dist left         o               o              brake
 dist front-left   o               o              steering
 dist front        o               o
 dist front-right  o
 dist right
        \__ W1 (5x6) __/ \_ W2 (4x5) _/ \_ W3 (3x4) _/
```

Every neuron connects to every neuron in the next layer (full connectivity). With 4 layers there are 3 weight matrices:

$$
\mathbf{h}_1 = \tanh(W_1\mathbf{x} + \mathbf{b}_1), \qquad
\mathbf{h}_2 = \tanh(W_2\mathbf{h}_1 + \mathbf{b}_2), \qquad
\mathbf{y} = \tanh(W_3\mathbf{h}_2 + \mathbf{b}_3)
$$

$$
W_1 \in \mathbb{R}^{5\times 6}, \qquad W_2 \in \mathbb{R}^{4\times 5}, \qquad W_3 \in \mathbb{R}^{3\times 4}
$$

Genome size:

$$
\underbrace{(5\cdot 6) + (4\cdot 5) + (3\cdot 4)}_{62\text{ weights}} + \underbrace{(5+4+3)}_{12\text{ biases}} = 74
\quad\Rightarrow\quad \boldsymbol{\theta}\in\mathbb{R}^{74}
$$

**Notes:**

1. With $\tanh$ outputs, throttle and brake would each land in $[-1,1]$, but a negative brake makes no sense. A common fix is a sigmoid $\sigma(z)=\frac{1}{1+e^{-z}}\in[0,1]$ for those two outputs, keeping $\tanh$ for steering.
2. Depth costs search effort: going from 2 to 3 weight matrices grew $P$ from 50 to 74 here. Extra layers only help if the task needs the extra expressive power.

---

## 12. Whole system in pseudocode

```
initialise population of N random genomes theta
repeat for each generation:
    for each theta in population:
        reset car and track
        while not crashed and t < T_max:
            x_t = sense(car, road)
            y_t = forward_pass(theta, x_t)      # [throttle, brake, steering]
            car = physics_step(car, y_t)
        F(theta) = w1*D + w2*v_avg - w3*e_avg
    keep top e genomes (elitism)
    while new population not full:
        A, B  = select_parents(population, F)
        child = crossover(A, B)
        child = mutate(child, p_m, sigma)
        add child to new population
    population = new population
```

---

## 13. Check-your-understanding questions raised

1. If you rewarded only how long the car stays alive (no progress term), what behaviour would evolution converge to, and why is it a "correct" answer from the algorithm's point of view?
2. In $W_2$, if the entry in row 2 (steering) and column 4 (hidden neuron 4) is a large positive number, what does that say about how hidden neuron 4 affects steering?
3. If you added a seventh input (such as the current steering angle) to the 6-5-4-3 network, which weight matrix would change shape, and what would its new shape be?
