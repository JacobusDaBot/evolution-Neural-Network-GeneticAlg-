## takes all the creatures and the world and does genetic stuff to their networks
## 2 views on how to do this:
#A: many worlds per sim
# we make all creatures of a type exactly the same for each world
# -> score the whole world for fitness
# ->mutate only the one instance of creature of the world

#B: one world per sim
# we make all creatures in world be unique 
# -> score each creature based on how long it lives+how many children it had
#->make new world with all creatures as unique children of worthy parents in old world

#C: hybrid
#start with B to get functional creatures
#later on add the world extinction time as part of the fintess function and run many worlds in paralell

#D: hybrid
#do B for actions
#do mutate on world best creatures but change the final fields of the creature



#doing B
from worldObjects.creatures import Creature
from .brain import NeuralNetwork
import numpy as np

import random
def fitness(creature: "Creature"):#return an int score
    #hunger,thirst out of 100
    #life out of 15000
    #numChildren range 0 to 6 currently
    
    #maxSCORE~300+6*numChildren
    res=0
    if creature.hunger>0:
        res+=creature.hunger
    if creature.thirst>0:
        res+=creature.thirst
    return creature.lifespan/100 + 10 * creature.numChildren+res
    return (creature.lifespan/200)+creature.numChildren*100+creature.hunger+creature.thirst
     

#funct to mix 2 objects and to make 1 child
def makeChildBrain(brain1, brain2, blend=0.2, mutation_rate=0.02, mutation_strength=0.1):
    n = len(brain1.genome)

    # Uniform crossover
    mask = np.random.rand(n) < 0.5
    base = np.where(mask, brain1.genome, brain2.genome)


    # Small baseline mutation, independent of the parents' difference
    mut_mask = np.random.rand(n) < mutation_rate
    mutation = mut_mask * np.random.normal(0, mutation_strength, n)

    return NeuralNetwork(brain1.layer_sizes, base  + mutation)

#funct to fitness( all objects) and make worthy of reproduction group then return children group
def getBest(
    creatures,
    num_children=None,
    survivor_frac=0.3,     # top slice that can reproduce normally
    wildcard_frac=0.15,    # extra random parents drawn from the REST of the population
    elite_frac=0.05,       # copied unchanged
    immigrant_frac=0.1,   # brand-new random brains
    tournament_size=10,
    mutation_rate=0.04,
    mutation_strength=0.2,
):
    if not creatures:
        return []
    if num_children is None:
        num_children = len(creatures)

    layer_sizes = creatures[0].brain.layer_sizes

    # 1. Score and sort (best first)
    scored = sorted(
        ((fitness(c), c.brain,c.lifespan,c.health,c.hunger,c.thirst,c.numChildren) for c in creatures),
        key=lambda pair: pair[0],
        reverse=True,
    )
    print("best Score:",scored[0][0],"lifespan",int(scored[0][2]),"health",int(scored[0][3]),"hunger",int(scored[0][4]),"thirst",int(scored[0][5]),"numChildren",int(scored[0][6]))
    # 2. Parent pool = top survivors + random wildcards from the rest
    n_top = max(2, int(len(scored) * survivor_frac))
    top = scored[:n_top]
    rest = scored[n_top:]

    n_wild = min(len(rest), int(len(scored) * wildcard_frac))
    wildcards = random.sample(rest, n_wild) if n_wild > 0 else []

    parent_pool = top + wildcards

    next_gen = []

    # 3. Elites (unchanged)
    n_elite = min(num_children, max(1, int(num_children * elite_frac)))
    for sc in scored[:n_elite]:
        brain=sc[1]
        next_gen.append(NeuralNetwork(brain.layer_sizes, brain.genome.copy()))

    # 4. Random immigrants (fresh genes, protects against premature convergence)
    n_immigrants = min(num_children - len(next_gen), int(num_children * immigrant_frac))
    for _ in range(n_immigrants):
        next_gen.append(NeuralNetwork(layer_sizes))

    # 5. Tournament selection over the combined pool
    def pick_parent():
        k = min(tournament_size, len(parent_pool))
        contenders = random.sample(parent_pool, k)
        return max(contenders, key=lambda pair: pair[0])[1]

    # 6. Breed the rest, with per-child mutation variation
    while len(next_gen) < num_children:
        p1 = pick_parent()
        p2 = pick_parent()
        tries = 0
        while p2 is p1 and len(parent_pool) > 1 and tries < 5:
            p2 = pick_parent()
            tries += 1

        # Some children mutate gently, some aggressively
        rate = mutation_rate * random.uniform(0.5, 3.0)
        strength = mutation_strength * random.uniform(0.5, 2.0)

        next_gen.append(makeChildBrain(p1, p2, mutation_rate=rate, mutation_strength=strength))

    return next_gen













