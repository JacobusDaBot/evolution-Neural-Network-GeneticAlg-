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
def fitness(creature: "Creature"):#return an int score
    
    return (creature.lifespan/50000)+creature.numChildren*5+creature.hunger+creature.health+creature.thirst
     

#funct to mix 2 objects and to make 1 child
def makeChildBrain(brain1, brain2, blend=0.5, mutation_rate=0.02, mutation_strength=0.1):
    n = len(brain1.genome)

    # Uniform crossover
    mask = np.random.rand(n) < 0.5
    base = np.where(mask, brain1.genome, brain2.genome)

    # +/- a fraction of the gap between the parents
    gap = np.abs(brain1.genome - brain2.genome)
    noise = gap * blend * np.random.uniform(-1, 1, n)

    # Small baseline mutation, independent of the parents' difference
    mut_mask = np.random.rand(n) < mutation_rate
    mutation = mut_mask * np.random.normal(0, mutation_strength, n)

    return NeuralNetwork(brain1.layer_sizes, base + noise + mutation)

#funct to fitness( all objects) and make worthy of reproduction group then return children group






















