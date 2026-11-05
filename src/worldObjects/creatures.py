## functions to be called by creatures
## defines creature data
## that control how they do stuff like move, see, act
## controls how long they can remember stuff after they stopp seeing it
## when it dies it must record the life time and weightss of the creature
## bla bla blaa



from math import sqrt
import math
import numpy as np
from typing import TYPE_CHECKING
from neuralNetStuff.brain import NeuralNetwork
import random
if TYPE_CHECKING:  # only for type hints; a real import here would be circular (world.py imports this file)
    from .world import World

memorySize=25#how many things it can remember at once
NUM_ENEMIES_MEM = 5
NUM_ALLIES_MEM  = 5
NUM_PLANT_MEM   = 5
NUM_WATER_MEM   = 5
NUM_CORPSE_MEM  = 5

BASE_INPUTS = 8
MEMORY_SLOTS = (NUM_ENEMIES_MEM + NUM_ALLIES_MEM + NUM_PLANT_MEM +
                NUM_WATER_MEM + NUM_CORPSE_MEM)
INPUT_SIZE = BASE_INPUTS + MEMORY_SLOTS * 2     # 58
SHAPE = [INPUT_SIZE, INPUT_SIZE,40, 25, 10,8]
class Creature:
    
    def __init__(self, id, type,world: "World",posX=-1,posY=-1,brain=None,health=50):
        self.world = world
        self.id = id
        self.type = type  # 0 herbivore, 1 carnivore

        # state
        self.remove = False        # is it dead
        self.lifespan = 0
        self.numChildren = 0

        # final/permanent stats
        self.maxSpeed = 1          # how fast it can move
        self.foodFromPlant = 0     # food gained from eating a plant
        self.foodFromMeat = 0      # food gained from eating a creature
        self.attackPower = 0       # how much damage it can do
        self.attackRange = 0       # how far it can attack

        self.fieldOfView = 1       # as a gradient m -> y = mx + c (ignore c)
        self.viewDistance = sqrt(10 + 10 * self.fieldOfView)  # derived from fieldOfView

        # TODO implement memorySize so the sum of all memory types must be < memorySize
        self.numberOfAlliesMemory = 5
        self.numberOfEnemiesMemory = 5
        self.numberOfPlantMemory = 5
        self.numberOfWaterMemory = 5
        self.numberOfCorpseMemory = 5

        # per-tick stats
        self.health = health           # out of 100
        self.hunger = 25           # out of 100
        self.thirst = 25           # out of 100
        self.posX=posX
        if posX==-1:
            self.posX = random.uniform(0, world.width)            
        self.posY=posY
        if posY==-1:
            self.posY = random.uniform(0, world.height)  
        self.direction = 0         # looking direction in radians

        # memories: lists of (x, y), max sizes set by numberOf...Memory above
        self.lastEnemiesSeen = []
        self.lastAlliesSeen = []
        self.lastPlantsSeen = []
        self.lastWaterSeen = []
        self.lastCorpsesSeen = []

        # neural net
        self.brain = brain if brain is not None else NeuralNetwork(SHAPE)
        
        self.decision = []         # output vector

        if type == 0:  # herbivore
            self.foodFromPlant = 1
            self.foodFromMeat = 0.1
            self.attackRange = 5
            self.attackPower = 10
        elif type == 1:  # carnivore
            self.foodFromPlant = 0.6
            self.foodFromMeat = 10
            self.attackRange = 10
            self.attackPower = 20
        self.decision=self.brain.evaluate(self.inputVector())# your brain forward pass
            
    def _flatten_memory(self, points, max_len):
        """[(x, y), ...] -> flat list of exactly max_len * 2 floats, zero padded."""
        flat = []
        for x, y in points[:max_len]:          # truncate if somehow too long
            flat.append(x)
            flat.append(y)
        flat.extend([0.0] * (max_len * 2 - len(flat)))   # pad the rest
        return flat

    def inputVector(self):
        v = [
            self.lifespan,
            self.numChildren,
            self.health,
            self.hunger,
            self.thirst,
            self.posX,
            self.posY,
            self.direction,
        ]
        v += self._flatten_memory(self.lastEnemiesSeen, self.numberOfEnemiesMemory)
        v += self._flatten_memory(self.lastAlliesSeen,  self.numberOfAlliesMemory)
        v += self._flatten_memory(self.lastPlantsSeen,  self.numberOfPlantMemory)
        v += self._flatten_memory(self.lastWaterSeen,   self.numberOfWaterMemory)
        v += self._flatten_memory(self.lastCorpsesSeen, self.numberOfCorpseMemory)
        return np.array(v, dtype=np.float32)

    
    def __str__(self):
        #TODO complete
        return f""
        
    def tick(self):#think
        if self.health <= 0:
            self.remove=True
        #self.health+=1
        #if self.health>100:
        #    self.health=100
        if not self.remove:
            self.hunger-=0.2
            self.thirst-=0.2
            if self.hunger<=0 or self.thirst<= 0:
                self.health=0
                self.remove=True
            
        
        #get NN output->think
        if not self.remove:
            self.lifespan+=1
            self.decision=self.brain.evaluate(self.inputVector())# your brain forward pass
        
    def act(self):
        #can allways turn but can only turn in one direction
        if self.health <= 0:
            self.remove=True
            return
        if self.remove==True:
            return
     
        
        if (self.health>0):
            if  (self.decision[0]>0) or (self.decision[1]>0):
                if self.decision[1]>self.decision[0]:
                    self.turnL()
                else:
                    self.turnR()
            
            #after turning we can move
            if self.decision[2]>0:
                self.move()
            
            #after that stuff we can only do one of the following
            actions = [self.attack, self.eatPlant, self.eatCorpse, self.drinkWater,self.procreate]
            choice = int(np.argmax(self.decision[3:8]))
            actions[choice]()
        
        
    def damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.remove=True
            
    
    def move(self):
        self.posX += self.maxSpeed * math.cos(self.direction)
        self.posY += self.maxSpeed * math.sin(self.direction)
    
    def turnL(self):
        self.direction -= math.pi / 9  # Turn left by 20 degrees
    
    def turnR(self):
        self.direction += math.pi / 9  # Turn right by 20 degrees
    
    def attack(self):
        """
        Attack enemies within the attack range and field of view.
        """
        enemies=[]
        if self.type==0:
            enemies=self.world.creaturesA
        elif self.type==1:
            enemies=self.world.creaturesB
        cos_a, sin_a = math.cos(self.direction), math.sin(self.direction)
        
        for e in enemies:
            dx, dy = e.posX - self.posX, e.posY - self.posY

            # Rotate so the cone direction `a` becomes the local x-axis
            forward = dx * cos_a + dy * sin_a
            lateral = -dx * sin_a + dy * cos_a

            if forward <= 0:                 # behind p (or level with it)
                continue
            if abs(lateral) > self.fieldOfView * forward:   # outside the -m..m cone
                continue

            dist = math.hypot(dx, dy)
            if dist > self.attackRange:                    # too far away
                continue

            e.damage(self.attackPower)
    
    def eatPlant(self):
        # Implement logic for eating a plant
        for p in self.world.plants:
            if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:
                # Assume eating a plant restores health or reduces hunger
                self.hunger +=p.tryEat(self.foodFromPlant)  
                if self.hunger > 100:
                    self.hunger = 100  # Cap hunger at 100  
                
                break
    
    def eatCorpse(self):
            # Implement logic for eating a corpse
            for p in self.world.corpses:
                if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:
                    # Assume eating a plant restores health or reduces hunger
                    self.hunger += self.foodFromMeat  
                    if self.hunger > 100:
                        self.hunger = 100  # Cap hunger at 100  
                    
                    break
    
    def drinkWater(self):#water doesnt move so we know that the creature will know exactly where the water is
                # Implement logic for eating a water
                for p in self.lastWaterSeen:
                    if sqrt((p[0] - self.posX) ** 2 + (p[1] - self.posY) ** 2) <= self.attackRange:
                        
                        self.thirst += 2  
                        if self.thirst > 100:
                            self.thirst = 100  # Cap thirst at 100  
                        
                        break
    
    def procreate(self):
        #advanced:
        allies=self.world.creaturesA
        if self.type==1:
            allies=self.world.creaturesB
       
        for p in allies:
            if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:    
                if ( not p.remove):
                    self.world.createChild(self.type,self.brain,p.brain,self.posX,self.posY,self.attackPower*2)
                    p.damage(self.attackPower)#reduce health of mate
                    self.damage(self.attackPower)
                    self.hunger-=self.attackPower #make self hungry
                    self.thirst-=self.attackPower #make self thirsty
                    self.numChildren+=1
                    break
        
        #easy mode
        ''' 
            
        for p in allies :
            if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:    
                self.numChildren+=1
                self.world.createChild(self.type,self.brain,p.brain,self.posX,self.posY)
                break'''
        
    
    

