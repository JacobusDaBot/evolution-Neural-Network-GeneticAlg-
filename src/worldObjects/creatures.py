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
NUM_ENEMIES_MEM = 10
NUM_ALLIES_MEM  = 10
NUM_PLANT_MEM   = 3
NUM_WATER_MEM   = 3
NUM_CORPSE_MEM  = 2

# how many ticks a memory lasts after the creature last saw that thing
ENEMY_MEM_TIME  = 50     # enemies move, so forget them quickly
ALLY_MEM_TIME   = 60     # allies move too
PLANT_MEM_TIME  = 300    # plants never move, but can be eaten down
WATER_MEM_TIME  = 600    # water never moves and never runs out
CORPSE_MEM_TIME = 200    # corpses rot away / get eaten

# ---- input standardisation ------------------------------------------------
# Every network input is rescaled to [0, 1] using its OWN expected range
# (lo, hi). Values outside the range are clipped, so the network never sees
# anything outside [0, 1] (e.g. a creature that wanders off the map).
CHILDREN_RANGE = (0.0, 10.0)           # expected number of children in a life
HEALTH_RANGE   = (0.0, 100.0)          # health is "out of 100"
HUNGER_RANGE   = (-40.0, 100.0)          # eating caps hunger at 100
THIRST_RANGE   = (-40.0, 100.0)          # drinking caps thirst at 100
DIRECTION_RANGE = (0.0, 2 * math.pi)   # turnL/turnR keep direction in [0, 2*pi)
# x / y positions (the creature's own and every remembered position) are
# scaled by the world size: x in (0, world.width), y in (0, world.height)


def normalise(value, lo, hi):
    """Map value from [lo, hi] to [0, 1], clipping anything outside."""
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


BASE_INPUTS = 7      # numChildren, health, hunger, thirst, posX, posY, direction
MEMORY_SLOTS = (NUM_ENEMIES_MEM + NUM_ALLIES_MEM + NUM_PLANT_MEM +
                NUM_WATER_MEM + NUM_CORPSE_MEM)
INPUT_SIZE = BASE_INPUTS + MEMORY_SLOTS * 2     # 57

SHAPE = [INPUT_SIZE, INPUT_SIZE,50, 45,40, 25,15 , 10,8]
class Creature:
    
    def __init__(self, id, type,world: "World",brain=None,posX:"float"=-1,posY:"float"=-1,health:"float"=50,water:"float"=25,food:"float"=25):
        self.world = world
        self.id = id
        self.type = type  # 0 herbivore, 1 carnivore

        # state
        self.remove = False        # is it dead
        self.lifespan = 0
        self.numChildren = 0

        # final/permanent stats
        
        self.foodFromPlant = 0     # food gained from eating a plant
        self.foodFromMeat = 0      # food gained from eating a creature
        self.attackPower = 0       # how much damage it can do
        self.attackRange = 0       # how far it can attack

       

        # TODO implement memorySize so the sum of all memory types must be < memorySize

        self.numberOfAlliesMemory = NUM_ALLIES_MEM
        self.numberOfEnemiesMemory = NUM_ENEMIES_MEM
        self.numberOfPlantMemory = NUM_PLANT_MEM
        self.numberOfWaterMemory = NUM_WATER_MEM
        self.numberOfCorpseMemory = NUM_CORPSE_MEM

        # memory time limits (in ticks) for each kind of memory
        self.enemyMemoryTime  = ENEMY_MEM_TIME
        self.allyMemoryTime   = ALLY_MEM_TIME
        self.plantMemoryTime  = PLANT_MEM_TIME
        self.waterMemoryTime  = WATER_MEM_TIME
        self.corpseMemoryTime = CORPSE_MEM_TIME

        # per-tick stats
        self.health = float(health)           # out of 100
        self.hunger =  float(food )          # out of 100
        self.thirst =  float(water )
        self.posX= float(posX)
        if posX==-1:
            self.posX = random.uniform(0, world.width)            
        self.posY= float(posY)
        if posY==-1:
            self.posY = random.uniform(0, world.height)  
        self.direction = 0         # looking direction in radians

        # memories: lists of (x, y, tickSeen), max sizes set by numberOf...Memory above,
        # entries are dropped once they are older than the ...MemoryTime limits
        self.lastEnemiesSeen = []
        self.lastAlliesSeen = []
        self.lastPlantsSeen = []
        self.lastWaterSeen = []
        self.lastCorpsesSeen = []

        # neural net
        self.brain = brain if brain is not None else NeuralNetwork(SHAPE)
        
        self.decision = []         # output vector

        if type == 0:  # herbivore
            self.maxSpeed = 1          # how fast it can move
            self.foodFromPlant = 1
            self.foodFromMeat = 0.1
            self.attackRange = 5
            self.attackPower = 10
            self.fieldOfView = 0.2       # as a gradient m -> y = mx + c (ignore c)
            self.viewDistance = 100#sqrt(70 + 70 * self.fieldOfView)  # derived from fieldOfView
        elif type == 1:  # carnivore
            self.maxSpeed = 2          # how fast it can move
            self.foodFromPlant = 0.5
            self.foodFromMeat = 10
            self.attackRange = 10
            self.attackPower = 20
            self.fieldOfView = 1       # as a gradient m -> y = mx + c (ignore c)
            self.viewDistance =150# sqrt(70 + 70 * self.fieldOfView)  # derived from fieldOfView
        self.decision=self.brain.evaluate(self.inputVector())# your brain forward pass
                
    def _flatten_memory(self, points, max_len):
        flat, c, s = [], math.cos(self.direction), math.sin(self.direction)
        for p in points[:max_len]:
            dx, dy = p[0] - self.posX, p[1] - self.posY
            d = math.hypot(dx, dy) + 1e-9
            w = 1.0 - min(d, 100.0) / 100.0 + 0.05
            flat += [w * (dx*c + dy*s) / d, w * (-dx*s + dy*c) / d]
        return flat + [0.0] * (max_len*2 - len(flat))

    def inputVector(self):
        # lifespan is deliberately NOT an input any more
        v = [
            normalise(self.numChildren, *CHILDREN_RANGE),
            normalise(self.health,      *HEALTH_RANGE),
            normalise(self.hunger,      *HUNGER_RANGE),
            normalise(self.thirst,      *THIRST_RANGE),
            normalise(self.posX, 0.0, self.world.width),
            normalise(self.posY, 0.0, self.world.height),
            normalise(self.direction,   *DIRECTION_RANGE),
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
    
    def isDead(self):
        if self.health<=0 or self.remove:
            self.remove=True
            return True
        return False
    
    def tick(self):#think
        
        #self.health+=1
        #if self.health>100:
        #    self.health=100
        if not self.isDead():
            self.hunger-=0.012
            self.thirst-=0.012
            if self.hunger>60 or self.thirst>60:
                if self.health<=100:
                    self.health+=1
            if self.thirst<=0 or self.hunger<=0:
                self.health-=0.5
            if self.lifespan>15000:
                self.remove=True
            
        
        #get NN output->think
        if not self.isDead():
            self.lifespan+=1
            self.decision=self.brain.evaluate(self.inputVector())# your brain forward pass
        
    def act(self):
        #can allways turn but can only turn in one direction
        if not self.isDead():
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
        self.isDead()
            
    
    def move(self):
        self.posX += self.maxSpeed * math.cos(self.direction)
        self.posY += self.maxSpeed * math.sin(self.direction)
    
    def turnL(self):
        self.direction -= math.pi / 18  # Turn left by 10 degrees
        self.direction %= 2*math.pi
    
    def turnR(self):
        self.direction += math.pi / 18  # Turn right by 10 degrees
        self.direction %= 2*math.pi
    
    def attack(self):
        """
        Attack enemies within the attack range and field of view.
        """
        enemies=[]
        if self.type==1:
            enemies=self.world.creaturesA
        elif self.type==0:
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
                    self.hunger += p.tryEat( self.foodFromMeat  )
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
            if p is self:
                continue
            if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:    
                if ( not p.isDead()):
                    p.damage(self.attackPower)#reduce health of mate
                    self.damage(20)
                    hungerCost=10
                    thirstCost=5
                    self.hunger-=hungerCost #make self hungry
                    self.thirst-=thirstCost #make self thirsty
                    p.hunger-=hungerCost #make p hungry
                    p.thirst-=thirstCost #make p thirsty
                    if (not self.isDead()) and (not p.isDead()):
                        self.world.createChild(self.type,self.brain,p.brain,self.posX,self.posY,self.attackPower+20,hungerCost*4,thirstCost*10)
                        self.numChildren+=1
                    break
        
        #easy mode
        ''' 
            
        for p in allies :
            if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:    
                self.numChildren+=1
                self.world.createChild(self.type,self.brain,p.brain,self.posX,self.posY)
                break'''
        
    
    

