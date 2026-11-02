## functions to be called by creatures
## defines creature data
## that control how they do stuff like move, see, act
## controls how long they can remember stuff after they stopp seeing it
## when it dies it must record the life time and weightss of the creature
## bla bla blaa



from math import sqrt
import math

memorySize=25#how many things it can remember at once

class Creature:
    remove=False#is it dead
    lifespan=0
    ##final/permanent stats
    id=0
    #gender=0#0 male 1 female
    maxSpeed=1#how fast it can move
    foodFromPlant=0#how much food it can get from eating a plant
    foodFromMeat=0#how much food it can get from eating a creature
    
    fieldOfView=1#as a gradient m->y=mx+c(ignore c)
    viewDistance=sqrt(10+10*fieldOfView)#how far it can see
    #TODO implement the memorySize sothat the sum of all memory types must be less than memorySize
    numberOfAlliesMemory=5#how many allies it can remember at once
    numberOfEnemiesMemory=5#how many enemies it can remember at once
    numberOfPlantMemory=5#how many plants it can remember at once
    numberOfWaterMemory=5#how many water sources it can remember at once
    numberOfCorpseMemory=5#how many corpses it can remember at once
    attackPower=0#how much damage it can do
    attackRange=0#how far it can attack
    
    ##update each tick stats    
    #weight=0
    hunger=0#out of 100
    thirst=0#out of 100
    health=0#out of 100
    posX=0
    posY=0
    direction=0#angle of looking direction in rads
    lastEnemiesSeen=[]#list of(x,y) coordinates of enemies seen->maxsize=numberOfEnemies
    lastAlliesSeen=[]#list of(x,y) coordinates of Allies seen->maxsize=numberOfAllies
    lastPlantsSeen=[]#list of(x,y) coordinates of Plants seen->maxsize=numberOfPlants
    lastWaterSeen=[]#list of(x,y) coordinates of Water seen->maxsize=numberOfWater
    lastCorpsesSeen=[]#list of(x,y) coordinates of Corpses seen->maxsize=numberOfCorpses
    type=-1
    brain=[]#this is the neural net weights
    decision=[]#decision is the neaural net output vector
    def __init__(self, id,type):
        self.id = id
        self.health=50
        self.hunger=50
        self.thirst=50
        #    TODO initialise posX,posY,direction from a file
        self.type=type
        if type==0:#herbivore
            self.foodFromPlant=1
            self.foodFromMeat=0.1
            self.attackRange=5
        elif type==1:#carnivore
            self.foodFromPlant=0.6
            self.foodFromMeat=10
            self.attackRange=10
    def __str__(self):
        #TODO complete
        return f""
        
    def tick(self):#think
        if self.health <= 0:
            self.remove=True
            return
        self.lifespan+=1
        #self.health+=1
        #if self.health>100:
        #    self.health=100
        self.hunger-=0.1
        self.thirst-=0.1
        if self.hunger<=0 or self.thirst<= 0:
                    self.health=0
                    self.remove=True
        #get NN output->think
        
        
    def act(self):
        #use decision to choose a function to do
        pass
    def damage(self, amount):
        self.health -= amount
    
    def move(self):
        self.posX += self.maxSpeed * math.cos(self.direction)
        self.posY += self.maxSpeed * math.sin(self.direction)
    
    def turnL(self):
        self.direction -= math.pi / 9  # Turn left by 20 degrees
    
    def turnR(self):
        self.direction += math.pi / 9  # Turn right by 20 degrees
    
    def attack(self,enemies):
        """
        Attack enemies within the attack range and field of view.
        """
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
    
    def eatPlant(self, plants):
        # Implement logic for eating a plant
        for p in plants:
            if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:
                # Assume eating a plant restores health or reduces hunger
                self.hunger +=p.tryEat(self.foodFromPlant)  
                if self.hunger > 100:
                    self.hunger = 100  # Cap hunger at 100  
                
                break
    
    def eatCorpse(self, corpses):
            # Implement logic for eating a corpse
            for p in corpses:
                if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:
                    # Assume eating a plant restores health or reduces hunger
                    self.hunger += self.foodFromMeat  
                    if self.hunger > 100:
                        self.hunger = 100  # Cap hunger at 100  
                    
                    break
    
    def drinkWater(self):
                # Implement logic for eating a water
                for p in self.lastWaterSeen:
                    if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:
                        
                        self.thirst += 2  
                        if self.thirst > 100:
                            self.thirst = 100  # Cap thirst at 100  
                        
                        break
    
    def Procreate(self,world):
        '''#advanced:
            def Procreate(self,world,allies_female):
        if (self.gender==1):#if male
            for p in allies_female:
                if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:    
                    world.createChild(self.type,self.brain,p.brain)
                    p.damage(self.attackPower)
                    break
        '''
        #easy mode
        for p in self.lastAlliesSeen:
            if sqrt((p.posX - self.posX) ** 2 + (p.posY - self.posY) ** 2) <= self.attackRange:    
                world.createChild(self.type,self.brain,p.brain)
                break
        
    
    

