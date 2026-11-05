## this is the rules of the world and the simulation space
## tick the simulation
## control canvas
## store creatures 
import numpy as np

from .creatures import Creature
from .plants import Corpse, Plant, Water
from .snapshot import ObjectView, CreatureView, WorldSnapshot
from neuralNetStuff.geneticAlg import makeChildBrain
#TODO load a template of creatures plants and water so each gen has same start 


def merge_memory(new, old, max_n):
    seen = set(new)
    merged = new + [p for p in old if tuple(p) not in seen]  # skip re-sightings
    return merged[:max_n]

def cone_candidates(creatures, pts, mem_counts, chunk=512):
    """
    For each creature, return its visible points as a list of (x, y) tuples,
    sorted nearest-first and capped at that creature's mem_counts[i].
    """
    if len(pts) == 0:
        return [[] for _ in creatures]

    pts = np.asarray(pts, dtype=float)
    px, py = pts[:, 0], pts[:, 1]
    results = []

    for s in range(0, len(creatures), chunk):
        cs = creatures[s:s + chunk]

        x    = np.array([c.posX for c in cs], dtype=float)[:, None]
        y    = np.array([c.posY for c in cs], dtype=float)[:, None]
        d    = np.array([c.direction for c in cs], dtype=float)[:, None]
        fov  = np.array([c.fieldOfView for c in cs], dtype=float)[:, None]
        view = np.array([c.viewDistance for c in cs], dtype=float)[:, None]

        cos_a, sin_a = np.cos(d), np.sin(d)
        dx = px[None, :] - x                 # shape (creatures, points)
        dy = py[None, :] - y

        forward = dx * cos_a + dy * sin_a
        lateral = -dx * sin_a + dy * cos_a
        dist = np.hypot(dx, dy)

        ok = (forward > 0) & (np.abs(lateral) <= fov * forward) & (dist <= view)
        dist[~ok] = np.inf

        order = np.argsort(dist, axis=1)
        for row in range(len(cs)):
            n = mem_counts[s + row]
            idx = order[row, :n]
            idx = idx[np.isfinite(dist[row, idx])]       # drop non-visible
            results.append([tuple(p) for p in pts[idx].tolist()])

    return results

def update_memory(group, pts, mem_attr, last_attr):
    mems = [getattr(c, mem_attr) for c in group]
    new_lists = cone_candidates(group, pts, mems)
    for c, new, m in zip(group, new_lists, mems):
        setattr(c, last_attr, merge_memory(new, getattr(c, last_attr), m))

class World():
    
    #TODO actually restrict creatures movement to world bounds
    width=500
    height=500
    def __init__(self, id):
        self.id = id
        self.tick_count = 0
        
        self.creaturesACount=10
        
        self.creaturesBCount=10
        self.creaturesA=[Creature(i,0,self) for i in range(self.creaturesACount)] 
        self.creaturesB=[Creature(i,1,self) for i in range(self.creaturesBCount)]
        numPlants=6
        self.plants=[Plant(x*self.width/numPlants,y*self.height/numPlants) for x in range(1,numPlants) for y in range(1,numPlants)]
        numWater=4
        self.waterSources=[Water(x*self.width/numWater,y*self.height/numWater) for x in range(1,numWater) for y in range(1,numWater)]
        self.corpses = []   # see note below
        self.newCreaturesA=[]
        self.newCreaturesB=[]
        self.deadCreaturesA=[]
        self.deadCreaturesB=[]
                
    def snapshot(self) -> WorldSnapshot:
        """Immutable copy for the UI. Call only from the sim thread, between ticks."""
        objs = []
        objs += [ObjectView("water",     w.posX, w.posY, 5) for w in self.waterSources]
        objs += [ObjectView("plant",     p.posX, p.posY, 2) for p in self.plants]
        objs += [ObjectView("corpse",    c.posX, c.posY, 2) for c in self.corpses]
        objs += [ObjectView("creatureA", c.posX, c.posY, 3) for c in self.creaturesA]
        objs += [ObjectView("creatureB", c.posX, c.posY, 3) for c in self.creaturesB]
        creatures = []
        for kind, group in (("creatureA", self.creaturesA), ("creatureB", self.creaturesB)):
            for c in group:
                if c.remove:
                    continue
                creatures.append(CreatureView(
                    c.id, kind,
                    float(c.health), float(c.hunger), float(c.thirst),
                    int(c.numChildren),
                    tuple(float(v) for v in c.decision),
                ))

        return WorldSnapshot(self.tick_count, self.width, self.height,
                            tuple(objs), tuple(creatures))
        # makeCorpse and createChild unchanged
    
    def makeCorpse(self,posX,posY):
        self.corpses.append(Corpse(0,posX,posY))
        
    def createChild(self,type,brain1,brain2,posX,posY,health=50):#type is 0 for creaturesA,1 for creaturesB
        brain=makeChildBrain(brain1,brain2)
        if type==0:
            self.creaturesACount+=1
            self.newCreaturesA.append(Creature(self.creaturesACount,0,self,posX,posY,brain,health))
        else:
            self.creaturesBCount+=1
            self.newCreaturesB.append(Creature(self.creaturesBCount,0,self,posX,posY,brain,health))
        
    def tick(self):
        self.tick_count += 1
        dead = [c for c in self.creaturesA  if c.remove]
        for c in dead:
            self.deadCreaturesA.append(c)
                    
            self.makeCorpse(c.posX, c.posY)
        dead = [c for c in self.creaturesB  if c.remove]
        for c in dead:
            self.deadCreaturesB.append(c)
            self.makeCorpse(c.posX, c.posY)
        self.corpses = [c for c in self.corpses if not c.remove]
        self.creaturesA = [c for c in self.creaturesA +self.newCreaturesA if not c.remove ]
        self.creaturesB = [c for c in self.creaturesB +self.newCreaturesB  if not c.remove]
        self.newCreaturesA=[]
        self.newCreaturesB=[]
        ### gather info for all creatures
        creaturesAPoints = [(c.posX, c.posY) for c in self.creaturesA]
        creaturesBPoints = [(c.posX, c.posY) for c in self.creaturesB]
        plantsPoints     = [(p.posX, p.posY) for p in self.plants]
        waterSourcesPoints = [(w.posX, w.posY) for w in self.waterSources]
        corpsesPoints = [(w.posX, w.posY) for w in self.corpses]
        
        ##update all creatures memory
        for group, enemyPts in ((self.creaturesA, creaturesBPoints),
                                (self.creaturesB, creaturesAPoints)):#creatures a and b both see eachother as enemies
            update_memory(group, enemyPts,              "numberOfEnemiesMemory", "lastEnemiesSeen")
            update_memory(group, plantsPoints,     "numberOfPlantMemory",   "lastPlantsSeen")
            update_memory(group, waterSourcesPoints, "numberOfWaterMemory", "lastWaterSeen")
            update_memory(group, corpsesPoints, "numberOfCorpseMemory", "lastCorpsesSeen")
        
        
        ##tick plants and creatures and corpses
        for c in self.creaturesA:
            c.tick()
        for c in self.creaturesB:
            c.tick()
        for c in self.corpses:
            c.tick()
        for p in self.plants:
            p.tick()
            
        for c in self.creaturesA:
            c.act()
        for c in self.creaturesB:
            c.act()
        ##remove dead stuff
        ##TODO   save dead stuff to mutate later
        
        
    def hasEnded(self):
        #TODO save the brains->genetic stuff->make new brains
        for c  in self.deadCreaturesA:
            print(c.lifespan)
        for c  in self.deadCreaturesB:
                    print(c.lifespan)
        pass