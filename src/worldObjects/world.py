## this is the rules of the world and the simulation space
## tick the simulation
## control canvas
## store creatures 
import numpy as np
from pathlib import Path

from .creatures import Creature
from .plants import Corpse, Plant, Water
from .snapshot import ObjectView, CreatureView, WorldSnapshot
from neuralNetStuff.geneticAlg import makeChildBrain,getBest,fitness

# evolution/weights, found relative to this file (src/worldObjects/world.py)
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
#TODO load a template of creatures plants and water so each gen has same start 


def merge_memory(new, old, max_n, now, max_age):
    """
    Merge this tick's sightings with older memories.
    new:     list of (x, y) seen this tick (nearest first)
    old:     list of (x, y, tickSeen) remembered from earlier ticks
    now:     current world tick
    max_age: how many ticks a memory lasts after the creature stops seeing it
    Returns a list of (x, y, tickSeen), newest sightings first, capped at max_n.
    """
    fresh = [(x, y, now) for x, y in new]                 # stamp new sightings with the current tick
    seen = set(new)
    kept = [m for m in old
            if (m[0], m[1]) not in seen                    # skip re-sightings (fresh copy already added)
            and now - m[2] < max_age]                      # forget memories that are too old
    merged = fresh + kept
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

def update_memory(group, pts, mem_attr, last_attr, time_attr, now):
    """
    mem_attr:  name of the capacity attribute   (e.g. "numberOfPlantMemory")
    last_attr: name of the memory list attribute (e.g. "lastPlantsSeen")
    time_attr: name of the memory time limit     (e.g. "plantMemoryTime")
    now:       current world tick
    """
    mems = [getattr(c, mem_attr) for c in group]
    new_lists = cone_candidates(group, pts, mems)
    for c, new, m in zip(group, new_lists, mems):
        setattr(c, last_attr,
                merge_memory(new, getattr(c, last_attr), m, now, getattr(c, time_attr)))

class World():
    
    #TODO actually restrict creatures movement to world bounds
    width=500
    height=500
    
    def __init__(self, id,brainsA=None,brainsB=None,initialCount=20):
        self.id = id
        self.tick_count = 0
        self.initialCount=initialCount
        if  not  brainsA==None:
            self.creaturesA=[Creature(i,0,self,b) for i,b in enumerate(brainsA)] 
            self.creaturesACount=len(brainsA)
        else:
            self.creaturesA=[Creature(i,0,self) for i in range(initialCount)] 
            self.creaturesACount=initialCount
        if  not  brainsB==None:
                self.creaturesB=[Creature(i,1,self,b) for  i,b in enumerate(brainsB)] 
                self.creaturesBCount=len(brainsB)
        else:
            self.creaturesB=[Creature(i,1,self) for i in range(initialCount)] 
            self.creaturesBCount=initialCount
        
        numPlants=8
        self.plants=[Plant(x*self.width/numPlants,y*self.height/numPlants) for x in range(1,numPlants) for y in range(1,numPlants)]
        numWater=6
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
                    int(c.lifespan),
                    tuple(float(v) for v in c.decision),
                ))

        return WorldSnapshot(self.tick_count, self.width, self.height,
                            tuple(objs), tuple(creatures))
        # makeCorpse and createChild unchanged
    
    def makeCorpse(self,posX,posY,creature :"Creature"):
        self.corpses.append(Corpse(0,posX,posY,creature.hunger+creature.thirst))
        
    def createChild(self,type,brain1,brain2,posX:"float",posY:"float",health:"float"=50,food:"float"=10,water:"float"=10):#type is 0 for creaturesA,1 for creaturesB
        brain=makeChildBrain(brain1,brain2)
        if type==0:
            self.creaturesACount+=1
            self.newCreaturesA.append(Creature(self.creaturesACount,0,self,brain,posX,posY,health=health, food=food, water=water))
        else:
            self.creaturesBCount+=1
            self.newCreaturesB.append(Creature(self.creaturesBCount,1,self,brain,posX,posY,health=health, food=food, water=water))
        
    def tick(self):
        self.tick_count += 1
        dead = [c for c in self.creaturesA  if c.remove]
        for c in dead:
            self.deadCreaturesA.append(c)
            self.makeCorpse(c.posX, c.posY,c)
        dead = [c for c in self.creaturesB  if c.remove]
        for c in dead:
            self.deadCreaturesB.append(c)
            self.makeCorpse(c.posX, c.posY,c)
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
            now = self.tick_count
            update_memory(group, enemyPts,           "numberOfEnemiesMemory", "lastEnemiesSeen", "enemyMemoryTime",  now)
            update_memory(group, plantsPoints,       "numberOfPlantMemory",   "lastPlantsSeen",  "plantMemoryTime",  now)
            update_memory(group, waterSourcesPoints, "numberOfWaterMemory",   "lastWaterSeen",   "waterMemoryTime",  now)
            update_memory(group, corpsesPoints,      "numberOfCorpseMemory",  "lastCorpsesSeen", "corpseMemoryTime", now)
        
        
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
        
        
    def saveBrains(self):
        """Save the brains of this world's dead creatures to weights/<world id>/.

        Writes one file per species (genomesA.npz / genomesB.npz) holding:
          genomes   (n, P) one row per creature, P = genome length
          layer_sizes the network shape (so a loader can rebuild NeuralNetwork)
          lifespan, numChildren, fitness   (n,) per creature, same row order
          ids       (n,) creature ids
        """
        folder = WEIGHTS_DIR / str(self.id)
        folder.mkdir(parents=True, exist_ok=True)
        for name, dead in (("A", self.deadCreaturesA), ("B", self.deadCreaturesB)):
            if not dead:
                continue
            np.savez(
                folder / f"genomes{name}.npz",
                genomes=np.stack([c.brain.genome for c in dead]),
                layer_sizes=np.array(dead[0].brain.layer_sizes),
                lifespan=np.array([c.lifespan for c in dead]),
                numChildren=np.array([c.numChildren for c in dead]),
                fitness=np.array([fitness(c) for c in dead]),
                ids=np.array([c.id for c in dead]),
            )

    def hasEnded(self):
        self.saveBrains()   # weights/<generation>/genomesA.npz and genomesB.npz
        '''
        for c  in self.deadCreaturesA:
            print(c.lifespan)
        for c  in self.deadCreaturesB:
            print(c.lifespan)
            '''
        brainsA=getBest(self.deadCreaturesA,self.initialCount,elite_frac=0.1,survivor_frac=0.5,immigrant_frac=0.1)
        brainsB=getBest(self.deadCreaturesB,self.initialCount,elite_frac=0.1,survivor_frac=0.5,immigrant_frac=0.1)
        return [brainsA,brainsB]
        