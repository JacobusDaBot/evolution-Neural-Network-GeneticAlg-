## this is the rules of the world and the simulation space
## tick the simulation
## control canvas
## store creatures 
import numpy as np

from src.worldObjects.creatures import Creature

import numpy as np

from src.worldObjects.plants import Corpse, Plant, Water

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
    creaturesA=[]
    creaturesB=[]
    plants=[]
    waterSources=[]
    corpses=[]
    #TODO actually restrict creatures movement to world bounds
    width=500
    height=500
    def __init__(self, id):
        self.id = id
        self.creaturesA=[Creature(i,0) for i in range(10)] 
        self.creaturesB=[Creature(i,1) for i in range(10)]
        self.plants=[Plant(x,y) for x in range(5) for y in range(5)]
        self.waterSources=[Water(x,y) for x in range(2) for y in range(2)]
        
    def makeCorpse(self,posX,posY):
        self.corpses.append(Corpse(0,posX,posY))
        
    def createChild(self,type,brain1,brain2):
        
        
        pass
        
        
        
        
        
    def tick(self):
        ### gather info for all creatures
        creaturesAPoints = [(c.posX, c.posY) for c in self.creaturesA]
        creaturesBPoints = [(c.posX, c.posY) for c in self.creaturesB]
        plantsPoints     = [(p.posX, p.posY) for p in self.plants]
        waterSourcesPoints = [(w.posX, w.posY) for w in self.waterSources]
        corpsesPoints = [(w.posX, w.posY) for w in self.corpses]
        #TODO draw->pass the above points to a listener on a different thread to draw it


        ##update all creatures memory
        for group, enemyPts in ((self.creaturesA, creaturesBPoints),
                                (self.creaturesB, creaturesAPoints)):#creatures a and b both see eachother as enemies
            update_memory(group, enemyPts,              "numberOfEnemiesMemory", "lastEnemiesSeen")
            update_memory(group, plantsPoints,     "numberOfPlantMemory",   "lastPlantsSeen")
            update_memory(group, waterSourcesPoints, "numberOfWaterMemory", "lastWaterSeen")
            update_memory(group, corpsesPoints, "numberOfCorpseMemory", "lastCorpsesSeen")
        
        
        ##tick plants and creatures and corpses
        for c in self.creaturesA:##TODO all creatures think
            c.tick()
        for c in self.creaturesB:##TODO all creatures think
            c.tick()
        for c in self.corpses:
            c.tick()
        for p in self.plants:
            p.tick()
            
        ## TODO all creatures act
        for c in self.creaturesA:
            c.act()
        for c in self.creaturesB:
            c.act()
        ##remove dead stuff TODO keep old creatures to mutate/evolve
        self.corpses = [c for c in self.corpses if not c.remove]
        self.creaturesA = [c for c in self.creaturesA if not c.remove]
        self.creaturesB = [c for c in self.creaturesB if not c.remove]
        
        