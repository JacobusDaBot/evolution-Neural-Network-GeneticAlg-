

class Corpse:
    id=0
    posX=0
    posY=0
    health=0
    remove=False
    def __init__(self, id,posX,posY):
        self.id=id
        self.posX=posX
        self.posY=posY
        
    def tick(self):
        self.health-=1
        if self.health<=0:
            self.remove=True
            
    def tryEat(self,damage):
            if self.health>damage:
                self.health-=damage
                return damage
            else:
                self.remove=True
                left=self.health
                self.health=0
                return left
class Water:
    posX=0
    posY=0
    def __init__(self, posX,posY):
        self.posX=posX
        self.posY=posY

class Plant:
    posX=0
    posY=0
    health=0
    def __init__(self, posX,posY):
     
        self.posX=posX
        self.posY=posY
        self.health=50
    
    def tick(self):
        self.health+=0.5
        
    def tryEat(self,damage):
        if self.health>damage:
            self.health-=damage
            return damage
        else:
            left=self.health
            self.health=0
            return left