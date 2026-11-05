

class Corpse:
    def __init__(self, id,posX,posY):
        self.id=id
        self.posX=posX
        self.posY=posY
        self.remove=False
        self.health=100
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
    
    def __init__(self, posX,posY):
        self.posX=posX
        self.posY=posY

class Plant:
    
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