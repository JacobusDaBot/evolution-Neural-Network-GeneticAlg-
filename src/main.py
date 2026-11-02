


from src.worldObjects import world



if __name__=="__main__":
    w= world.World(1)
    
    while True:
        w.tick()