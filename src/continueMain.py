# continueMain.py
# Same as main.py, but instead of starting from random brains it loads the most
# progressed generation saved in weights/<generation>/ and carries on from there.
#
# Run from inside src (same as main.py):
#     python continueMain.py                 # continue from the newest saved generation, 100 more worlds
#     python continueMain.py --gens 50       # only 50 more worlds
#     python continueMain.py --start 42      # continue from weights/42 instead of the newest
import argparse
import threading
from types import SimpleNamespace

import numpy as np

from main import SharedState, sim_loop, ui_loop          # reuse the UI + sim thread from main.py
from worldObjects import world
from worldObjects.world import WEIGHTS_DIR
from worldObjects.creatures import SHAPE
from neuralNetStuff.brain import NeuralNetwork
from neuralNetStuff.geneticAlg import getBest


class ContinueMain:
    def __init__(self, weights_dir=WEIGHTS_DIR, start_gen=None, initial_count=20):
        self.weights_dir = weights_dir
        self.initial_count = initial_count
        # the generation we load from: given, or the highest-numbered saved one
        self.start_gen = start_gen if start_gen is not None else self.latestGeneration()

    # ------------------------------------------------------------ loading
    def savedGenerations(self):
        """All generation numbers that have a folder with saved genomes, sorted."""
        gens = []
        if self.weights_dir.exists():
            for folder in self.weights_dir.iterdir():
                if folder.is_dir() and folder.name.isdigit() and any(folder.glob("genomes*.npz")):
                    gens.append(int(folder.name))
        return sorted(gens)

    def latestGeneration(self):
        gens = self.savedGenerations()
        return gens[-1] if gens else None

    def loadSpecies(self, species):
        """Load weights/<gen>/genomes<species>.npz as stand-in 'dead creatures'.

        saveBrains() stores the DEAD creatures of a world, i.e. the parents of the
        next generation. getBest() only needs .brain, .lifespan and .numChildren
        (for fitness), so lightweight objects are enough.
        If this generation has no file for the species (nobody of it died, so
        saveBrains skipped it), the newest earlier generation that has one is used.
        """
        for gen in reversed([g for g in self.savedGenerations() if g <= self.start_gen]):
            path = self.weights_dir / str(gen) / f"genomes{species}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            layer_sizes = [int(n) for n in data["layer_sizes"]]
            if layer_sizes != list(SHAPE):
                raise ValueError(
                    f"{path} was saved with network shape {layer_sizes}, "
                    f"but creatures.py now uses SHAPE = {list(SHAPE)}. "
                    "Use --start with a generation from the same shape, or start fresh with main.py."
                )
            dead = [
                SimpleNamespace(
                    brain=NeuralNetwork(layer_sizes, genome),
                    lifespan=int(life), numChildren=int(kids),
                    health=0.0, hunger=0.0, thirst=0.0,   # not saved; only used by old fitness versions
                )
                for genome, life, kids in zip(data["genomes"], data["lifespan"], data["numChildren"])
            ]
            print(f"species {species}: loaded {len(dead)} brains from weights/{gen}  "
                  f"(best lifespan {max(c.lifespan for c in dead)}, "
                  f"children {sum(c.numChildren for c in dead)})")
            return dead
        print(f"species {species}: no saved brains found, starting with random brains")
        return None

    def loadNextBrains(self):
        """Breed the next generation from the saved one, exactly like World.hasEnded() does."""
        brains = []
        for species in ("A", "B"):
            dead = self.loadSpecies(species)
            if dead is None:
                brains.append(None)          # World() makes random creatures for None
            else:
                brains.append(getBest(dead, self.initial_count, elite_frac=0.1, survivor_frac=0.5))
        return brains

    # ------------------------------------------------------------ running
    def run(self, generations=100):
        if self.start_gen is None:
            print("No saved generations in", self.weights_dir, "- starting from random brains.")
            brains1, brains2 = None, None
            first = 0
        else:
            print("continuing from generation", self.start_gen)
            brains1, brains2 = self.loadNextBrains()
            first = self.start_gen + 1       # new worlds get new ids, so nothing is overwritten

        for i in range(first, first + generations):
            print("world :", i)
            w = world.World(i, brains1, brains2, self.initial_count)
            shared = SharedState()
            sim_thread = threading.Thread(target=sim_loop, args=(w, shared), daemon=True)
            sim_thread.start()

            try:
                ui_loop(shared)              # blocks until the window closes or everything is dead
            finally:
                shared.stop.set()
                sim_thread.join(timeout=2)
            brains1, brains2 = shared.result   # GA output -> next world (saveBrains already ran)


if __name__ == "__main__":
    

    ContinueMain( initial_count=60).run(100)
