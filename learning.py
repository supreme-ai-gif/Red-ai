# ===================== learning.py ==========================
import numpy as np

class Updater:
    def __init__(self, core, voice=None):
        self.core = core
        self.voice = voice

    def reinforce_response(self, index, reward=0.05):
        """
        Increase fitness for a given response index.
        """
        key=str(index)
        current=self.core.memory["response_fitness"].get(key,1.0)
        self.core.memory["response_fitness"][key]=current+reward
        self.core._save_memory()

    def mutate_brain(self, strength=0.05):
        """
        Slightly mutate neural weights.
        """
        for k in self.core.weights:
            noise=np.random.normal(0,strength,self.core.weights[k].shape)
            self.core.weights[k]+=noise
            norm=np.linalg.norm(self.core.weights[k])
            if norm>0: self.core.weights[k]/=norm
        self.core._save_memory()
