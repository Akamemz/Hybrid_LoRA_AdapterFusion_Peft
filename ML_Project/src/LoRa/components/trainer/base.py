# Either add a proper base class:
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def train(self):
        pass

