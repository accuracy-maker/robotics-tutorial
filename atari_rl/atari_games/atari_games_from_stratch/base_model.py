from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def predict(self,obs):
        pass