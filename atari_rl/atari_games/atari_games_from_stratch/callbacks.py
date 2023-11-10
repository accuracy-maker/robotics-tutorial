from abc import ABC,abstractmethod
from typing import Any, Dict
import torch as T

class BaseCallback(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def after_step(self,
                   locals_: Dict[str,Any],
                   globals_: Dict[str,Any]) -> None:
        """
        Callback to run after each training step

        :param locals_: local variables at time of call
        :param globals_: global variables at time of call
        :return: None
        """
        pass
    
class SaveQNetworkCallback(BaseCallback):
    """
    Callback to save the Q network state dict of the model, after every training step
    """
    def __init__(self,
                 save_freq: int,
                 save_dir: str,
                 save_prefix: str) -> None:
        """
        :param save_dir: directory to save the model's state dict
        :param save_prefix: prefix for saved file name. Full name will be
            `f"{save_prefix}_step{step_number}"`
        """
        super().__init__()
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        
    def after_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Save Q network state dict

        :param locals_:
        :param globals_:
        :return:
        """
        step = locals_["step"]
        if step % self.save_freq == 0:
            save_file = f"{self.save_dir}/{self.save_prefix}_step{step}"
            q = locals_["step"].q
            T.save(q.state_dict(),save_file)
        