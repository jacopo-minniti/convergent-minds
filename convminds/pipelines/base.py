from abc import ABC, abstractmethod
from typing import Any, Dict

class BasePipeline(ABC):
    """
    Abstract base class for all brain-to-language pipelines.
    Provides a standardized interface for training, evaluation, and inference.
    """
    
    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the training process."""
        pass
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """Run evaluation metrics on the test set."""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Perform inference on new brain activity."""
        pass
