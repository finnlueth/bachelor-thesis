from abc import ABC, abstractmethod
import typing as T


class TrajectoryTokenizer(ABC):
    def __init__(self):
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the tokenizer model into memory."""
        pass

    @abstractmethod
    def tokenize(self, pdb_files: T.List[str]) -> T.List[str]:
        """Tokenize the input data."""
        pass

    @abstractmethod
    def detokenize(self, tokens: T.List[str]) -> T.List[str]:
        """Detokenize the input data."""
        pass
