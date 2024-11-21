from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    def __init__(self):
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the tokenizer model into memory."""
        pass

    @abstractmethod
    def tokenize(self, input_path: str, output_path: str):
        """Tokenize the input data and save the results to the output path."""
        pass 