from transformers import PretrainedConfig

class PLMConfig(PretrainedConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trim_value = kwargs.get("trim_value", 0)
        self.mean_pooling = kwargs.get("mean_pooling", False)