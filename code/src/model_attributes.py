class ModelAttributes():
    def __init__(self, name) -> None:
        self.model_checkpoint = None
        self.mask = None
        self.start = None
        self.generate_attributes(name)

    def generate_attributes(self, name: str) -> None:
        if name == "roberta-base":
            self.model_checkpoint = "roberta-base"
            self.mask = "<mask>"
            self.start = 1
        
        if name == "bert-base-uncased":
            self.model_checkpoint = "bert-base-uncased"
            self.mask = "[MASK]"
            self.start = 0

        if name == "bert-large-uncased-whole-word-masking":
            self.model_checkpoint = "bert-large-uncased-whole-word-masking"
            self.mask = "[MASK]"
            self.start = 0

        if name == "facebook/bart-base":
            self.model_checkpoint = "facebook/bart-base"
            self.mask = "<mask>"
            self.start = 1

        if name == "gpt2":
            self.model_checkpoint = "gpt2"
            self.mask = "[MASK]"
            self.start = 0