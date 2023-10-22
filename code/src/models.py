import torch
from torch import nn, Tensor

class NNClassifier(nn.Module):
    def __init__(self, input_length) -> None:
        super().__init__()

        self.input_length = input_length

        self.layers = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.input_length, 2),
        )


    def forward(self, x):
        return torch.nn.functional.log_softmax(self.layers(x), dim=1)
    


