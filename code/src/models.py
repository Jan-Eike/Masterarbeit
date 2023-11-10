import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel, AutoConfig


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
    
# adapted from https://jovian.com/rajbsangani/emotion-tuned-sarcasm/v/1?utm_source=embed
class NNClassifierWithBert(nn.Module):
    def __init__(self, checkpoint, num_labels, custom_embedding_length): 
        super().__init__() 
        self.num_labels = num_labels 
        self.custom_embedding_length = custom_embedding_length

        #Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True))
        self.dropout = nn.Dropout(0.1) 

        self.classifier = nn.Linear(768 + self.custom_embedding_length, num_labels) # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, custom_embeddings=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
        bert_embedding = sequence_output[:,0,:].view(-1,768) # [CLS] token

        concat_output = torch.cat((bert_embedding, torch.nn.functional.log_softmax(custom_embeddings, dim=1)), dim=1)

        logits = self.classifier(concat_output) # calculate losses

        logits = torch.nn.functional.log_softmax(logits, dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)