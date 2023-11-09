import torch
from torch import nn, Tensor
from pytorch_pretrained_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig


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
    

class NNClassifierWithBERT(nn.Module):
    def __init__(self, input_length, bert_path) -> None:
        super().__init__()

        self.input_length = input_length
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(p=0.1)
        # 768 hidden size of bert
        self.linear = nn.Linear(768, 2)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_length + 768, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),            
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tokens, masks, custom_embeddings):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        #concat_output = torch.cat((dropout_output, custom_embeddings), dim=1)
        #mlp_output = self.mlp(concat_output)
        #proba = self.softmax(mlp_output)
        #return mlp_output
        return self.linear(dropout_output)
    

class CustomModel(nn.Module):
    def __init__(self,checkpoint,num_labels): 
        super(CustomModel,self).__init__() 
        self.num_labels = num_labels 

        #Load Model with given checkpoint and extract its body
        self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
        self.dropout = nn.Dropout(0.1) 
        self.classifier = nn.Linear(768,num_labels) # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None,labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
        return TokenClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

