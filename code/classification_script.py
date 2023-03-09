# %%
import transformers
import torch
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from model_attributes import ModelAttributes
from tqdm import tqdm
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback, 
    IntervalStrategy, 
    AutoTokenizer,
    AutoModelForMaskedLM
)
from load_data import (
    load_data, 
    load_pretraining_data, 
    load_chatGPT_data
)
# %%
model_attributes = ModelAttributes("bert-base-uncased")

load = "./"
Path(load).mkdir(parents=True, exist_ok=True)

# change this for new models
save = "./models/model2"
Path(save).mkdir(parents=True, exist_ok=True)

load_prev = False # further train a previously trained model
train = True # train model or use a saved one. If this is False, load_prev is always True

DEVICE = "cuda:0"
# %%
model = AutoModelForMaskedLM.from_pretrained(model_attributes.model_checkpoint).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_attributes.model_checkpoint)
# %%
train_dataset, test_dataset, val_dataset = load_data(["TaskA_train.csv", "TaskA_test.csv", "TaskA_dev.csv"])
# %%
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
# %%
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
# %%
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
# %%
# load pretraining data
train_pre_dataset, test_pre_dataset = load_pretraining_data()
tokenized_train_pre = train_pre_dataset.map(tokenize_function, batched=True)
tokenized_test_pre = test_pre_dataset.map(tokenize_function, batched=True)
# %%
# load chatgpt data
train_pre_chatgpt_dataset, test_pre_chatgpt_dataset = load_chatGPT_data()
tokenized_train_pre_chatgpt = train_pre_chatgpt_dataset.map(tokenize_function, batched=True)
tokenized_test_pre_chatgpt = test_pre_chatgpt_dataset.map(tokenize_function, batched=True)
# %%
# %%
if load_prev or not train:
    model = AutoModelForMaskedLM.from_pretrained(load).to(DEVICE)
# %%
# 3 consecutive training loops with early stopping. First on and second are "further pretraining"
training_args = transformers.TrainingArguments(
    output_dir=save,
    overwrite_output_dir=False,
    per_device_train_batch_size=3, # change if not enough cuda memory
    per_device_eval_batch_size=3, # change if not enough cuda memory
    num_train_epochs=50,
    warmup_steps=100,
    load_best_model_at_end=True,
    evaluation_strategy = IntervalStrategy.STEPS,
    eval_steps = 50,
    save_total_limit = 10,
    metric_for_best_model="eval_loss",
    eval_accumulation_steps=10
)
# %%
if train:
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_pre,
        eval_dataset=tokenized_test_pre,
        data_collator=data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
        args=training_args
    )
    trainer.train()
    trainer.model.eval()
    trainer.save_model(output_dir=save)
# %%
if train:
    model = AutoModelForMaskedLM.from_pretrained(save).to(DEVICE) # don't know if necessary
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_pre_chatgpt,
        eval_dataset=tokenized_test_pre_chatgpt,
        data_collator=data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
        args=training_args
    )
    trainer.train()
    trainer.model.eval()
    trainer.save_model(output_dir=save)
# %%
if train:
    model = AutoModelForMaskedLM.from_pretrained(save).to(DEVICE) # don't know if necessary
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
        args=training_args
    )
    trainer.train()
    trainer.model.eval()
    trainer.save_model(output_dir=save)
# %%
model = model.to("cpu")
# %%
# example masking
text = f"This is a great {model_attributes.mask}."
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
mask_token_logits = token_logits[0, mask_token_index, :][0,1]
top_5_tokens = torch.argsort(-mask_token_logits)[:10].tolist()

for token in top_5_tokens:
    print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode(token)[model_attributes.start:])}")
# %%
words = ["therefore", "consequently", "hence", "thus", "so", "nevertheless", "however", "yet", "anyway", "although"]
word_mapping = {word: i for i, word in enumerate(words)} # map words to numbers to sort them later
# %%
def calculate_embedding_vector(dataset):
    probs = []
    for a in tqdm(range(len(dataset["text"]))):
        sentences = dataset["text"][a].split("<_c>") # split conclusion and rest
        conclusion = "<_c> " + sentences[-1][1:] # add "<_c>" symbol back to the conslusion

        # take every sentence and remove empty sentences
        sentences = sentences[0].split(".")
        sentences = [x for x in sentences if x]
        sentences = [x for x in sentences if x != " "]
        topic = sentences[0]
        
        # calculate the probability of each linking word occuring 
        # as first word in the conclusion for each sentence
        sentence_probs = []
        for sentence in sentences[1:]:
            probs_each = []
            # insert mask token at the beginning of the conclusion
            text = topic + "." + sentence + ", " + conclusion.replace("<_c>", "<_c> " + model_attributes.mask)
            
            label = dataset["labels"][a]
            inputs = tokenizer(text, return_tensors="pt")
            token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
            # Find the location of [MASK], extract its logits and sort
            mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
            mask_token_logits = token_logits[0, mask_token_index, :][0,1]
            masked_words_name = torch.sort(mask_token_logits, descending=True)[0]
            masked_words_id = torch.sort(mask_token_logits, descending=True)[1]

            # append the probability of each linking word for the current sentence to a list
            for i, masked_word in enumerate(masked_words_id):
                if tokenizer.decode(masked_word).lower() in words:
                    probs_each.append((label, tokenizer.decode(masked_word), torch.nn.functional.softmax(masked_words_name, dim=-1)[i].item()))
            sentence_probs.append(probs_each)
        
        # only take the maximum of all sentences in the final list
        probs.append(max(sentence_probs, key=lambda x: x[2]))
    
    # sort probabilities with the word mapping order to always keep the same ordering
    probs_sorted = [sorted(x, key=lambda x: word_mapping[x[1]]) for x in probs]

    # build the final embedding as list of (label, embedding vector) tuples
    embedding = [(np.int_(data_point[0,0]), np.float_(data_point[:, 2])) for data_point in np.array(probs_sorted)]

    return embedding
# %%
embedding_train = calculate_embedding_vector(train_dataset)
embedding_test = calculate_embedding_vector(test_dataset)

# just in case (doesn't work atm)
#np.savetxt(save + '/embedding_train.txt', embedding_train)
#np.savetxt(save +'/embedding_test.txt', embedding_test)
# %%
# standardise embeddings and transform to numpy arrays
x_train = np.array(list(zip(*embedding_train))[1])
x_train = StandardScaler().fit_transform(x_train)
x_train = np.array([StandardScaler().fit_transform(np.array(sample).reshape(-1,1)) for sample in x_train], dtype=object).reshape(-1,len(words))

x_test = np.array(list(zip(*embedding_test))[1])
x_test = StandardScaler().fit_transform(x_test)
x_test = np.array([StandardScaler().fit_transform(np.array(sample).reshape(-1,1)) for sample in x_test], dtype=object).reshape(-1,len(words))

y_train = np.array(list(zip(*embedding_train))[0])

y_test = np.array(list(zip(*embedding_test))[0])
# %%
# lgbm classification
train_data = lgb.Dataset(x_train, label=y_train)
validation_data = lgb.Dataset(x_test, label=y_test)
# %%
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'f1'},
    'num_leaves': 21,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'first_metric_only': True
}
num_round = 1000
# %%
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities, also not sure about just rounding...
    return 'f1', f1_score(y_true, y_hat, average='macro'), True
# %%
bst = lgb.train(params, train_data, num_round, valid_sets=[validation_data], callbacks=[lgb.early_stopping(stopping_rounds=40)], feval=lgb_f1_score)
# %%
ypred = bst.predict(x_test)
print(f1_score(y_test, (np.round(ypred)), average='macro'))
# %%
# svm classification
clf = svm.SVC(kernel="poly")
clf.fit(x_train, y_train)
# %%
ypred_svm = clf.predict(x_test)
print(f1_score(y_test, ypred_svm, average='macro'))
# %%
clf = svm.SVC(kernel="rbf")
clf.fit(x_train, y_train)
# %%
ypred_svm = clf.predict(x_test)
print(f1_score(y_test, ypred_svm, average='macro'))
# %%
# pca visualization
pca = PCA(n_components=2)
# %%
X_r = pca.fit(x_test).transform(x_test)
plt.figure()
colors = ["navy", "turquoise"]
lw = 2
y = y_test

for color, i, target_name in zip(colors, [0, 1], [0, 1]):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA")
# %%
# simple neural net classification (haven't tried a better one atm)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )


    def forward(self, x):
        return self.layers(x)
# %%
x_train = np.array(x_train, dtype=float)
dataset = TensorDataset(Tensor(x_train), Tensor(y_train))
trainloader = DataLoader(dataset, batch_size=10, shuffle=True)
# %%
torch.manual_seed(42)
mlp = MLP()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
# %%
for epoch in range(100):
    print(f'Starting epoch {epoch+1}')
    current_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs, targets = data
        targets = targets.long()
        
        optimizer.zero_grad()
        
        outputs = mlp(inputs)
        
        loss = loss_function(outputs, targets)
        
        loss.backward()
        
        optimizer.step()
        
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
            current_loss = 0.0

print('Training process has finished.')
# %%
x_test = np.array(x_test, dtype=float)
dataset_test = TensorDataset(Tensor(x_test), Tensor(y_test))
# %%
pred = []
testloader = DataLoader(dataset_test, batch_size=10, shuffle=False)
with torch.no_grad():
    for i, data in enumerate(testloader, 0):

        inputs, targets = data
        targets = targets.long()

        outputs = mlp(inputs)

        _, predicted = torch.max(outputs, 1)
        pred.extend(predicted.tolist())
# %%
print(f1_score(y_test, pred, average='macro'))
# %%
