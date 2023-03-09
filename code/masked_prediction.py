# %%
from transformers import AutoModelForMaskedLM
from model_attributes import ModelAttributes
from transformers import AutoTokenizer
import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling
import transformers
from load_data import load_data, load_file
# %%
model_attributes = ModelAttributes("bert-base-uncased")
# %%
model = AutoModelForMaskedLM.from_pretrained(model_attributes.model_checkpoint)
#model = GPT2Model.from_pretrained(model_attributes.model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_attributes.model_checkpoint)
# %%
train_dataset, test_dataset, val_dataset = load_data(["TaskA_train.csv", "TaskA_test.csv", "TaskA_dev.csv"])
# %%
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
# %%
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
# %%
print(tokenized_train)
# %%
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
model = AutoModelForMaskedLM.from_pretrained(model_attributes.model_checkpoint).to("cuda:0")
# %%
from os import listdir
files = listdir("./argumentUnits")
train_pre = []
for file in files:
    with open("./argumentUnits/" + file) as f:
        lines = f.readlines()
        lines = " ".join(lines)
    train_pre.append(lines)

train_pre = [s.replace("\n", "") for s in train_pre]
train_pre = [s.replace("\"", "") for s in train_pre]
train_pre = [s.replace("\'", "") for s in train_pre]
# %%
train_pre_df = pd.DataFrame({"text": train_pre})
train_pre_df, test_pre_df = train_test_split(train_pre_df, train_size=0.9)
train_pre_dataset = Dataset.from_pandas(train_pre_df)
test_pre_dataset = Dataset.from_pandas(test_pre_df)
tokenized_train_pre = train_pre_dataset.map(tokenize_function, batched=True)
tokenized_test_pre = test_pre_dataset.map(tokenize_function, batched=True)
print(tokenized_train_pre)
# %%
save = "./models/model1"
# %%
model = AutoModelForMaskedLM.from_pretrained("./").to("cuda:0")
# %%
DEVICE = "cuda:0"

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_pre,
    eval_dataset=tokenized_test_pre,
    data_collator=data_collator,
    args=transformers.TrainingArguments(
        output_dir="./",
        overwrite_output_dir=False,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=3,
        warmup_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
)

trainer.train()

trainer.model.eval()
trainer.save_model(output_dir=save)
# %%
def first_word_to_lower(line: str) -> str:
    return " ".join(i[0].lower()+i[1:] for i in line.split(" "))
# %%
with open("./train_chatgpt.txt") as f:
    train_pre_chatgpt = f.readlines()
train_pre_chatgpt = [s.replace("\n", "") for s in train_pre_chatgpt]
train_pre_chatgpt = [s.split(": ")[-1] for s in train_pre_chatgpt]
train_pre_chatgpt = [". ".join(s.split(". ")[:-1]) + ", <_c> " + first_word_to_lower(s.split(". ")[-1]) for s in train_pre_chatgpt]
print(train_pre_chatgpt)
# %%
train_pre_chatgpt_df = pd.DataFrame({"text": train_pre_chatgpt})
train_pre_chatgpt_df, test_pre_chatgpt_df = train_test_split(train_pre_chatgpt_df, train_size=0.9)
train_pre_chatgpt_dataset = Dataset.from_pandas(train_pre_chatgpt_df)
test_pre_chatgpt_dataset = Dataset.from_pandas(test_pre_chatgpt_df)
tokenized_train_pre_chatgpt = train_pre_chatgpt_dataset.map(tokenize_function, batched=True)
tokenized_test_pre_chatgpt = test_pre_chatgpt_dataset.map(tokenize_function, batched=True)
# %%
model = AutoModelForMaskedLM.from_pretrained(save).to("cuda:0")
DEVICE = "cuda:0"

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_pre_chatgpt,
    eval_dataset=tokenized_test_pre_chatgpt,
    data_collator=data_collator,
    args=transformers.TrainingArguments(
        output_dir=save,
        overwrite_output_dir=False,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=15,
        warmup_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
)

trainer.train()

trainer.model.eval()
trainer.save_model(output_dir=save)
# %%
model = AutoModelForMaskedLM.from_pretrained(save).to("cuda:0")
# %%
DEVICE = "cuda:0"

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    args=transformers.TrainingArguments(
        output_dir=save,
        overwrite_output_dir=False,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=3,
        warmup_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
)

trainer.train()

trainer.model.eval()
trainer.save_model(output_dir=save)
# %%
model = model.to("cpu")
text = f"This is a great {model_attributes.mask}."
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
print(token_logits)
# Find the location of [MASK] and extract its logits
mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
mask_token_logits = token_logits[0, mask_token_index, :][0,1]
# Pick the [MASK] candidates with the highest logits
# We negate the array before argsort to get the largest, not the smallest, logits
top_5_tokens = torch.argsort(-mask_token_logits)[:10].tolist()

for token in top_5_tokens:
    print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode(token)[model_attributes.start:])}")

# %%
a = 1
text = test_dataset["text"][a].replace("<_c>", model_attributes.mask)
print(test_dataset["labels"])
print(text)
# %%
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
# Find the location of [MASK] and extract its logits
mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
mask_token_logits = token_logits[0, mask_token_index, :][0,1]
# Pick the [MASK] candidates with the highest logits
# We negate the array before argsort to get the largest, not the smallest, logits
top_5_tokens = torch.argsort(-mask_token_logits)[:50].tolist()
top_5_prob = torch.sort(-mask_token_logits)[0][:50]

for token in top_5_tokens:
    print(tokenizer.decode(token))
# %%
print(torch.nn.functional.softmax(top_5_prob, dim=-1))
# %%
print(tokenizer.decode(torch.argsort(-mask_token_logits)[1]), torch.max(torch.nn.functional.softmax(torch.sort(-mask_token_logits)[0], dim=-1)))
# %%
print(torch.nn.functional.softmax(torch.sort(mask_token_logits, descending=True)[0], dim=-1)[0].item())
print(tokenizer.decode(torch.sort(mask_token_logits, descending=True)[1][1]))
# %%
for a in range(10):
    text = test_dataset["text"][a].replace("<_c>", "<_c> " + model_attributes.mask)
    label = test_dataset["labels"][a]
    print(text)
    print(label)
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
    mask_token_logits = token_logits[0, mask_token_index, :][0,1]
    for i in range(10):
        print(torch.nn.functional.softmax(torch.sort(mask_token_logits, descending=True)[0], dim=-1)[i].item())
        print(tokenizer.decode(torch.sort(mask_token_logits, descending=True)[1][i]))
        print()
# %%
#words = ["thus", "a", "not", "the", "us", "but", "these", "offshore"]
words = ["therefore", "consequently", "hence", "thus", "so", "nevertheless", "however", "yet", "anyway", "although"]
word_mapping = {word: i for i, word in enumerate(words)}
#words = ["nevertheless", "however", "yet", "although", "anyway", "therefore", "hence", "thus", "so", "because"]
# %%
from tqdm import tqdm
probs = []
for a in tqdm(range(len(train_dataset["text"]))):
    probs_each = []
    text = train_dataset["text"][a].replace("<_c>", "<_c> " + model_attributes.mask)
    label = train_dataset["labels"][a]
    # print(text)
    # print(f"{label}: ", end="")
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
    mask_token_logits = token_logits[0, mask_token_index, :][0,1]
    masked_words_name = torch.sort(mask_token_logits, descending=True)[0]
    masked_words_id = torch.sort(mask_token_logits, descending=True)[1]
    for i, masked_word in enumerate(masked_words_id):
        for word_check in words:
            if tokenizer.decode(masked_word).lower() == word_check:
                probs_each.append((label, tokenizer.decode(masked_word), torch.nn.functional.softmax(masked_words_name, dim=-1)[i].item()))
    probs.append(probs_each)
#print(probs)
# %%
from tqdm import tqdm
probs = []
for a in tqdm(range(len(train_dataset["text"]))):
    sentences = train_dataset["text"][a].split("<_c>")
    conclusion = "<_c> " + sentences[-1][1:]
    sentences = sentences[0].split(".")
    sentences = [x for x in sentences if x]
    sentences = [x for x in sentences if x != " "]
    topic = sentences[0]
    sentence_probs = []
    for sentence in sentences[1:]:
        probs_each = []
        text = topic + "." + sentence + ", " + conclusion.replace("<_c>", "<_c> " + model_attributes.mask)
        label = train_dataset["labels"][a]
        inputs = tokenizer(text, return_tensors="pt")
        token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
        # Find the location of [MASK] and extract its logits
        mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
        mask_token_logits = token_logits[0, mask_token_index, :][0,1]
        masked_words_name = torch.sort(mask_token_logits, descending=True)[0]
        masked_words_id = torch.sort(mask_token_logits, descending=True)[1]
        for i, masked_word in enumerate(masked_words_id):
            if tokenizer.decode(masked_word).lower() in words:
                probs_each.append((label, tokenizer.decode(masked_word), torch.nn.functional.softmax(masked_words_name, dim=-1)[i].item()))
        sentence_probs.append(probs_each)
    probs.append(max(sentence_probs, key=lambda x: x[2]))
#print(probs)
# %%
probs_sorted = [sorted(x, key=lambda x: word_mapping[x[1]]) for x in probs]
probs_train = probs_sorted
print(probs_sorted)
# %%
probs_sorted = [sorted(x, key=lambda x: x[1]) for x in probs]
probs_df = pd.DataFrame(probs_sorted, columns=sorted(words))
probs_df
# %%
df = pd.DataFrame(columns=sorted(words))
for func in [np.mean, np.min, np.max]:
    for label in [0, 1]:
        l = []
        for i in range(len(words)):
            tmp_df = probs_df.iloc[:, i].apply(lambda x:pd.Series(x))
            l.append(func(tmp_df[2][tmp_df[0] == label]))
        df.loc[len(df)] = l
        l = []
        for i in range(len(words)):
            tmp_df = probs_df.iloc[:, i].apply(lambda x:pd.Series(x))
            l.append(-np.log(func(tmp_df[2][tmp_df[0] == label])))
        df.loc[len(df)] = l
    l = []
    for i in range(len(words)):
        tmp_df = probs_df.iloc[:, i].apply(lambda x:pd.Series(x))
        l.append(func(tmp_df[2]))
    df.loc[len(df)] = l
    l = []
    for i in range(len(words)):
        tmp_df = probs_df.iloc[:, i].apply(lambda x:pd.Series(x))
        l.append(-np.log(func(tmp_df[2])))
    df.loc[len(df)] = l
    #df.loc[len(df)] = (df.loc[len(df) - 1] - df.loc[len(df) - 2]) / df.loc[len(df) - 2]
    if func == np.mean:
        df.loc[len(df)] = (df.loc[1] - df.loc[5]) / df.loc[5]
        df.loc[len(df)] = (df.loc[3] - df.loc[5]) / df.loc[5]
df = df.rename(index={
    0: "mean_0", 1: "mean_0_log", 
    2: "mean_1", 3: "mean_1_log", 
    4: "mean_0_1", 5: "mean_0_1_log", 
    6: "percent diff mean_0_log and mean_0_1_log",
    7: "percent diff mean_1_log and mean_0_1_log",
    8: "min_0", 9: "min_0_log", 
    10: "min_1", 11: "min_1_log", 
    12: "min_0_1", 13: "min_0_1_log", 
    14: "max_0", 15: "max_0_log", 
    16: "max_1", 17: "max_1_log", 
    18: "max_0_1", 19: "max_0_1_log"
})
df
# %%
with open('probs.csv', 'w') as f:
    df.to_csv(f, index=True, header=True, decimal='.', sep=',', float_format='%.10f')
# %%
pd.read_csv("./probs.csv")
# %%
embedding_train = [(np.int_(data_point[0,0]), np.float_(data_point[:, 2])) for data_point in np.array(probs_train)]
# %%
embedding_train_percent = [(np.int_(data_point[0,0]), ((df.loc["mean_0_1_log"] + np.log(np.float_(data_point[:, 2]))) / np.log(np.float_(data_point[:, 2]))).to_numpy()) for data_point in np.array(probs_sorted)]
# %%
probs = []
for a in tqdm(range(len(test_dataset["text"]))):
    probs_each = []
    text = test_dataset["text"][a].replace("<_c>", "<_c> " + model_attributes.mask)
    label = test_dataset["labels"][a]
    # print(text)
    # print(f"{label}: ", end="")
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
    mask_token_logits = token_logits[0, mask_token_index, :][0,1]
    masked_words_name = torch.sort(mask_token_logits, descending=True)[0]
    masked_words_id = torch.sort(mask_token_logits, descending=True)[1]
    for i, masked_word in enumerate(masked_words_id):
        for word_check in words:
            if tokenizer.decode(masked_word).lower() == word_check:
                probs_each.append((label, tokenizer.decode(masked_word), torch.nn.functional.softmax(masked_words_name, dim=-1)[i].item()))
    probs.append(probs_each)
#print(probs)
# %%
probs = []
for a in tqdm(range(len(test_dataset["text"]))):
    sentences = test_dataset["text"][a].split("<_c>")
    conclusion = "<_c> " + sentences[-1][1:]
    sentences = sentences[0].split(".")
    sentences = [x for x in sentences if x]
    sentences = [x for x in sentences if x != " "]
    topic = sentences[0]
    sentence_probs = []
    for sentence in sentences[1:]:
        probs_each = []
        text = topic + "." + sentence + ", " + conclusion.replace("<_c>", "<_c> " + model_attributes.mask)
        label = test_dataset["labels"][a]
        inputs = tokenizer(text, return_tensors="pt")
        token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
        # Find the location of [MASK] and extract its logits
        mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
        mask_token_logits = token_logits[0, mask_token_index, :][0,1]
        masked_words_name = torch.sort(mask_token_logits, descending=True)[0]
        masked_words_id = torch.sort(mask_token_logits, descending=True)[1]
        for i, masked_word in enumerate(masked_words_id):
            if tokenizer.decode(masked_word).lower() in words:
                probs_each.append((label, tokenizer.decode(masked_word), torch.nn.functional.softmax(masked_words_name, dim=-1)[i].item()))
        sentence_probs.append(probs_each)
    probs.append(max(sentence_probs, key=lambda x: x[2])) 
#print(probs)
# %%
probs_sorted = [sorted(x, key=lambda x: word_mapping[x[1]]) for x in probs]
probs_test = probs_sorted
# %%
probs_df = pd.DataFrame(probs_sorted, columns=sorted(words))
probs_sorted = [sorted(x, key=lambda x: x[1]) for x in probs]
# %%
df2 = pd.DataFrame(columns=sorted(words))
for func in [np.mean, np.min, np.max]:
    for label in [0, 1]:
        l = []
        for i in range(len(words)):
            tmp_df = probs_df.iloc[:, i].apply(lambda x:pd.Series(x))
            l.append(func(tmp_df[2][tmp_df[0] == label]))
        df2.loc[len(df2)] = l
        l = []
        for i in range(len(words)):
            tmp_df = probs_df.iloc[:, i].apply(lambda x:pd.Series(x))
            l.append(-np.log(func(tmp_df[2][tmp_df[0] == label])))
        df2.loc[len(df2)] = l
    l = []
    for i in range(len(words)):
        tmp_df = probs_df.iloc[:, i].apply(lambda x:pd.Series(x))
        l.append(func(tmp_df[2]))
    df2.loc[len(df2)] = l
    l = []
    for i in range(len(words)):
        tmp_df = probs_df.iloc[:, i].apply(lambda x:pd.Series(x))
        l.append(-np.log(func(tmp_df[2])))
    df2.loc[len(df2)] = l
    #df.loc[len(df)] = (df.loc[len(df) - 1] - df.loc[len(df) - 2]) / df.loc[len(df) - 2]
    if func == np.mean:
        df2.loc[len(df2)] = (df2.loc[1] - df2.loc[5]) / df2.loc[5]
        df2.loc[len(df2)] = (df2.loc[3] - df2.loc[5]) / df2.loc[5]
df2 = df2.rename(index={
    0: "mean_0", 1: "mean_0_log", 
    2: "mean_1", 3: "mean_1_log", 
    4: "mean_0_1", 5: "mean_0_1_log", 
    6: "percent diff mean_0_log and mean_0_1_log",
    7: "percent diff mean_1_log and mean_0_1_log",
    8: "min_0", 9: "min_0_log", 
    10: "min_1", 11: "min_1_log", 
    12: "min_0_1", 13: "min_0_1_log", 
    14: "max_0", 15: "max_0_log", 
    16: "max_1", 17: "max_1_log", 
    18: "max_0_1", 19: "max_0_1_log"
})
df2
# %%
embedding_test = [(np.int_(data_point[0,0]), np.float_(data_point[:, 2])) for data_point in np.array(probs_test)]
# %%
embedding_test_percent = [(np.int_(data_point[0,0]), ((df2.loc["mean_0_1_log"] - np.log(np.float_(data_point[:, 2]))) / np.log(np.float_(data_point[:, 2]))).to_numpy()) for data_point in np.array(probs_sorted)]
# %%
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.metrics import f1_score
min_max_scaler = preprocessing.MinMaxScaler()
embedding_test_percent = embedding_test
embedding_train_percent = embedding_train
cutoff = np.array(embedding_test_percent, dtype=object)[:, 0].sum()/len(embedding_test_percent)
for j in range(1, 31, 2):
    lab = []
    pred = []
    for i in range(len(test_dataset["text"])):
        x = np.array(list(zip(*embedding_test_percent))[1])[i].reshape(1, -1)
        #x = (x-np.min(x))/(np.max(x)-np.min(x))
        y = np.array(list(zip(*embedding_train_percent))[1])
        #y = min_max_scaler.fit_transform(y)
        sim = cosine_similarity(x, y)
        idx = np.argsort(sim)[0, -j:]
        lab.append(embedding_test_percent[i][0])
        pred.append(np.array(embedding_train_percent, dtype=object)[idx, 0].sum()/np.array(embedding_train_percent, dtype=object)[idx, 0].shape[0] >= cutoff)
    print((np.array(lab) == np.array(pred)).sum()/np.array(lab).shape[0])
    print(f1_score(np.array(lab), np.array(pred), average='macro'))
# %%
from tqdm import tqdm
probs = []
for a in tqdm(range(len(train_dataset["text"]))):
    probs_each = []
    text = train_dataset["text"][a].replace("<_c>", "<_c> " + model_attributes.mask)
    label = train_dataset["labels"][a]
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
    mask_token_logits = token_logits[0, mask_token_index, :][0,1]
    masked_words_name = torch.sort(mask_token_logits, descending=True)[0]
    masked_words_id = torch.sort(mask_token_logits, descending=True)[1]
    for i in range(300):
        probs_each.append((label, tokenizer.decode(torch.sort(mask_token_logits, descending=True)[1][i])))
    probs.append(probs_each)
#print(probs)
# %%
zeros = []
ones = []
for l in probs:
    print(l[0])
    if l[0][0] == 0:
        zeros.append(l[0][1])
    else:
        ones.append(l[0][1])
# %%
from collections import Counter
print(Counter(zeros))
print()
print(Counter(ones))
# %%
print(np.array(test_dataset["labels"]).sum()/len(test_dataset["labels"]))
# %%
print(token_logits.shape)
# %%
a = 7
text = test_dataset["text"][a].replace("<_c>", model_attributes.mask)
label = test_dataset["labels"][a]
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
print(token_logits[0].shape)
print(len(text.split()))
# %%
print(text.split())
# %%
print(token_logits["logits"][0, 2, :].shape)
# %%
print(torch.nn.functional.softmax(token_logits["logits"][0, 2, :], dim=-1).shape)
# %%
# %%
import lightgbm as lgb
# %%
from sklearn.preprocessing import StandardScaler
x_test = np.array(list(zip(*embedding_test_percent))[1])
#x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
x_test = StandardScaler().fit_transform(x_test)
x_test = np.array([StandardScaler().fit_transform(np.array(sample).reshape(-1,1)) for sample in x_test], dtype=object).reshape(-1,len(words))
#x_test = StandardScaler().fit_transform(x_test)
x_train = np.array(list(zip(*embedding_train_percent))[1])
#x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
x_train = StandardScaler().fit_transform(x_train)
x_train = np.array([StandardScaler().fit_transform(np.array(sample).reshape(-1,1)) for sample in x_train], dtype=object).reshape(-1,len(words))
#x_train = StandardScaler().fit_transform(x_train)
print(x_test.shape)
y_test = np.array(list(zip(*embedding_test_percent))[0])
y_train = np.array(list(zip(*embedding_train_percent))[0])
# %%
np.savetxt('x_train.txt', x_train)
np.savetxt('x_test.txt', x_test)
np.savetxt('y_train.txt', y_train)
np.savetxt('y_test.txt', y_test)
# %%
train_data = lgb.Dataset(x_train, label=y_train)
# %%
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 21,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}
num_round = 10000
# %%
validation_data = lgb.Dataset(x_test, label=y_test)
# %%
from sklearn.metrics import f1_score
def round2(x, nump=True):
    if nump == True:
        return np.round(x)
    return np.int_(x >= (y_test.sum()/len(y_test)))

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = round2(y_hat, nump=True) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat, average='macro'), True
# %%
bst = lgb.train(params, train_data, num_round, valid_sets=[validation_data], feval=lgb_f1_score, )
# %%
lgb.cv(params, train_data, num_round, nfold=5)
# %%
print(bst)
# %%
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[validation_data], callbacks=[lgb.early_stopping(stopping_rounds=40)], feval=lgb_f1_score)
# %%
ypred = bst.predict(x_test)
print((round2(ypred) == y_test).sum()/len(ypred))
print(len((round2(ypred) == y_test)[(round2(ypred) == y_test) == True]))
print(len((round2(ypred) == y_test)[(round2(ypred) == y_test) == False]))
print(round2(ypred))
print(y_test)
print(round2(ypred).sum())
print(y_test.sum())
print(len(y_test))
print(y_test.sum()/len(y_test))
print(ypred)
print(f1_score(y_test, (round2(ypred)), average='macro'))
# %%
import shap
shap.initjs()
df_shap = pd.DataFrame(x_train, columns=words)
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(df_shap)
# %%
shap.summary_plot(shap_values, df_shap)
# %%
from sklearn import svm
# %%
clf = svm.SVC(kernel="poly")
clf.fit(x_train, y_train)
# %%
ypred_svm = clf.predict(x_test)
print(ypred_svm)
print(np.array(ypred_svm).sum()/len(ypred_svm))
print(y_test)
# %%
print(f1_score(y_test, ypred_svm, average='macro'))
print((ypred_svm == y_test).sum()/len(ypred))
# %%
print(np.mean(x_test, axis=1))
# %%
from sklearn.decomposition import PCA
# %%
pca = PCA(n_components=2)
# %%
import matplotlib.pyplot as plt
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
from tqdm import tqdm
words = sorted(["therefore", "consequently", "hence", "thus", "so", "nevertheless", "however", "yet", "anyway", "although"])
words.append("")
model = AutoModelForMaskedLM.from_pretrained("./trained").to("cpu")
test_df = pd.DataFrame({"text": test_dataset["text"], "labels": test_dataset["labels"]})
ppls_per_data_point_test = []
for _, data_point in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    ppls = []
    for word in words:
        tmp = data_point["text"].replace("<_c> ", "<_c> " + f"{word} ")
        if word == "":
            tmp = data_point["text"]
        test_dataset2 = Dataset.from_pandas(pd.DataFrame({"text": [tmp]}))
        encodings = tokenizer("\n\n".join(test_dataset2["text"]), return_tensors="pt")
        max_length = model.config.max_position_embeddings
        stride = 512
        seq_len = encodings["input_ids"].size()[1]
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings["input_ids"][:, begin_loc:end_loc].to("cpu")
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        ppls.append(ppl)
    ppls_per_data_point_test.append(ppls)
# %%
train_df = pd.DataFrame({"text": train_dataset["text"], "labels": train_dataset["labels"]})
ppls_per_data_point_train = []
for _, data_point in tqdm(train_df.iterrows(), total=train_df.shape[0]):
    ppls = []
    for word in words:
        tmp = data_point["text"].replace("<_c> ", "<_c> " + f"{word} ")
        if word == "":
            tmp = data_point["text"]
        train_dataset2 = Dataset.from_pandas(pd.DataFrame({"text": [tmp]}))
        encodings = tokenizer("\n\n".join(train_dataset2["text"]), return_tensors="pt")
        max_length = model.config.max_position_embeddings
        stride = 512
        seq_len = encodings["input_ids"].size()[1]
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings["input_ids"][:, begin_loc:end_loc].to("cpu")
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        ppls.append(ppl)
    ppls_per_data_point_train.append(ppls)
# %%
ppls_per_data_point_train
# %%
ppl_diff_per_data_point_test = [[((data_point[-1] - x) / x).item() for x in data_point[:-1]] for data_point in ppls_per_data_point_test]
ppl_diff_per_data_point_train = [[((data_point[-1] - x) / x).item() for x in data_point[:-1]] for data_point in ppls_per_data_point_train]
# %%
ppl_diff_per_data_point_train
# %%
import itertools
final_embedding_test = [[x for x in itertools.chain.from_iterable(itertools.zip_longest(list1,list2)) if x] for list1, list2 in zip(ppl_diff_per_data_point_test, x_test)]
final_embedding_train = [[x for x in itertools.chain.from_iterable(itertools.zip_longest(list1,list2)) if x] for list1, list2 in zip(ppl_diff_per_data_point_train, x_train)]
# %%
clf = svm.SVC(kernel="poly")
clf.fit(final_embedding_train, y_train)
# %%
ypred_svm = clf.predict(final_embedding_test)
print(ypred_svm)
print(np.array(ypred_svm).sum()/len(ypred_svm))
print(y_test)
# %%
print(f1_score(y_test, ypred_svm, average='macro'))
print((ypred_svm == y_test).sum()/len(ypred))
# %%
import matplotlib.pyplot as plt
X_r = pca.fit(final_embedding_test).transform(final_embedding_test)
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
np.array(ppl_diff_per_data_point_train)
# %%
labels_sorted = pd.Series(np.array(probs_sorted)[:, :, 0][:, 0], dtype=int)
# %%
probs_df = pd.DataFrame(np.array(ppl_diff_per_data_point_train), columns=words[:-1])
df3 = pd.DataFrame(columns=words[:-1])
for func in [np.mean, np.min, np.max]:
    for label in [0, 1]:
        l = []
        for i in range(len(words[:-1])):
            tmp_df = pd.Series(probs_df.iloc[:, i])
            l.append(func(tmp_df[labels_sorted == label]))
            print(l)
        df3.loc[len(df3)] = l
        l = []
        for i in range(len(words[:-1])):
            tmp_df = probs_df.iloc[:, i]
            l.append(-np.log(func(tmp_df[labels_sorted == label])))
        df3.loc[len(df3)] = l
    l = []
    for i in range(len(words[:-1])):
        tmp_df = probs_df.iloc[:, i]
        l.append(func(tmp_df))
    df3.loc[len(df3)] = l
    l = []
    for i in range(len(words[:-1])):
        tmp_df = probs_df.iloc[:, i]
        l.append(-np.log(func(tmp_df)))
    df3.loc[len(df3)] = l
    #df.loc[len(df)] = (df.loc[len(df) - 1] - df.loc[len(df) - 2]) / df.loc[len(df) - 2]
    if func == np.mean:
        df3.loc[len(df3)] = (df3.loc[1] - df3.loc[5]) / df3.loc[5]
        df3.loc[len(df3)] = (df3.loc[3] - df3.loc[5]) / df3.loc[5]
df3 = df3.rename(index={
    0: "mean_0", 1: "mean_0_log", 
    2: "mean_1", 3: "mean_1_log", 
    4: "mean_0_1", 5: "mean_0_1_log", 
    6: "percent diff mean_0_log and mean_0_1_log",
    7: "percent diff mean_1_log and mean_0_1_log",
    8: "min_0", 9: "min_0_log", 
    10: "min_1", 11: "min_1_log", 
    12: "min_0_1", 13: "min_0_1_log", 
    14: "max_0", 15: "max_0_log", 
    16: "max_1", 17: "max_1_log", 
    18: "max_0_1", 19: "max_0_1_log"
})
df3
# %%
with open('perplexity.csv', 'w') as f:
    df3.to_csv(f, index=True, header=True, decimal='.', sep=',', float_format='%.15f')
# %%
print(probs_sorted)
# %%
train_df = load_file("TaskA_train.csv")
# %%
test_df = load_file("TaskA_test.csv")
# %%
probas = np.array(probs_train)[:, :, 2]
# %%
df_proba = pd.DataFrame(probas, columns=words)
# %%
final_df_train = pd.concat([train_df, df_proba], axis=1)
# %%
with open('final_df_train.csv', 'w') as f:
    final_df_train.to_csv(f)
# %%
probas = np.array(probs_test)[:, :, 2]
# %%
df_proba = pd.DataFrame(probas, columns=words)
df_label = pd.DataFrame(round2(ypred).astype(int), columns=["pred_label"])
# %%
final_df_test = pd.concat([test_df, df_proba, df_label], axis=1)
# %%
with open('final_df_test.csv', 'w') as f:
    final_df_test.to_csv(f)
# %%
x_train = np.loadtxt('x_train.txt')
x_test = np.loadtxt('x_test.txt')
y_train = np.loadtxt('y_train.txt')
y_test = np.loadtxt('y_test.txt')
# %%
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_train[0].shape)
# %%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

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
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
x_train = np.array(x_train, dtype=float)
# %%
dataset = TensorDataset(Tensor(x_train), Tensor(y_train))
print(dataset[0])
# %%
trainloader = DataLoader(dataset, batch_size=10, shuffle=True)
print(next(iter(trainloader)))
# %%
torch.manual_seed(42)
mlp = MLP()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
# %%
for epoch in range(0, 100): # 5 epochs at maximum
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

# Process is complete.
print('Training process has finished.')
# %%
x_test = np.array(x_test, dtype=float)
dataset_test = TensorDataset(Tensor(x_test), Tensor(y_test))
print(dataset[0])
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
print(f1_score(y_test, (round2(ypred)), average='macro'))
# %%
print(pred)
print(y_test)
# %%
print(f1_score(y_test, pred, average='macro'))
# %%
