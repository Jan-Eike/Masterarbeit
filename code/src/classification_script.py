import transformers
import torch
import json
import numpy as np
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import Dataset, DatasetDict
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from model_attributes import ModelAttributes
from tqdm import tqdm
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor
from transformers import (
    DataCollatorForLanguageModeling,
    IntervalStrategy, 
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertForSequenceClassification
)
from load_data import (
    load_data, 
    load_pretraining_data, 
    load_chatGPT_data,
    load_arg_quality
)
DEVICE = "cuda:0"


def load_all_datasets():
    # load premise conclusion data
    train_dataset, test_dataset, val_dataset = load_data(["data/TaskA_train.csv", "data/TaskA_test.csv", "data/TaskA_dev.csv"])
    train_dataset_for_complete = train_dataset.to_pandas()
    test_dataset_for_complete = test_dataset.to_pandas()
    val_dataset_for_complete = val_dataset.to_pandas()
    complete_dataset = Dataset.from_pandas(pd.concat([train_dataset_for_complete, test_dataset_for_complete, val_dataset_for_complete], ignore_index=True))

    premise_conclusion_data = complete_dataset

    # load pretraining data
    pretraining_data = load_pretraining_data()

    # load chatgpt data
    chatGPT_data = load_chatGPT_data()

    # load arg quality data
    train_arg_quality, test_arg_quality, val_arg_quality = load_arg_quality()
    train_arg_quality_complete = train_arg_quality.to_pandas()
    test_arg_quality_complete = test_arg_quality.to_pandas()
    val_arg_quality_complete = val_arg_quality.to_pandas()
    complete_arg_quality = Dataset.from_pandas(pd.concat([train_arg_quality_complete, test_arg_quality_complete, val_arg_quality_complete], ignore_index=True))

    arg_quality_data = complete_arg_quality

    return premise_conclusion_data, pretraining_data, chatGPT_data, arg_quality_data


def compute_metrics(p):    
    preds, labels = p
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"f1": f1} 


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output["hidden_states"][0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def extract_bert_embeddings(dataset, tokenizer, model):
    sentence_embeddings = []
    for a in tqdm(range(len(dataset["text"])), desc="Calculating BERT Embeddings"):
        input = tokenizer(dataset["text"][a], padding=True, truncation=True, max_length=512, return_tensors="pt")
        model = model.to("cpu")
        with torch.no_grad():
            output = model(**input)
        sentence_embeddings.append(mean_pooling(output, input['attention_mask']))
    sentence_embeddings = np.stack(sentence_embeddings, axis=0)
    return sentence_embeddings


def calculate_embedding_vector(dataset, tokenizer, model, words, word_mapping):
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


def calculate_embedding_vector_stance(dataset, tokenizer, model, words, word_mapping, model_attributes):
    with ProcessPoolExecutor(max_workers=4) as executor:
        args = ((dataset, tokenizer, model, words, model_attributes, a) for a in range(len(dataset["text"])))
        probs = list(tqdm(executor.map(calculate_embedding_vector_stance_parallel, args), total=len(dataset["text"])))

    print(dataset["text"][:5])
    print(probs[:5])

    # sort probabilities with the word mapping order to always keep the same ordering
    probs_sorted = [sorted(x, key=lambda x: word_mapping[x[1]]) for x in probs]

    # build the final embedding as list of (label, embedding vector) tuples
    embedding = [(np.int_(data_point[0,0]), np.float_(data_point[:, 2])) for data_point in np.array(probs_sorted)]

    return embedding


def calculate_embedding_vector_stance_parallel(args):
    dataset, tokenizer, model, words, model_attributes, a = args
    sentences = dataset["text"][a].split("<_c>") # split conclusion and rest
    conclusion = "<_c> " + sentences[-1][1:] # add "<_c>" symbol back to the conslusion
    topic = "<_t>" + sentences[0]
    text = topic + "." + conclusion.replace("<_c>", "<_c> " + model_attributes.mask)

    label = dataset["labels"][a]
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
    # Find the location of [MASK], extract its logits and sort
    mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
    mask_token_logits = token_logits[0, mask_token_index, :][0,1]
    masked_words_name = torch.sort(mask_token_logits, descending=True)[0]
    masked_words_id = torch.sort(mask_token_logits, descending=True)[1]

    probs_each = []
    # append the probability of each linking word for the current sentence to a list
    for i, masked_word in enumerate(masked_words_id):
        if tokenizer.decode(masked_word).lower() in words:
            probs_each.append((label, tokenizer.decode(masked_word), torch.nn.functional.softmax(masked_words_name, dim=-1)[i].item()))

    return probs_each


def feature_extraction(x_train_dft, x_test_dft, x_dev_dft, y_train_dft, params):
    nfolds = 5
    nrepeats = 2 
    folds = RepeatedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=11)
    fold_pred = np.zeros(len(x_train_dft))
    feature_importance_df = pd.DataFrame()
    lgb_preds = np.zeros(len(x_dev_dft))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train_dft.values, y_train_dft.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(x_train_dft.iloc[trn_idx], label=y_train_dft.iloc[trn_idx]) #categorical_feature=categorical_feats
        val_data = lgb.Dataset(x_train_dft.iloc[val_idx], label=y_train_dft.iloc[val_idx]) #categorical_feature=categorical_feats

        iteration = 10000
        lgb_m = lgb.train(params, trn_data, iteration, valid_sets = [trn_data, val_data], callbacks=[lgb.early_stopping(stopping_rounds=100)], feval=lgb_f1_score)
        fold_pred[val_idx] = lgb_m.predict(x_train_dft.iloc[val_idx], num_iteration=lgb_m.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = x_train_dft.columns
        fold_importance_df["importance"] = lgb_m.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        lgb_preds += lgb_m.predict(x_dev_dft, num_iteration=lgb_m.best_iteration) / (nfolds*nrepeats)

    print("CV score: {:<8.5f}".format(f1_score(np.round(fold_pred), y_train_dft, average="macro")))

    all_features = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)
    all_features.reset_index(inplace=True)
    important_features = all_features[all_features["importance"] > (all_features["importance"].mean()) // 4]["feature"]
    print(all_features)


    # Check feature correlation 
    #df = x_train_dft[important_features]
    #corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    #upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than 0.95
    #high_cor = [column for column in upper.columns if any(upper[column] > 0.95)]
    #print(len(high_cor))
    #print(high_cor)

    #features = [i for i in important_features if i not in high_cor]
    features = important_features
    #features = [i for i in important_features]
    #print(len(features))
    #print(features)

    print(y_train_dft)
    print(x_train_dft)
    x_train_dft = x_train_dft[features]
    print(x_train_dft)
    x_test_dft = x_test_dft[features]
    x_dev_dft = x_dev_dft[features]

    return x_train_dft, x_test_dft, x_dev_dft


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities, also not sure about just rounding...
    return 'f1', f1_score(y_true, y_hat, average='macro'), True


def train_test_split_custom(data, train_size=0.8):
    train_test = data.train_test_split(shuffle=True, seed=200, test_size=1-train_size)
    return train_test['train'], train_test['test']


def train_test_dev_split(data, train_size=0.8):
    train_devtest = data.train_test_split(shuffle=True, seed=200, test_size=1-train_size)
    dev_test = train_devtest['test'].train_test_split(shuffle=True, seed=200, test_size=0.5)
    train_dev_test = DatasetDict({
        'train': train_devtest['train'],
        'test': dev_test['test'],
        'dev': dev_test['train']})
    return train_dev_test['train'], train_dev_test['test'], train_dev_test['dev']


def train_loop(save_i, model_attributes, load, save, all_bool_args):
    load_prev, train, train_on_premise_conclusion, pretrain, train_on_arg_quality, load_embedding, stance, pca, baseline, calc_bert_embeddings, load_bert_embeddings = all_bool_args
    # save location for each loop
    save = save + f"/{save_i}"
    load = load + f"/{save_i}"
    Path(save).mkdir(parents=True, exist_ok=True)

    # define model, tokenizer and data collator
    model = AutoModelForMaskedLM.from_pretrained(model_attributes.model_checkpoint, output_hidden_states=True).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_attributes.model_checkpoint)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    premise_conclusion_data, pretraining_data, chatGPT_data, arg_quality_data = load_all_datasets()
    # split data
    premise_conclusion_train, premise_conclusion_test, premise_conclusion_dev = train_test_dev_split(premise_conclusion_data)
    train_pre_dataset, test_pre_dataset = train_test_split_custom(pretraining_data)
    train_pre_chatgpt_dataset, test_pre_chatgpt_dataset = train_test_split_custom(chatGPT_data)
    train_arg_quality, test_arg_quality, dev_arg_quality = train_test_dev_split(arg_quality_data)
    # tokenize premise conclusion data
    tokenized_train = premise_conclusion_train.map(tokenize_function, batched=True)
    tokenized_test = premise_conclusion_test.map(tokenize_function, batched=True)
    tokenized_dev = premise_conclusion_dev.map(tokenize_function, batched=True)
    # tokenize pretraining data
    tokenized_train_pre = train_pre_dataset.map(tokenize_function, batched=True)
    tokenized_test_pre = test_pre_dataset.map(tokenize_function, batched=True)
    # tokenize chatgpt data
    tokenized_train_pre_chatgpt = train_pre_chatgpt_dataset.map(tokenize_function, batched=True)
    tokenized_test_pre_chatgpt = test_pre_chatgpt_dataset.map(tokenize_function, batched=True)
    # tokenize arg quality data
    tokenized_train_arg_quality = train_arg_quality.map(tokenize_function, batched=True)
    tokenized_test_arg_quality = test_arg_quality.map(tokenize_function, batched=True)
    tokenized_dev_arg_quality = dev_arg_quality.map(tokenize_function, batched=True)

    print(tokenized_train["input_ids"])

    # load already fine tuned model
    if load_prev or not train:
        model = AutoModelForMaskedLM.from_pretrained(load, output_hidden_states=True).to(DEVICE)

    # training arguments for all masked LM 
    training_args = transformers.TrainingArguments(
        output_dir=save,
        overwrite_output_dir=False,
        per_device_train_batch_size=3, # change if not enough cuda memory
        per_device_eval_batch_size=3, # change if not enough cuda memory
        num_train_epochs=3,
        warmup_steps=100,
        load_best_model_at_end=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=1500,
        save_total_limit=10,
        metric_for_best_model="eval_loss",
        eval_accumulation_steps=10,
        save_steps=1500
    )
    
    training_args = transformers.TrainingArguments(
        output_dir=save,
        overwrite_output_dir=False,
        per_device_train_batch_size=4, # change if not enough cuda memory
        per_device_eval_batch_size=4, # change if not enough cuda memory
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        warmup_steps=100,
        load_best_model_at_end=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=150,
        save_total_limit=10,
        metric_for_best_model="eval_loss",
        eval_accumulation_steps=10,
        save_steps=1500
    )

    saved = False
    from_dir = save if saved else load if load_prev else model_attributes.model_checkpoint
    # consecutive training loops on different datasets
    if train and pretrain:
        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_train_pre,
            eval_dataset=tokenized_test_pre,
            data_collator=data_collator,
            args=training_args
        )
        trainer.train()
        trainer.model.eval()
        trainer.save_model(output_dir=save)
        saved = True

    from_dir = save if saved else load if load_prev else model_attributes.model_checkpoint
    if train and pretrain:
        model = AutoModelForMaskedLM.from_pretrained(from_dir, output_hidden_states=True).to(DEVICE) # don't know if necessary
        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_train_pre_chatgpt,
            eval_dataset=tokenized_test_pre_chatgpt,
            data_collator=data_collator,
            args=training_args
        )
        trainer.train()
        trainer.model.eval()
        trainer.save_model(output_dir=save)
        saved = True

    from_dir = save if saved else load if load_prev else model_attributes.model_checkpoint
    if train and train_on_premise_conclusion:
        model = AutoModelForMaskedLM.from_pretrained(from_dir, output_hidden_states=True).to(DEVICE) # don't know if necessary
        print(model.hidden_size)
        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            data_collator=data_collator,
            args=training_args
        )
        trainer.train()
        trainer.model.eval()
        trainer.save_model(output_dir=save)
        saved = True

    from_dir = save if saved else load if load_prev else model_attributes.model_checkpoint
    print(from_dir)
    if train and train_on_arg_quality:
        model = AutoModelForMaskedLM.from_pretrained(from_dir, output_hidden_states=True).to(DEVICE) # don't know if necessary
        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_train_arg_quality,
            eval_dataset=tokenized_dev_arg_quality,
            data_collator=data_collator,
            args=training_args
        )
        trainer.train()
        trainer.model.eval()
        trainer.save_model(output_dir=save)
        saved = True
    

    baseline_scores = 0
    # baseline direct bert classification
    if baseline:
        num_labels = 2
        model_class = BertForSequenceClassification.from_pretrained(save, num_labels=num_labels).to(DEVICE)
        training_args = transformers.TrainingArguments(
            output_dir=save,
            overwrite_output_dir=False,
            per_device_train_batch_size=4, # change if not enough cuda memory
            per_device_eval_batch_size=4, # change if not enough cuda memory
            gradient_accumulation_steps=16,
            num_train_epochs=3,
            warmup_steps=100,
            load_best_model_at_end=True,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=150,
            save_total_limit=10,
            save_steps=150
        )
        trainer = transformers.Trainer(
            model=model_class,
            train_dataset=tokenized_train_arg_quality,
            eval_dataset=tokenized_dev_arg_quality,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=training_args
        )
        trainer.train()
        trainer.model.eval()
        baseline_scores = trainer.evaluate(eval_dataset=tokenized_test_arg_quality)
        print(baseline_scores)
        trainer.save_model(output_dir=save+"/LM_classification")

    model = AutoModelForMaskedLM.from_pretrained(save+"/LM_classification", output_hidden_states=True).to(DEVICE)
    model_class = model
    model_class = model_class.to("cpu")
    model = model.to("cpu")
    # example masking
    text = f"This is a great {model_attributes.mask}."
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
    mask_token_index = torch.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)
    mask_token_logits = token_logits[0, mask_token_index, :][0,1]
    top_5_tokens = torch.argsort(-mask_token_logits)[:10].tolist()

    for token in top_5_tokens:
        print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode(token)[model_attributes.start:])}")

    #words = ["t"herefore", "consequently", "hence", "thus", "so", "nevertheless", "however", "yet", "anyway", "although"]
    #words = ["t"he", "a", "hence", "thus", "and", "this", "he", "she", "it", "yet", "be", "to", "that", "for", "as", "have", "but", "by", "from", "say", "his", "her", "its", "with", "will", "can", "of", "in", "i", "not"]

    words = list(dict.fromkeys([
        "the",
        "be",
        "and",
        "a",
        "of",
        "to",
        "in",
        "i",
        "you",
        "it",
        "have",
        "to",
        "that",
        "for",
        "do",
        "he",
        "with",
        "on",
        "this",
        "therefore",
        "we",
        "that",
        "not",
        "but",
        "they",
        "say",
        "at",
        "what",
        "his",
        "from",
        "go",
        "or",
        "by",
        "get",
        "she",
        "my",
        "can",
        "as",
        "know",
        "if",
        "me",
        "your",
        "all",
        "who",
        "about",
        "their",
        "will",
        "so",
        "would",
        "make",
        "just",
        "up",
        "think",
        "time",
        "there",
        "see",
        "her",
        "as",
        "out",
        "one",
        "come",
        "people",
        "take",
        "year",
        "him",
        "them",
        "some",
        "want",
        "how",
        "when",
        "which",
        "now",
        "like",
        "other",
        "could",
        "our",
        "into",
        "here",
        "then",
        "than",
        "look",
        "way",
        "more",
        "these",
        "no",
        "thing",
        "well",
        "because",
        "also",
        "two",
        "use",
        "tell",
        "good",
        "first",
        "man",
        "day",
        "find",
        "give",
        "more",
        "new",
    ]))
    """
    words = list(dict.fromkeys([
        "because",
        "although",
        "therefore",
        "but",
        "still",
        "whereas",
        "while",
        "however",
        "since",
        "therefore",
        "as",
        "for",
        "consequently",
        "hence",
        "thus",
        "so",
        "nevertheless",
        "yet",
        "anyway",
        "still"
    ]))
    """
    #words = ["he", "her", "she", "can", "and", "yet", "as", "it", "that", "will"]
    #words = ["the", "a", "hence", "thus", "and", "this", "he", "she", "it", "yet"]
    word_mapping = {word: i for i, word in enumerate(words)} # map words to numbers to sort them later

    if calc_bert_embeddings and load_bert_embeddings:
        bert_embeddings_train = np.load(load + '/bert_embeddings_train.npy', allow_pickle=True)
        bert_embeddings_test = np.load(load + '/bert_embeddings_test.npy', allow_pickle=True)
        bert_embeddings_dev = np.load(load + '/bert_embeddings_dev.npy', allow_pickle=True)
        np.save(save + '/bert_embeddings_train.npy', bert_embeddings_train, allow_pickle=True)
        np.save(save + '/bert_embeddings_test.npy', bert_embeddings_test, allow_pickle=True)
        np.save(save + '/bert_embeddings_dev.npy', bert_embeddings_dev, allow_pickle=True)
    elif calc_bert_embeddings:
        bert_embeddings_train = extract_bert_embeddings(train_arg_quality, tokenizer, model_class)
        bert_embeddings_test = extract_bert_embeddings(test_arg_quality, tokenizer, model_class)
        bert_embeddings_dev = extract_bert_embeddings(dev_arg_quality, tokenizer, model_class)
        print(bert_embeddings_train.shape)
        print(bert_embeddings_test.shape)
        print(bert_embeddings_dev.shape)
        np.save(save + '/bert_embeddings_train.npy', bert_embeddings_train, allow_pickle=True)
        np.save(save + '/bert_embeddings_test.npy', bert_embeddings_test, allow_pickle=True)
        np.save(save + '/bert_embeddings_dev.npy', bert_embeddings_dev, allow_pickle=True)

    if load_embedding:
        embedding_train = np.load(load + '/embedding_train.npy', allow_pickle=True)
        embedding_test = np.load(load + '/embedding_test.npy', allow_pickle=True)
        embedding_dev = np.load(load + '/embedding_dev.npy', allow_pickle=True)
    elif stance:
        embedding_train = calculate_embedding_vector_stance(train_arg_quality, tokenizer, model, words, word_mapping, model_attributes)
        embedding_test = calculate_embedding_vector_stance(test_arg_quality, tokenizer, model, words, word_mapping, model_attributes)
        embedding_dev = calculate_embedding_vector_stance(dev_arg_quality, tokenizer, model, words, word_mapping, model_attributes)
    else:
        embedding_train = calculate_embedding_vector(premise_conclusion_train, tokenizer, model, words, word_mapping)
        embedding_test = calculate_embedding_vector(premise_conclusion_test, tokenizer, model, words, word_mapping)
        embedding_dev = calculate_embedding_vector(premise_conclusion_dev, tokenizer, model, words, word_mapping)
    np.save(save + '/embedding_train.npy', embedding_train, allow_pickle=True)
    np.save(save + '/embedding_test.npy', embedding_test, allow_pickle=True)
    np.save(save + '/embedding_dev.npy', embedding_dev, allow_pickle=True)

    # standardise embeddings and transform to numpy arrays
    x_train = np.array(list(zip(*embedding_train))[1])
    #x_train = StandardScaler().fit_transform(x_train)
    #x_train = np.array([StandardScaler().fit_transform(np.array(sample).reshape(-1,1)) for sample in x_train], dtype=object).reshape(-1,len(words))

    x_test = np.array(list(zip(*embedding_test))[1])
    #x_test = StandardScaler().fit_transform(x_test)
    #x_test = np.array([StandardScaler().fit_transform(np.array(sample).reshape(-1,1)) for sample in x_test], dtype=object).reshape(-1,len(words))

    x_dev = np.array(list(zip(*embedding_dev))[1])

    y_train = np.array(list(zip(*embedding_train))[0])

    y_test = np.array(list(zip(*embedding_test))[0])

    y_dev = np.array(list(zip(*embedding_dev))[0])

    x_train_dft = pd.DataFrame(x_train, columns=words).astype(float)
    x_test_dft = pd.DataFrame(x_test, columns=words).astype(float)
    x_dev_dft = pd.DataFrame(x_dev, columns=words).astype(float)

    y_train_dft = pd.DataFrame(y_train).astype(int)
    y_test_dft = pd.DataFrame(y_test).astype(int)
    y_dev_dft = pd.DataFrame(y_dev).astype(int)

    print(y_train.shape)
    print(x_train.shape)

    # lgbm classification
    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test)
    validation_data = lgb.Dataset(x_dev, label=y_dev)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'f1'},
        'num_leaves': 21,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'first_metric_only': True,
        'device': 'gpu'
    }
    num_round = 10000
    iteration = 10000

    x_train_dft, x_test_dft, x_dev_dft = feature_extraction(x_train_dft, x_test_dft, x_dev_dft, y_train_dft, params)

    lgb_modelt = lgb.train(params, lgb.Dataset(x_train_dft, label=y_train_dft), num_round, valid_sets=[lgb.Dataset(x_dev_dft, label=y_dev_dft)], callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(100)], feval=lgb_f1_score)

    ypred = lgb_modelt.predict(x_test_dft, num_iteration=lgb_modelt.best_iteration)
    f1_lgbm_best = f1_score(y_test_dft, (np.round(ypred)), average='macro')
    print(f"f1_lgbm: {f1_lgbm_best}")

    bst = lgb.train(params, train_data, num_round, valid_sets=[validation_data], callbacks=[lgb.early_stopping(stopping_rounds=40)], feval=lgb_f1_score)

    print(bst.feature_importance())
    imp = [(x, y) for y, x in sorted(zip(bst.feature_importance(), words), key=lambda x: x[0], reverse=True)]
    for t in imp:
        print(f"{t[0]}: {t[1]}")

    ypred = bst.predict(x_test)
    f1_lgbm = f1_score(y_test, (np.round(ypred)), average='macro')
    print(f1_lgbm)

    # svm classification
    clf = svm.SVC(kernel="poly")
    clf.fit(x_train, y_train)

    ypred_svm = clf.predict(x_test)
    print(f1_score(y_test, ypred_svm, average='macro'))

    clf = svm.SVC(kernel="rbf")
    clf.fit(x_train, y_train)

    ypred_svm = clf.predict(x_test)
    f1_svm = f1_score(y_test, ypred_svm, average='macro')
    print(f1_svm)
    # pca visualization
    if pca:
        pca = PCA(n_components=2)

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

    # simple neural net classification same as in BERT Classification Head but without Dropout
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(len(words), 2),
            )


        def forward(self, x):
            return self.layers(x)

    x_train = np.array(x_train, dtype=float)
    dataset = TensorDataset(Tensor(x_train), Tensor(y_train))
    trainloader = DataLoader(dataset, batch_size=10, shuffle=True)

    torch.manual_seed(42)
    mlp = MLP()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    for epoch in range(10):
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
            """
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0
            """
    print('Training process has finished.')

    x_test = np.array(x_test, dtype=float)
    dataset_test = TensorDataset(Tensor(x_test), Tensor(y_test))

    pred = []
    testloader = DataLoader(dataset_test, batch_size=10, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):

            inputs, targets = data
            targets = targets.long()

            outputs = mlp(inputs)

            _, predicted = torch.max(outputs, 1)
            pred.extend(predicted.tolist())

    f1_nn = f1_score(y_test, pred, average='macro')
    print(f1_nn)

    embedding_train_array = np.array(list(zip(*embedding_train))[1])
    #embedding_complete_array = np.array(list(zip(*embedding_complete))[1])
    #embedding_train_array = x_train_dft.to_numpy()

    # multiply the probability of each word with all the inverse probabilities of the other words and remove the diagonal
    # this yields precentage deviation instead of pure probabilities
    matrix = (embedding_train_array[...,None]*(1/embedding_train_array[:,None]))
    #matrix = (embedding_complete_array[...,None]*(1/embedding_complete_array[:,None]))
    idx = np.where(~np.eye(matrix[0].shape[0],dtype=bool))

    matrix_without_diag_flat_train = np.array([mat[idx] for mat in matrix])

    # create the same array but for the words themselves (-> used as naming scheme for importance)
    word_stack1 = [w for _ in range(len(words)) for w in words]
    word_stack2 = [w for w in words for _ in range(len(words))]
    word_matrix = [f"{y}/{x}" for y, x in zip(word_stack2, word_stack1)]
    idx_words = word_matrix[0::(len(words)+1)]
    word_matrix_without_diag_flat = np.array([mat for mat in word_matrix if mat not in idx_words])
    matrix_without_diag_flat_train_df = pd.DataFrame(matrix_without_diag_flat_train, columns=word_matrix_without_diag_flat).astype(float)
    x_train_dft = matrix_without_diag_flat_train_df

    embedding_test_array = np.array(list(zip(*embedding_test))[1])
    #embedding_test_array = x_test_dft.to_numpy()

    matrix = (embedding_test_array[...,None]*(1/embedding_test_array[:,None]))
    idx = np.where(~np.eye(matrix[0].shape[0],dtype=bool))

    matrix_without_diag_flat_test = np.array([mat[idx] for mat in matrix])

    matrix_without_diag_flat_test_df = pd.DataFrame(matrix_without_diag_flat_test, columns=word_matrix_without_diag_flat).astype(float)
    x_test_dft = matrix_without_diag_flat_test_df

    embedding_dev_array = np.array(list(zip(*embedding_dev))[1])
    #embedding_test_array = x_test_dft.to_numpy()

    matrix = (embedding_dev_array[...,None]*(1/embedding_dev_array[:,None]))
    idx = np.where(~np.eye(matrix[0].shape[0],dtype=bool))

    matrix_without_diag_flat_dev = np.array([mat[idx] for mat in matrix])

    matrix_without_diag_flat_dev_df = pd.DataFrame(matrix_without_diag_flat_dev, columns=word_matrix_without_diag_flat).astype(float)
    x_dev_dft = matrix_without_diag_flat_dev_df

    train_data = lgb.Dataset(matrix_without_diag_flat_train, label=y_train)
    validation_data = lgb.Dataset(matrix_without_diag_flat_dev, label=y_dev)
    test_data = lgb.Dataset(matrix_without_diag_flat_test, label=y_test)

    x_train_dft, x_test_dft, x_dev_dft = feature_extraction(x_train_dft, x_test_dft, x_dev_dft, y_train_dft, params)

    lgb_modelt = lgb.train(params, lgb.Dataset(x_train_dft, label=y_train_dft), iteration, valid_sets = [lgb.Dataset(x_train_dft, label=y_train_dft), lgb.Dataset(x_dev_dft, label=y_dev_dft)], verbose_eval=100, callbacks=[lgb.early_stopping(stopping_rounds=40)], feval=lgb_f1_score)

    ypred = lgb_modelt.predict(x_test_dft)
    f1_lgbm_best_matrix = f1_score(y_test_dft, (np.round(ypred)), average='macro')
    print(f"f1_lgbm_matrix: {f1_lgbm_best_matrix}")

    bst = lgb.train(params, train_data, num_round, valid_sets=[validation_data], callbacks=[lgb.early_stopping(stopping_rounds=40)], feval=lgb_f1_score)

    df_feature_importance2 = (
        pd.DataFrame({
            'feature': bst.feature_name(),
            'importance': bst.feature_importance(),
        })
        .sort_values('importance', ascending=False)
    )
    print(df_feature_importance2)

    ypred = bst.predict(matrix_without_diag_flat_test)
    f1_matrix_lgbm = f1_score(y_test, (np.round(ypred)), average='macro')

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(matrix_without_diag_flat_train.shape[1], 2),
            )


        def forward(self, x):
            return self.layers(x)

    x_train = matrix_without_diag_flat_train
    dataset = TensorDataset(Tensor(x_train), Tensor(y_train))
    trainloader = DataLoader(dataset, batch_size=10, shuffle=True)

    torch.manual_seed(42)
    mlp = MLP()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-5)

    for epoch in range(15):
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
            """
            if i % 50 == 49:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 50))
                current_loss = 0.0
            """
    print('Training process has finished.')

    x_test = matrix_without_diag_flat_test
    dataset_test = TensorDataset(Tensor(x_test), Tensor(y_test))

    pred = []
    testloader = DataLoader(dataset_test, batch_size=10, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):

            inputs, targets = data
            targets = targets.long()

            outputs = mlp(inputs)

            _, predicted = torch.max(outputs, 1)
            pred.extend(predicted.tolist())

    f1_matrix_nn = f1_score(y_test, pred, average='macro')

    combined_bert_embeddings_train = np.array(list(zip(*embedding_train))[1])
    bert_embeddings_train = bert_embeddings_train.reshape(bert_embeddings_train.shape[0], -1)
    print(bert_embeddings_train.shape, combined_bert_embeddings_train.shape)
    x_train = np.concatenate((bert_embeddings_train, combined_bert_embeddings_train), axis=1)

    combined_bert_embeddings_test = np.array(list(zip(*embedding_test))[1])
    bert_embeddings_test = bert_embeddings_test.reshape(bert_embeddings_test.shape[0], -1)
    x_test = np.concatenate((bert_embeddings_test, combined_bert_embeddings_test), axis=1)

    combined_bert_embeddings_dev = np.array(list(zip(*embedding_dev))[1])
    bert_embeddings_dev = bert_embeddings_dev.reshape(bert_embeddings_dev.shape[0], -1)
    x_dev = np.concatenate((bert_embeddings_dev, combined_bert_embeddings_dev), axis=1)

    y_train = np.array(list(zip(*embedding_train))[0])

    y_test = np.array(list(zip(*embedding_test))[0])

    y_dev = np.array(list(zip(*embedding_dev))[0])

    x_train_dft = pd.DataFrame(x_train, columns=([f"{i}" for i in range(768)] + words)).astype(float)
    x_test_dft = pd.DataFrame(x_test, columns=([f"{i}" for i in range(768)] + words)).astype(float)
    x_dev_dft = pd.DataFrame(x_dev, columns=([f"{i}" for i in range(768)] + words)).astype(float)

    y_train_dft = pd.DataFrame(y_train).astype(int)
    y_test_dft = pd.DataFrame(y_test).astype(int)
    y_dev_dft = pd.DataFrame(y_dev).astype(int)

    # lgbm classification
    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test)
    validation_data = lgb.Dataset(x_dev, label=y_dev)

    lgb_modelt = lgb.train(params, lgb.Dataset(x_train_dft, label=y_train_dft), iteration, valid_sets = [lgb.Dataset(x_train_dft, label=y_train_dft), lgb.Dataset(x_dev_dft, label=y_dev_dft)], verbose_eval=100, callbacks=[lgb.early_stopping(stopping_rounds=40)], feval=lgb_f1_score)

    ypred = lgb_modelt.predict(x_test_dft, num_iteration=lgb_modelt.best_iteration)
    f1_lgbm_combined = f1_score(y_test_dft, (np.round(ypred)), average='macro')
    print(f"f1_lgbm_WARNUNG: {f1_lgbm_combined}")

    x_train_dft, x_test_dft, x_dev_dft = feature_extraction(x_train_dft, x_test_dft, x_dev_dft, y_train_dft, params)

    lgb_modelt = lgb.train(params, lgb.Dataset(x_train_dft, label=y_train_dft), iteration, valid_sets = [lgb.Dataset(x_train_dft, label=y_train_dft), lgb.Dataset(x_dev_dft, label=y_dev_dft)], verbose_eval=100, callbacks=[lgb.early_stopping(stopping_rounds=40)], feval=lgb_f1_score)

    ypred = lgb_modelt.predict(x_test_dft, num_iteration=lgb_modelt.best_iteration)
    f1_lgbm_best_combined = f1_score(y_test_dft, (np.round(ypred)), average='macro')
    print(f"f1_lgbm: {f1_lgbm_best_combined}")

    print(matrix_without_diag_flat_train.shape)
    print(bert_embeddings_train.shape)
    x_train = np.concatenate((bert_embeddings_train, matrix_without_diag_flat_train), axis=1)
    x_test = np.concatenate((bert_embeddings_test, matrix_without_diag_flat_test), axis=1)
    x_dev = np.concatenate((bert_embeddings_dev, matrix_without_diag_flat_dev), axis=1)
    print(x_train.shape)
    print(matrix_without_diag_flat_train.shape)
    print(bert_embeddings_train.shape)
    print(word_matrix_without_diag_flat.shape)
    x_train_dft = pd.DataFrame(x_train, columns=([f"{i}" for i in range(768)] + list(word_matrix_without_diag_flat))).astype(float)
    x_test_dft = pd.DataFrame(x_test, columns=([f"{i}" for i in range(768)] + list(word_matrix_without_diag_flat))).astype(float)
    x_dev_dft = pd.DataFrame(x_dev, columns=([f"{i}" for i in range(768)] + list(word_matrix_without_diag_flat))).astype(float)

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test)
    validation_data = lgb.Dataset(x_dev, label=y_dev)

    lgb_modelt = lgb.train(params, lgb.Dataset(x_train_dft, label=y_train_dft), iteration, valid_sets = [lgb.Dataset(x_train_dft, label=y_train_dft), lgb.Dataset(x_dev_dft, label=y_dev_dft)], verbose_eval=100, callbacks=[lgb.early_stopping(stopping_rounds=40)], feval=lgb_f1_score)

    ypred = lgb_modelt.predict(x_test_dft, num_iteration=lgb_modelt.best_iteration)
    f1_lgbm_combined_matrix = f1_score(y_test_dft, (np.round(ypred)), average='macro')
    print(f"f1_lgbm_WARNUNG_matrix: {f1_lgbm_combined}")

    x_train_dft, x_test_dft, x_dev_dft = feature_extraction(x_train_dft, x_test_dft, x_dev_dft, y_train_dft, params)

    lgb_modelt = lgb.train(params, lgb.Dataset(x_train_dft, label=y_train_dft), iteration, valid_sets = [lgb.Dataset(x_train_dft, label=y_train_dft), lgb.Dataset(x_dev_dft, label=y_dev_dft)], verbose_eval=100, callbacks=[lgb.early_stopping(stopping_rounds=40)], feval=lgb_f1_score)

    ypred = lgb_modelt.predict(x_test_dft, num_iteration=lgb_modelt.best_iteration)
    f1_lgbm_best_combined_matrix = f1_score(y_test_dft, (np.round(ypred)), average='macro')
    print(f"f1_lgbm: {f1_lgbm_best_combined_matrix}")

    final_scores = {"f1_lgbm": f1_lgbm, "f1_svm": f1_svm, "f1_nn": f1_nn, "f1_matrix_lgbm": f1_matrix_lgbm, "f1_matrix_nn": f1_matrix_nn, "f1_lgbm_best_select": f1_lgbm_best, "f1_lgbm_best_matrix": f1_lgbm_best_matrix, "f1_lgbm_combined": f1_lgbm_combined, "f1_lgbm_best_combined": f1_lgbm_best_combined, "baseline_scores": baseline_scores, "f1_lgbm_combined_matrix": f1_lgbm_combined_matrix, "f1_lgbm_best_combined_matrix": f1_lgbm_best_combined_matrix}
    print(final_scores)
    with open(save + "/scores.json", "w", encoding="utf8") as f:
        json.dump(final_scores, f)

    return final_scores


if __name__ == "__main__":
    model_attributes = ModelAttributes("bert-base-uncased")

    load = "./models/loop16"
    Path(load).mkdir(parents=True, exist_ok=True)

    # change this for new models
    save = "./models/loop16"
    Path(save).mkdir(parents=True, exist_ok=True)

    load_prev = True # further train a previously trained model
    train = False # train model or use a saved one. If this is False, load_prev is always True
    train_on_premise_conclusion = False # train on premise conclusion dataset
    train_on_arg_quality = True # train on arg quality dataset
    pretrain = False # pretrain on chatgpt and argumentsUnits data
    load_embedding = False # load old embedding or calcualte a new one
    stance = True # stance classification or arg val
    pca = False # visualize with pca
    baseline = False # baseline classification
    calc_bert_embeddings = True # extract embeddings from bert
    load_bert_embeddings = False # load saved bert embeddings
    all_bool_args = load_prev, train, train_on_premise_conclusion, pretrain, train_on_arg_quality, load_embedding, stance, pca, baseline, calc_bert_embeddings, load_bert_embeddings

    scores = []
    for i in range(1):
        scores.append(train_loop(i, model_attributes, load, save, all_bool_args))

    print(scores)