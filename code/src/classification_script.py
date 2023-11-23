import transformers
import torch
import json
import configparser
import numpy as np
import lightgbm as lgb
import pandas as pd
from pathlib import Path
from datasets import DatasetDict
from model_attributes import ModelAttributes
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold
from concurrent.futures import ProcessPoolExecutor
from classifier import Classifier
from load_data import load_all_datasets
from transformers import (
    DataCollatorForLanguageModeling,
    IntervalStrategy, 
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertForSequenceClassification,
    set_seed
)
from utils import (
    compute_metrics,
    preprocess_logits_for_metrics,
    lgb_f1_score
)
DEVICE = "cuda:0" 


def mean_pooling(model_output):
    size = model_output.shape[1]
    sum_embeddings = torch.sum(model_output, 1)
    pooled_embeddings = sum_embeddings / size
    return pooled_embeddings


def extract_bert_embeddings(dataset, tokenizer, model):
    model = model.to("cpu")
    with ProcessPoolExecutor(max_workers=1) as executor:
        args = ((dataset, tokenizer, model, a) for a in range(len(dataset["text"])))
        sentence_embeddings = list(tqdm(executor.map(extract_bert_embeddings_parallel, args), total=len(dataset["text"]), desc="Calculating BERT Embeddings"))
    sentence_embeddings = np.stack(sentence_embeddings, axis=0)
    return sentence_embeddings


def extract_bert_embeddings_parallel(args):
    dataset, tokenizer, model, a = args
    input = tokenizer(dataset["text"][a], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**input)
    embedding = mean_pooling(output["hidden_states"][-1]) # mean pooling of all token embeddings
    #embedding = output["hidden_states"][-1][:,0,:] # CLS token embedding   [:,0,:] = (Batch_size, Sequence_length, Hidden_size)
    return  embedding


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

            probs_each = []
            checked_words = [] # depending on the model, some words appear in different cases
            # append the probability of each linking word for the current sentence to a list
            for i, masked_word in enumerate(masked_words_id):
                current_word = tokenizer.decode(masked_word).lower()
                if current_word in words and current_word not in checked_words:
                    checked_words.append(current_word)
                    probs_each.append((label, current_word, torch.nn.functional.softmax(masked_words_name, dim=-1)[i].item()))

            sentence_probs.append(probs_each)
        
        # only take the maximum of all sentences in the final list
        probs.append(max(sentence_probs, key=lambda x: x[2]))
    
    # sort probabilities with the word mapping order to always keep the same ordering
    probs_sorted = [sorted(x, key=lambda x: word_mapping[x[1].lower()]) for x in probs]

    # build the final embedding as list of (label, embedding vector) tuples
    embedding = [(np.int_(data_point[0,0]), np.float_(data_point[:, 2])) for data_point in np.array(probs_sorted)]

    return embedding


def calculate_embedding_vector_stance(dataset, tokenizer, model, words, word_mapping, model_attributes):
    with ProcessPoolExecutor(max_workers=4) as executor:
        args = ((dataset, tokenizer, model, words, model_attributes, a) for a in range(len(dataset["text"])))
        probs = list(tqdm(executor.map(calculate_embedding_vector_stance_parallel, args), total=len(dataset["text"])))

    # sort probabilities with the word mapping order to always keep the same ordering
    probs_sorted = [sorted(x, key=lambda x: word_mapping[x[1].lower()]) for x in probs]

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
    checked_words = [] # depending on the model, some words appear in different cases
    # append the probability of each linking word for the current sentence to a list
    for i, masked_word in enumerate(masked_words_id):
        current_word = tokenizer.decode(masked_word).lower()
        if current_word in words and current_word not in checked_words:
            checked_words.append(current_word)
            probs_each.append((label, current_word, torch.nn.functional.softmax(masked_words_name, dim=-1)[i].item()))
    return probs_each


# adapted from https://www.kaggle.com/code/ashishpatel26/feature-importance-of-lightgbm
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

    print("CV score: {:<8.5f}".format(f1_score(np.round(fold_pred), y_train_dft)))

    all_features = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)
    all_features.reset_index(inplace=True)
    important_features = all_features[all_features["importance"] > (all_features["importance"].mean()) // 4]["feature"]

    x_train_dft = x_train_dft[important_features]
    x_test_dft = x_test_dft[important_features]
    x_dev_dft = x_dev_dft[important_features]

    return x_train_dft, x_test_dft, x_dev_dft


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


def create_word_matrix_idx(words):
    # create the same array but for the words themselves (-> used as naming scheme for importance)
    word_stack1 = [w for _ in range(len(words)) for w in words]
    word_stack2 = [w for w in words for _ in range(len(words))]
    word_matrix = [f"{y}/{x}" for y, x in zip(word_stack2, word_stack1)]
    idx_words = word_matrix[0::(len(words)+1)]
    word_matrix_without_diag_flat = np.array([mat for mat in word_matrix if mat not in idx_words])
    return word_matrix_without_diag_flat


def create_matrix_embedding(embedding, words):
    # multiply the probability of each word with all the inverse probabilities of the other words and remove the diagonal
    # this yields precentage deviation instead of pure probabilities
    matrix = (embedding[...,None]*(1/embedding[:,None]))
    idx = np.where(~np.eye(matrix[0].shape[0], dtype=bool))
    matrix_embedding = np.array([mat[idx] for mat in matrix])

    word_matrix_without_diag_flat = create_word_matrix_idx(words)

    matrix_embedding_df = pd.DataFrame(matrix_embedding, columns=word_matrix_without_diag_flat).astype("float32")

    return matrix_embedding, matrix_embedding_df


def train_loop(save_i, model_attributes, load, save, seed, linking_word_index, all_bool_args):
    load_prev, train, train_on_premise_conclusion, train_on_arg_quality, pretrain, load_embedding, stance, baseline, calc_bert_embeddings, load_bert_embeddings = all_bool_args
    set_seed(seed)
    # save location for each loop
    save = save + f"/{save_i}"
    load = load + f"/0"
    Path(save).mkdir(parents=True, exist_ok=True)

    # define model, tokenizer and data collator
    model = AutoModelForMaskedLM.from_pretrained(model_attributes.model_checkpoint, output_hidden_states=True).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_attributes.model_checkpoint)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    premise_conclusion_data, pretraining_data, chatGPT_data, arg_quality_data = load_all_datasets()
    # split data
    #premise_conclusion_train, premise_conclusion_test, premise_conclusion_dev = train_test_dev_split(premise_conclusion_data)
    train_dataset, test_dataset, dev_dataset = premise_conclusion_data
    train_pre_dataset, test_pre_dataset = train_test_split_custom(pretraining_data)
    train_pre_chatgpt_dataset, test_pre_chatgpt_dataset = train_test_split_custom(chatGPT_data)
    train_arg_quality, test_arg_quality, dev_arg_quality = arg_quality_data
    # tokenize premise conclusion data
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    tokenized_dev = dev_dataset.map(tokenize_function, batched=True)
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

    if stance:
        tokenized_train = tokenized_train_arg_quality
        tokenized_test = tokenized_test_arg_quality
        tokenized_dev = tokenized_dev_arg_quality

        train_dataset = train_arg_quality
        test_dataset = test_arg_quality
        dev_dataset = dev_arg_quality

        eval_steps = 250 # this has to be changed in classifier file
    else:
        eval_steps = 30
    tokenized_complete = tokenized_complete = (tokenized_train, tokenized_dev, tokenized_test)

    # load already fine tuned model
    if load_prev or not train:
        model = AutoModelForMaskedLM.from_pretrained(load, output_hidden_states=True).to(DEVICE)
        trainer = transformers.Trainer(
            model=model,
        )
        trainer.save_model(output_dir=save)

    # training arguments for all masked LM     
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
        eval_steps=eval_steps,
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
    model_class = None
    # baseline direct bert classification
    if baseline:
        num_labels = 2
        # save instead of model_attributes.model_checkpoint
        model_class = BertForSequenceClassification.from_pretrained(model_attributes.model_checkpoint, num_labels=num_labels, output_hidden_states=True).to(DEVICE)
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
            eval_steps=eval_steps,
            save_total_limit=10,
            save_steps=eval_steps
        )
        trainer = transformers.Trainer(
            model=model_class,
            train_dataset=tokenized_train, # CHANGE
            eval_dataset=tokenized_dev, # CHANGE
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=training_args
        )
        trainer.train()
        trainer.model.eval()
        baseline_scores = trainer.evaluate(eval_dataset=tokenized_test)["eval_f1"] # CHANGE
        print("Baseline test f1: {}".format(baseline_scores))
        trainer.save_model(output_dir=save+"/LM_classification")

    model_class = model_class.to("cpu") if model_class is not None else model.to("cpu")
    model = model.to("cpu")
    
    with open("./src/linking_words.json", "r") as file:
        data = json.load(file) 

    linking_word_list = f"list{linking_word_index}"
    words = list(dict.fromkeys([x.lower() for x in data[linking_word_list]]))

    print(len(words))
    word_mapping = {word: i for i, word in enumerate(words)} # map words to numbers to sort them later

    if calc_bert_embeddings and load_bert_embeddings:
        bert_embeddings_train = np.load(load + '/bert_embeddings_train.npy', allow_pickle=True)
        bert_embeddings_test = np.load(load + '/bert_embeddings_test.npy', allow_pickle=True)
        bert_embeddings_dev = np.load(load + '/bert_embeddings_dev.npy', allow_pickle=True)
    elif calc_bert_embeddings:
        bert_embeddings_train = extract_bert_embeddings(train_arg_quality, tokenizer, model_class) #CHANGE
        bert_embeddings_test = extract_bert_embeddings(test_arg_quality, tokenizer, model_class) #CHANGE
        bert_embeddings_dev = extract_bert_embeddings(dev_arg_quality, tokenizer, model_class) #CHANGE
    np.save(save + '/bert_embeddings_train.npy', bert_embeddings_train, allow_pickle=True)
    np.save(save + '/bert_embeddings_test.npy', bert_embeddings_test, allow_pickle=True)
    np.save(save + '/bert_embeddings_dev.npy', bert_embeddings_dev, allow_pickle=True)

    if load_embedding:
        embedding_train = np.load(load + '/embedding_train.npy', allow_pickle=True)
        embedding_test = np.load(load + '/embedding_test.npy', allow_pickle=True)
        embedding_dev = np.load(load + '/embedding_dev.npy', allow_pickle=True)
    elif stance:
        embedding_train = calculate_embedding_vector_stance(train_dataset, tokenizer, model, words, word_mapping, model_attributes)
        embedding_test = calculate_embedding_vector_stance(test_dataset, tokenizer, model, words, word_mapping, model_attributes)
        embedding_dev = calculate_embedding_vector_stance(dev_dataset, tokenizer, model, words, word_mapping, model_attributes)
    else:
        embedding_train = calculate_embedding_vector(train_dataset, tokenizer, model, words, word_mapping)
        embedding_test = calculate_embedding_vector(test_dataset, tokenizer, model, words, word_mapping)
        embedding_dev = calculate_embedding_vector(dev_dataset, tokenizer, model, words, word_mapping)
    np.save(save + '/embedding_train.npy', embedding_train, allow_pickle=True)
    np.save(save + '/embedding_test.npy', embedding_test, allow_pickle=True)
    np.save(save + '/embedding_dev.npy', embedding_dev, allow_pickle=True)

    # Extract and transform embeddings to numpy arrays
    x_train_orig = np.array(list(zip(*embedding_train))[1])
    x_test_orig = np.array(list(zip(*embedding_test))[1])
    x_dev_orig = np.array(list(zip(*embedding_dev))[1])

    y_train = np.array(list(zip(*embedding_train))[0])
    y_test = np.array(list(zip(*embedding_test))[0])
    y_dev = np.array(list(zip(*embedding_dev))[0])

    # build DataFrames with words as columns for feature extraction
    x_train_df_orig = pd.DataFrame(x_train_orig, columns=words).astype("float32")
    x_test_df_orig = pd.DataFrame(x_test_orig, columns=words).astype("float32")
    x_dev_df_orig = pd.DataFrame(x_dev_orig, columns=words).astype("float32")

    y_train_df = pd.DataFrame(y_train).astype(int)
    y_test_df = pd.DataFrame(y_test).astype(int)
    y_dev_df = pd.DataFrame(y_dev).astype(int)

    bert_embeddings_train = bert_embeddings_train.squeeze()
    bert_embeddings_test = bert_embeddings_test.squeeze()
    bert_embeddings_dev = bert_embeddings_dev.squeeze()

    params_lgbm = {
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
    lgbm_classifier = Classifier(class_type="lgbm")

    params_nn = {
        "input_length" : len(words),
        "batch_size" : 32,
        "lr" : 2e-3, 
        "patience" : 10,
        "seed" : seed,
        "epochs" : 200
    }
    nn_classifier = Classifier(class_type="nn")

    ###############################################
    # NORMAL EMBEDDINGS WITHOUT FEATURE SELECTION #
    ###############################################

    f1_lgbm = lgbm_classifier.train_and_eval(params_lgbm, x_train_df_orig, y_train_df, x_test_df_orig, y_test_df, x_dev_df_orig, y_dev_df, num_round)
    f1_nn = nn_classifier.train_and_eval(params_nn, x_train_orig, y_train, x_dev_orig, y_dev, x_test_orig, y_test)
    f1_nn_complete = (nn_classifier.train_and_eval_nn_complete((x_train_orig, x_dev_orig, x_test_orig), save, tokenized_complete, seed, params_nn["input_length"], model_attributes))

    ############################################
    # NORMAL EMBEDDINGS WITH FEATURE SELECTION #
    ############################################

    x_train_df, x_test_df, x_dev_df = feature_extraction(x_train_df_orig, x_test_df_orig, x_dev_df_orig, y_train_df, params_lgbm)

    f1_lgbm_best = lgbm_classifier.train_and_eval(params_lgbm, x_train_df, y_train_df, x_test_df, y_test_df, x_dev_df, y_dev_df, num_round)
    f1_nn_best = nn_classifier.train_and_eval(params_nn, x_train_df.to_numpy(), y_train, x_dev_df.to_numpy(), y_dev, x_test_df.to_numpy(), y_test)
    f1_nn_best_complete = nn_classifier.train_and_eval_nn_complete((x_train_df.to_numpy(), x_dev_df.to_numpy(), x_test_df.to_numpy()), save, tokenized_complete, seed, params_nn["input_length"], model_attributes)

    ############################
    # CREATE MATRIX EMBEDDINGS #
    ############################

    x_train_matrix, x_train_matrix_df = create_matrix_embedding(x_train_orig, words)
    x_test_matrix, x_test_matrix_df = create_matrix_embedding(x_test_orig, words)
    x_dev_matrix, x_dev_matrix_df = create_matrix_embedding(x_dev_orig, words)

    params_nn = {
        "input_length" : x_train_matrix.shape[1],
        "batch_size" : 32,
        "lr" : 2e-3, 
        "patience" : 10,
        "seed" : seed,
        "epochs" : 200
    }

    ###############################################
    # MATRIX EMBEDDINGS WITHOUT FEATURE SELECTION #
    ###############################################

    f1_matrix_lgbm = lgbm_classifier.train_and_eval(params_lgbm, x_train_matrix_df, y_train_df, x_test_matrix_df, y_test_df, x_dev_matrix_df, y_dev_df, num_round)
    f1_matrix_nn = nn_classifier.train_and_eval(params_nn, x_train_matrix, y_train, x_dev_matrix, y_dev, x_test_matrix, y_test)
    f1_matrix_nn_complete = nn_classifier.train_and_eval_nn_complete((x_train_matrix, x_dev_matrix, x_test_matrix), save, tokenized_complete, seed, params_nn["input_length"], model_attributes)

    ############################################
    # MATRIX EMBEDDINGS WITH FEATURE SELECTION #
    ############################################

    x_train_matrix_df, x_test_matrix_df, x_dev_matrix_df = feature_extraction(x_train_matrix_df, x_test_matrix_df, x_dev_matrix_df, y_train_df, params_lgbm)

    params_nn = {
        "input_length" : x_train_matrix_df.to_numpy().shape[1],
        "batch_size" : 32,
        "lr" : 2e-3, 
        "patience" : 10,
        "seed" : seed,
        "epochs" : 200
    }

    f1_matrix_lgbm_best = lgbm_classifier.train_and_eval(params_lgbm, x_train_matrix_df, y_train_df, x_test_matrix_df, y_test_df, x_dev_matrix_df, y_dev_df, num_round)
    f1_matrix_nn_best = nn_classifier.train_and_eval(params_nn, x_train_matrix_df.to_numpy(), y_train, x_dev_matrix_df.to_numpy(), y_dev, x_test_matrix_df.to_numpy(), y_test)
    f1_matrix_nn_best_complete = nn_classifier.train_and_eval_nn_complete((x_train_matrix_df.to_numpy(), x_dev_matrix_df.to_numpy(), x_test_matrix_df.to_numpy()), save, tokenized_complete, seed, params_nn["input_length"], model_attributes)

    ##################################################
    # BERT EMBEDDINGS ONLY WITHOUT FEATURE SELECTION #
    ##################################################

    x_train_df = pd.DataFrame(bert_embeddings_train, columns=[f"{i}" for i in range(768)]).astype("float32")
    x_test_df = pd.DataFrame(bert_embeddings_test, columns=[f"{i}" for i in range(768)]).astype("float32")
    x_dev_df = pd.DataFrame(bert_embeddings_dev, columns=[f"{i}" for i in range(768)]).astype("float32")

    params_nn = {
        "input_length" : bert_embeddings_train.shape[1],
        "batch_size" : 32,
        "lr" : 2e-3, 
        "patience" : 10,
        "seed" : seed,
        "epochs" : 200
    }

    f1_bert_only_lgbm = lgbm_classifier.train_and_eval(params_lgbm, x_train_df, y_train_df, x_test_df, y_test_df, x_dev_df, y_dev_df, num_round)
    f1_bert_only_nn = nn_classifier.train_and_eval(params_nn, x_train_df.to_numpy(), y_train, x_dev_df.to_numpy(), y_dev, x_test_df.to_numpy(), y_test)

    ##################################################
    # COMBINE BERT EMBEDDINGS WITH NORMAL EMBEDDINGS #
    ##################################################

    x_train_bert = np.concatenate((bert_embeddings_train, x_train_orig), axis=1)
    x_test_bert = np.concatenate((bert_embeddings_test, x_test_orig), axis=1)
    x_dev_bert = np.concatenate((bert_embeddings_dev, x_dev_orig), axis=1)

    # just use numbers from 0 to 767 to name dimension of BERT embeddings
    x_train_bert_df = pd.DataFrame(x_train_bert, columns=([f"{i}" for i in range(768)] + words)).astype("float32")
    x_test_bert_df = pd.DataFrame(x_test_bert, columns=([f"{i}" for i in range(768)] + words)).astype("float32")
    x_dev_bert_df = pd.DataFrame(x_dev_bert, columns=([f"{i}" for i in range(768)] + words)).astype("float32")

    params_nn = {
        "input_length" : x_train_bert.shape[1],
        "batch_size" : 32,
        "lr" : 2e-3, 
        "patience" : 10, 
        "seed" : seed,
        "epochs" : 200
    }

    ####################################################################
    # BERT EMBEDDINGS WITH NORMAL EMBEDDINGS WITHOUT FEATURE SELECTION #
    ####################################################################

    f1_bert_normal_lgbm = lgbm_classifier.train_and_eval(params_lgbm, x_train_bert_df, y_train_df, x_test_bert_df, y_test_df, x_dev_bert_df, y_dev_df, num_round)
    f1_bert_normal_nn = nn_classifier.train_and_eval(params_nn, x_train_bert, y_train, x_dev_bert, y_dev, x_test_bert, y_test)

    #################################################################
    # BERT EMBEDDINGS WITH NORMAL EMBEDDINGS WITH FEATURE SELECTION #
    #################################################################

    x_train_df, x_test_df, x_dev_df = feature_extraction(x_train_bert_df, x_test_bert_df, x_dev_bert_df, y_train_df, params_lgbm)

    params_nn = {
        "input_length" : x_train_df.to_numpy().shape[1],
        "batch_size" : 32,
        "lr" : 2e-3, 
        "patience" : 10,
        "seed" : seed,
        "epochs" : 200
    }

    f1_bert_lgbm_normal_best = lgbm_classifier.train_and_eval(params_lgbm, x_train_df, y_train_df, x_test_df, y_test_df, x_dev_df, y_dev_df, num_round)
    f1_bert_nn_normal_best = nn_classifier.train_and_eval(params_nn, x_train_df.to_numpy(), y_train, x_dev_df.to_numpy(), y_dev, x_test_df.to_numpy(), y_test)

    ##################################################
    # COMBINE BERT EMBEDDINGS WITH MATRIX EMBEDDINGS #
    ##################################################

    x_train_matrix_bert = np.concatenate((bert_embeddings_train, x_train_matrix), axis=1)
    x_test_matrix_bert = np.concatenate((bert_embeddings_test, x_test_matrix), axis=1)
    x_dev_matrix_bert = np.concatenate((bert_embeddings_dev, x_dev_matrix), axis=1)

    word_matrix_without_diag_flat = create_word_matrix_idx(words)
    x_train_matrix_bert_df = pd.DataFrame(x_train_matrix_bert, columns=([f"{i}" for i in range(768)] + list(word_matrix_without_diag_flat))).astype("float32")
    x_test_matrix_bert_df = pd.DataFrame(x_test_matrix_bert, columns=([f"{i}" for i in range(768)] + list(word_matrix_without_diag_flat))).astype("float32")
    x_dev_matrix_bert_df = pd.DataFrame(x_dev_matrix_bert, columns=([f"{i}" for i in range(768)] + list(word_matrix_without_diag_flat))).astype("float32")

    params_nn = {
        "input_length" : x_train_matrix_bert.shape[1],
        "batch_size" : 32,
        "lr" : 2e-3, 
        "patience" : 10,
        "seed" : seed,
        "epochs" : 200
    }

    ####################################################################
    # BERT EMBEDDINGS WITH MATRIX EMBEDDINGS WITHOUT FEATURE SELECTION #
    ####################################################################
    
    f1_bert_matrix_lgbm = lgbm_classifier.train_and_eval(params_lgbm, x_train_matrix_bert_df, y_train_df, x_test_matrix_bert_df, y_test_df, x_dev_matrix_bert_df, y_dev_df, num_round)
    f1_bert_matrix_nn = nn_classifier.train_and_eval(params_nn, x_train_matrix_bert, y_train, x_dev_matrix_bert, y_dev, x_test_matrix_bert, y_test)
    f1_bert_matrix_nn_complete = nn_classifier.train_and_eval_nn_complete((x_train_matrix_bert, x_dev_matrix_bert, x_test_matrix_bert), save, tokenized_complete, seed, params_nn["input_length"], model_attributes)

    #################################################################
    # BERT EMBEDDINGS WITH MATRIX EMBEDDINGS WITH FEATURE SELECTION #
    #################################################################

    x_train_df, x_test_df, x_dev_df = feature_extraction(x_train_matrix_bert_df, x_test_matrix_bert_df, x_dev_matrix_bert_df, y_train_df, params_lgbm)

    params_nn = {
        "input_length" : x_train_df.to_numpy().shape[1],
        "batch_size" : 32,
        "lr" : 2e-3, 
        "patience" : 10,
        "seed" : seed,
        "epochs" : 200
    }

    f1_bert_matrix_lgbm_best = lgbm_classifier.train_and_eval(params_lgbm, x_train_df, y_train_df, x_test_df, y_test_df, x_dev_df, y_dev_df, num_round)
    f1_bert_matrix_nn_best = nn_classifier.train_and_eval(params_nn, x_train_df.to_numpy(), y_train, x_dev_df.to_numpy(), y_dev, x_test_df.to_numpy(), y_test)

    #################################################################

    final_scores = {
        "f1_lgbm": f1_lgbm,
        "f1_nn": f1_nn, 
        "f1_lgbm_best" : f1_lgbm_best,
        "f1_nn_best" : f1_nn_best,
        "f1_matrix_lgbm" : f1_matrix_lgbm,
        "f1_matrix_nn" : f1_matrix_nn,
        "f1_matrix_lgbm_best" : f1_matrix_lgbm_best,
        "f1_matrix_nn_best" : f1_matrix_nn_best,
        "f1_bert_only_lgbm" : f1_bert_only_lgbm,
        "f1_bert_only_nn" : f1_bert_only_nn,
        "f1_bert_normal_lgbm" : f1_bert_normal_lgbm,
        "f1_bert_normal_nn" : f1_bert_normal_nn,
        "f1_bert_lgbm_normal_best" : f1_bert_lgbm_normal_best,
        "f1_bert_nn_normal_best" : f1_bert_nn_normal_best,
        "f1_bert_matrix_lgbm" : f1_bert_matrix_lgbm,
        "f1_bert_matrix_nn" : f1_bert_matrix_nn,
        "f1_bert_matrix_lgbm_best" : f1_bert_matrix_lgbm_best,
        "f1_bert_matrix_nn_best" : f1_bert_matrix_nn_best,
        "baseline_scores" : baseline_scores,
        "f1_nn_complete" : f1_nn_complete,
        "f1_nn_best_complete" : f1_nn_best_complete,
        "f1_matrix_nn_complete" : f1_matrix_nn_complete,
        "f1_matrix_nn_best_complete" : f1_matrix_nn_best_complete,
        "f1_bert_matrix_nn_complete" : f1_bert_matrix_nn_complete      
    }

    print(final_scores)
    with open(save + "/scores.json", "w", encoding="utf8") as f:
        json.dump(final_scores, f)

    return final_scores


if __name__ == "__main__":
    model_attributes = ModelAttributes("bert-base-uncased")
    seed = 25
    linking_word_index = 1

    load = "./models/run_13_config_0110000110"
    Path(load).mkdir(parents=True, exist_ok=True)

    # load and read boolean arguments from the config file
    config = configparser.ConfigParser()
    config.read('./src/config.ini')
    all_bool_args = [True if config.get('BooleanArgs',i) == "True" else False for i in config['BooleanArgs']]
    
    save_suffix = "".join([str(int(i)) for i in all_bool_args])
    save = "./models/run_46_config_" + save_suffix
    print(save)

    Path(save).mkdir(parents=True, exist_ok=True)

    scores = []
    for i in range(3):
        seed += i
        scores.append(train_loop(i, model_attributes, load, save, seed, linking_word_index, all_bool_args))

    print(scores)
    print()

    # calculate average scores
    keys = scores[0].keys()
    values = np.array([list(dictx.values()) for dictx in scores])
    mean_values = list(np.mean(values, axis=0))
    average_scores = dict(zip(keys, mean_values))

    print(average_scores)

    with open(save + "/scores.json", "w", encoding="utf8") as f:
        json.dump(average_scores, f)
