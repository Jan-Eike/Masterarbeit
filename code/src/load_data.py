import pandas as pd
from datasets import Dataset
from os import listdir
from sklearn.model_selection import train_test_split

def load_file(file="data/TaskA_train.csv"):
    df = pd.read_csv(file)[["topic", "Premise", "Conclusion", "Validity"]]
    df["text"] = (
        "<_t> " + df["topic"] + ". " + 
        "<_p> " + df["Premise"].apply(lambda x: x[:-1]) + ". " +
        "<_c> " + df["Conclusion"]
    )

    df = df.drop(["topic", "Premise", "Conclusion"], axis=1)
    df = df.rename({"Validity": "labels"}, axis=1)
    df["labels"] = df.labels.apply(lambda x: 0 if x == -1 else x)

    return df


def load_data(files):
    datasets = []
    for file in files:
        df = load_file(file)
        ds = Dataset.from_pandas(df)
        datasets.append(ds)
    return datasets


def load_pretraining_data(location="data/argumentUnits"):
    files = listdir(location)
    train_pre = []
    for file in files:
        with open(location + "/" + file) as f:
            lines = f.readlines()
            lines = " ".join(lines)
        train_pre.append(lines)

    train_pre = [s.replace("\n", "") for s in train_pre]
    train_pre = [s.replace("\"", "") for s in train_pre]
    train_pre = [s.replace("\'", "") for s in train_pre]

    train_pre_df = pd.DataFrame({"text": train_pre})

    train_pre_dataset = Dataset.from_pandas(train_pre_df)

    return train_pre_dataset


def first_word_to_lower(line: str) -> str:
    return " ".join(i[0].lower()+i[1:] for i in line.split(" "))


def load_chatGPT_data(file="data/train_chatgpt.txt"):
    with open(file) as f:
        train_pre_chatgpt = f.readlines()
    
    train_pre_chatgpt = [s.replace("\n", "") for s in train_pre_chatgpt]
    train_pre_chatgpt = [s.split(": ")[-1] for s in train_pre_chatgpt]
    train_pre_chatgpt = [". ".join(s.split(". ")[:-1]) + ", <_c> " + first_word_to_lower(s.split(". ")[-1]) for s in train_pre_chatgpt]

    train_pre_chatgpt_df = pd.DataFrame({"text": train_pre_chatgpt})

    train_pre_chatgpt_dataset = Dataset.from_pandas(train_pre_chatgpt_df)

    return train_pre_chatgpt_dataset


def load_arg_quality(file="data/arg_quality_rank_30k.csv"):
    df = pd.read_csv(file)[["argument", "topic", "set", "stance_WA"]]
    df["text"] = (
        "<_t> " + df["topic"] + ". " + 
        "<_c> " + df["argument"]
    )

    df = df.drop(["topic", "argument"], axis=1)

    df = df.rename({"stance_WA": "labels"}, axis=1)
    # remove 0 labels
    #df = df[df["labels"] != 0]
    df["labels"] = df.labels.apply(lambda x: 0 if x == -1 else 1 if x == 1 else 2)
    
    train_df = Dataset.from_pandas(df[df["set"] == "train"].drop(["set"], axis=1))
    test_df = Dataset.from_pandas(df[df["set"] == "test"].drop(["set"], axis=1))
    val_df = Dataset.from_pandas(df[df["set"] == "dev"].drop(["set"], axis=1))

    return train_df, test_df, val_df


def load_all_datasets():
    # load premise conclusion data
    premise_conclusion_data = load_data(["data/TaskA_train.csv", "data/TaskA_test.csv", "data/TaskA_dev.csv"])

    # load pretraining data
    pretraining_data = load_pretraining_data()

    # load chatgpt data
    chatGPT_data = load_chatGPT_data()

    # load arg quality data
    arg_quality_data = load_arg_quality()

    return premise_conclusion_data, pretraining_data, chatGPT_data, arg_quality_data