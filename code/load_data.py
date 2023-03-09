import pandas as pd
from datasets import Dataset

def load_file(file="TaskA_train.csv"):
    df = pd.read_csv(file)[["topic", "Premise", "Conclusion", "Validity"]]
    df["text"] = (
        "<_t> " + df["topic"] + "." + 
        " <_p> " + df["Premise"].apply(lambda x: x[:-1]) + 
        ". <_c> " + df["Conclusion"]
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