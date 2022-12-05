import argparse
from pathlib import Path
import pandas as pd

def first_word_to_lower(line: str) -> str:
    return " ".join(i[0].lower()+i[1:] for i in line.split(" "))

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("file", type=Path, action="store", help="File to be prepared.")
    args = argument_parser.parse_args()
    df = pd.read_csv(str(args.file), encoding="utf8")
    df2 = pd.read_csv(str(args.file)[:-4]+"-without-labels.csv", encoding="utf8")
    df3 = df.merge(df2, how="inner", on=["topic", "Premise", "Conclusion"])
    mask = (df3["Validity"] == 1)
    df3.loc[mask, "Conclusion"] = "Therefore, " + df["Conclusion"].apply(first_word_to_lower)
    df3.to_csv(str(args.file)[:-4]+"_therefore.csv", encoding="utf8", index=False)
    df3[["topic", "Premise", "Conclusion"]].to_csv(str(args.file)[:-4]+"-without-labels"+"_therefore.csv", encoding="utf8", index=False)
