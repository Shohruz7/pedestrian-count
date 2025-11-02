import pandas as pd


def main():
    df = pd.read_csv("data/Bi_Annual_Pedestrian_Counts_20251009.csv")
    print(df)


if __name__ == "__main__":
    main()
