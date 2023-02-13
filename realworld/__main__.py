import pandas as pd
import matplotlib.pyplot as plt

def makeax():
    ax, fig = plt.subplots()
    return ax, fig

def main(path):
    df = pd.read_csv(path)
    usa_new_cases = df[df["iso_code"] == "USA"]["new_cases"]
    plt.plot(usa_new_cases)
    plt.show()

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
            prog = "realworld",
            description = "Analyzes real-world timeseries using chaos and information theory")
    parser.add_argument("-f", "--file", help="The path to the data file",
            default="data/owid-covid-data.csv")

    args = parser.parse_args()

    main(args.file)
