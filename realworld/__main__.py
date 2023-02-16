import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def makeax():
    fig, ax = plt.subplots(figsize=(7.16,2))
    return fig, ax

def plot_next(series):
    plt.scatter(series[:-1],series[1:])
    plt.show()

def main(path):
    df = pd.read_csv(path)
    isos = {
#             "CAN": "Canada",
            "FRA": "France",
#             "DEU": "Germany",
#             "ITA": "Italy",
            "JPN": "Japan",
            "GBR": "United Kingdom",
            "USA": "United States",
#             "CHN": "China",
            "OWID_WRL": "World",
    }

    usdf = df[df["iso_code"] == "USA"]
    xticks_pos = [0, len(usdf)]
    xticks = [usdf["date"].iloc[0], usdf["date"].iloc[-1]]

    fig, ax = makeax()
    ax.set_xticks(xticks_pos, xticks)
    for iso, name in isos.items():
        rows = df[df["iso_code"] == iso]
        ax.plot(rows["date"], rows["new_cases_smoothed_per_million"], label=name)
    ax.set_title("New Cases Smoothed per Million")
    ax.legend()

    fig, ax = makeax()
    ax.set_xticks(xticks_pos, xticks)
    for iso, name in isos.items():
        rows = df[df["iso_code"] == iso]
        ax.plot(rows["date"], rows["hosp_patients_per_million"], label=name)
    ax.set_title("Hospital Patients per Million")
    ax.legend()

    fig, ax = makeax()
    ax.hist(np.log(max(0.1, usdf["new_cases_per_million"].all())))

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
