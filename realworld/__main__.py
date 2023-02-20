import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from jpype import *

jarlocation = "../infodynamics-dist-1.6/infodynamics.jar"

HALF=(3.5, 2)
FULL=(7.16,3.5)

def makeax():
    fig, ax = plt.subplots(figsize=(7.16,2))
    return fig, ax

def plot_next(series):
    plt.scatter(series[:-1],series[1:])
    plt.show()

def te_cont(sourceArray, destArray):
    teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true")
    teCalc.initialise(1, 0.5)
    teCalc.setObservations(JArray(JDouble, 1)(sourceArray),
            JArray(JDouble, 1)(destArray))
    result = teCalc.computeAverageLocalOfObservations()
    return result

def te(sourceArray, destArray):
    teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true")
    teCalc.initialise(1, 0.5)
    teCalc.setObservations(JArray(JDouble, 1)(sourceArray),
            JArray(JDouble, 1)(destArray))
    result = teCalc.computeAverageLocalOfObservations()
    return result

def hist(series, nbins):
    fig, ax = makeax()
    ax.hist(series, bins=nbins)
    print(f"{counts}, {bins}")
    return fig, ax

def H(counts):
    N = np.sum(counts)
    func = lambda x: 0 if x == 0 else x * np.log2(x / N)
    return - np.sum([ func(count) for count in counts]) / N

def mi(source, dest):
    miCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    miCalc = miCalcClass()
    miCalc.initialise(1, 1) # Univariate
    miCalc.setProperty("k", "4") # 4 k nearest neighbors
    miCalc.setObservations(JArray(JDouble, 1)(source),
            JArray(JDouble, 1)(dest))
    result = miCalc.computeAverageLocalOfObservations()
    return result

def digitize(series):
    counts, bins = np.histogram(series, bins="auto")
    digitized = np.digitize(series, bins)

    plt.stairs(counts, bins)
    plt.show()
    print(f"Entropy of series: {H(counts)}")

    fig, ax = makeax()
    ax.plot(series / max(series), label="Before digitizing")
    ax.plot(digitized / max(digitized), label="After digitizing")
    ax.legend()
    plt.show()

    return digitized

def series_hist(x, y, ydig, counts, bins, colors, label, ax, ax_hist):
    ax_hist.tick_params(axis="y", labelleft=False)

#     ax.plot(ydig * bins[1], color=colors[0], label=f"{label} Discretized")
    ax.plot(y, color=colors[0], label=f"{label}")

    ax_hist.stairs(counts, bins, color=colors[0], linestyle="--", orientation="horizontal")

def subsamplete(source, dest, N, length):
    indices = np.random.randint(low=0,high=len(source)-length,size=N)

    forward = np.zeros(N)
    for i, index in enumerate(indices):
        forward[i] = te(source[index:index+length], dest[index:index+length])

    print(f"Forward: {te(source, dest)} \t | {np.mean(forward)} +/- {np.std(forward)}")



def foo(series, plot=True, N=100, length=100, print_bins=False, bins="auto"):

    colors = [("red", "orange"), ("blue", "lightblue"), ("green", "lightgreen"), ("purple", "lightpurple")]
    i = 0
    for label, serie in series.items():
        counts, bins = np.histogram(serie, bins=bins)
        digitized = np.digitize(serie, bins)
        if print_bins:
            print(bins)
            print(counts)
            print(digitized)
        print(f"H({label}) = {H(counts)}")

        if not plot:
            continue

        fig = plt.figure(figsize=HALF)
        gs = fig.add_gridspec(
                1, # Rows
                2, # Columns
                width_ratios=(4, 1),
    #             height_ratios=(1, 1),
                left = 0.1,
                right = 0.9, bottom = 0.1, top = 0.9,
                wspace = 0.00, hspace = 0.05)

        ax = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax)
        series_hist(None, serie, digitized, counts, bins, colors[i], label, ax, ax_hist)
        series[label]=digitized
        i += 1

        ax.legend()

    i = 0
    for sourcelabel, sourcearray in series.items():
        for destlabel, destarray in series.items():
            if sourcelabel == destlabel:
                i += 1
                if i == len(series.items()):
                    break
                continue
            print(f"MI({sourcelabel:8s}, {destlabel:8s}): {mi(sourcearray, destarray)}")
            subsamplete(sourcearray, destarray, N, length)

    plt.show()

#Logistic Equation: x(n+1) = r * x(n) * (1 - x(n-1))
def logistic(x0, r, t_end):
    x = np.zeros(t_end)
    x[0] = x0
    for i in range(1, t_end):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

def logmap():
    r_vals = [2, 3.9] # growth rate
    x0_vals = [0.5, 0.5 + 1e-6] # initial population value; change between 0.5 and 0.50000001
    split = 32
    t_end = split * 2 # number of time steps

    data = np.zeros((2,2,t_end))
    for i, r in enumerate(r_vals):
        for j, x0 in enumerate(x0_vals):
            res = logistic(x0, r, t_end)
            for t, x in enumerate(res):
                data[i][j][t] = x

    dic = {"before": data[0][0][:split], "after":data[0][0][split:]}
    foo(dic, plot=False, N=100, length=10)
    dic = {"before": data[0][1][:split], "after":data[0][1][split:]}
    foo(dic, plot=False, N=100, length=10)
    dic = {"before": data[1][0][:split], "after":data[1][0][split:]}
    foo(dic, plot=False, N=100, length=10)
    dic = {"before": data[1][1][:split], "after":data[1][1][split:]}
    foo(dic, plot=False, N=100, length=10)

def SIMCoV():
    df1 = pd.read_csv("data/per_480_inf_0.001.csv")
    df2 = pd.read_csv("data/per_600_inf_0.01.csv")
    df3 = pd.read_csv("data/per_1200_inf_0.1.csv")
    df4 = pd.read_csv("data/per_5000_inf_0.025.csv")

    dic = {
            "inft1": df1["inft"].values,
            "inft2": df2["inft"].values,
            "inft3": df3["inft"].values,
            "inft4": df4["inft"].values,
    }

    foo(dic, plot = False)

    dic = {
            "virs1": df1["virs"].values,
            "virs2": df2["virs"].values,
            "virs3": df3["virs"].values,
            "virs4": df4["virs"].values,
    }

    foo(dic, plot = False)

    dic = {
            "inft1": df1["inft"].values,
            "virs1": df1["virs"].values,
    }
    foo(dic, plot = False)
    dic = {
            "inft2": df2["inft"].values,
            "virs2": df2["virs"].values,
    }
    foo(dic, plot = False)
    dic = {
            "inft3": df3["inft"].values,
            "virs3": df3["virs"].values,
    }
    foo(dic, plot = False)
    dic = {
            "inft4": df4["inft"].values,
            "virs4": df4["virs"].values,
    }
    foo(dic, plot = False)

def OWID():
    path="data/owid-covid-data.csv"
    df = pd.read_csv(path)
    iso_dict = {
#             "CAN": "Canada",
#             "FRA": "France",
#             "DEU": "Germany",
#             "ITA": "Italy",
            "JPN": "Japan",
#             "GBR": "United Kingdom",
            "USA": "United States",
#             "CHN": "China",
            "OWID_WRL": "World",
    }
    # Select only the relevant rows and columns
    isos = ["OWID_WRL", "USA", "JPN"]
    attr = "new_cases"
    columns = ["iso_code", "date", attr, "hosp_patients"]
    mask = [iso in isos for iso in df["iso_code"].values]
    df = df[mask][columns]

    # Remove NaNs
    bad_dates = []
    for date in set(df["date"].values):
        if np.any(np.isnan(df[df["date"] == date][attr].values)):
            bad_dates.append(date)
        usadf = df[df["iso_code"] == "USA"]
        if np.any(np.isnan(usadf[usadf["date"] == date]["hosp_patients"].values)):
            bad_dates.append(date)

    # Make sure each country has the same dates
    for date in set(df["date"].values):
        for iso in isos:
            if not date in df[df["iso_code"] == iso]["date"].values:
                bad_dates.append(date)
                break

    mask = [not df.iloc[i]["date"] in bad_dates for i in range(len(df))]
    df = df[mask]

    hospitilizations = df[df["iso_code"] == "USA"]["hosp_patients"].values

    dic = { iso_dict[iso]: df[df["iso_code"] == iso][attr].values for iso in isos}
    dic["US Hospital Patients"] = hospitilizations

    fig, ax = plt.subplots(figsize=(4,2))
    for key, series in dic.items():
        ax.plot(series / max(series), label=key)
    ax.legend()
    ax.set_xticks([0,len(dic["United States"])], [df["date"].values[0], df["date"].values[-1]])
    ax.set_xlabel("Date")
    ax.set_ylabel("Count (Normalized)")
    ax.set_title("New COVID Cases and Hospital Patients by Day")
    plt.savefig("owid.pdf")

    if np.any(np.isnan(hospitilizations)):
        print("Something went wrong")
        return
    
    foo({"United States": dic["United States"], "World": dic["World"]}, plot=False)
    foo({"Japan": dic["Japan"], "World": dic["World"]}, plot=False)
    foo({"United States": dic["United States"], "US Hospital Patients": dic["US Hospital Patients"]}, plot=False)

    return
    counts, bins = np.histogram(sourceArray, bins="auto")
    plt.stairs(counts, bins, label="USA")
    print(f"Entropy of USA: {H(counts)}")
    sourceArray = getArray(df, "CAN", "new_cases_smoothed_per_million")
    counts, bins = np.histogram(sourceArray, bins="auto")
    plt.stairs(counts, bins, label="CAN")
    print(f"Entropy of CAN: {H(counts)}")
    plt.legend()
    
    fig, ax = makeax()
    array = getArray(df, "USA", "new_cases_per_million")
    ax.plot(array / max(array), label="Before digitizing")
    counts, bins = np.histogram(array, bins="auto")
    digitized = np.digitize(array, bins)
    ax.plot(digitized / max(digitized), label="After digitizing")
    ax.legend()

    plt.show()

    return

    # https://github.com/jlizier/jidt/wiki/PythonExamples#example-9---transfer-entropy-on-continuous-data-using-kraskov-estimators-with-auto-embedding
    # Example 3
    numObservations = 1000
    covariance=0.4

    sourceArray = list(df[df["iso_code"] == "USA"]["new_cases_smoothed_per_million"])
    destArray = list(df[df["iso_code"] == "USA"]["hosp_patients_per_million"])

    fig, ax = makeax()
    ax.hist(sourceArray, bins=10, label="new cases")
    ax.hist(destArray, bins=10, label="hospital patients")
    ax.legend()


    sourceArray = [random.normalvariate(0,1) for r in range(numObservations)]
    destArray = [0] + [sum(pair) for pair in zip([covariance*y for y in sourceArray[0:numObservations-1]], \
            [(1-covariance)*y for y in [random.normalvariate(0,1) for r in range(numObservations-1)]] ) ]
    
    print("TE result %.4f bits" % te(sourceArray, destArray))


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


    plt.show()

def main(path):
    SIMCoV()
    logmap()
    OWID()

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
            prog = "realworld",
            description = "Analyzes real-world timeseries using chaos and information theory")
    parser.add_argument("-f", "--file", help="The path to the data file",
            default="data/owid-covid-data.csv")

    args = parser.parse_args()

    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    startJVM(
            getDefaultJVMPath(),
            "-ea", "-Djava.class.path=" + jarlocation)

    main(args.file)
