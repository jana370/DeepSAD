import pandas as pd
import pathlib
import statistics
import matplotlib.pyplot as plt
import numpy as np


def get_data(dataset, decimal):
    """get data from .csv-files"""
    data = pd.read_csv(pathlib.Path(f"results/{dataset}_standard_pollution0_{decimal}.csv"), names=["a"])
    return data["a"].values.tolist()


def get_statistics(data):
    """return mean and sd of data"""
    return statistics.mean(data), statistics.stdev(data)


def make_subplot(dataset_iteration, data1, data2, data3):
    """create subplot for given dataset"""
    ax = plt.subplot(3, 1, dataset_iteration + 1)
    mean, sd = get_statistics(data1)
    # error bar for using no pollution
    plt.errorbar(0.5, mean, sd, fmt="o", linewidth=4, capsize=8, label="no pollution", color="yellowgreen")
    mean, sd = get_statistics(data2)
    # error bar for using 0.01 pollution
    plt.errorbar(1.0, mean, sd, fmt="o", linewidth=4, capsize=8, label="0.01 pollution",
                     color="limegreen")
    mean, sd = get_statistics(data3)
    # error bar for using 0.05 pollution
    plt.errorbar(1.5, mean, sd, fmt="o", linewidth=4, capsize=8, color="darkgreen",
                 label="0.05 pollution")
    ax.set_ylabel(datasets[dataset_iteration], labelpad=10, fontsize=20)
    ax.tick_params(axis="y", labelsize=15)
    ax.set(xlim=(0, 2), xticks=[], ylim=(0.5, 1), yticks=np.arange(0.5, 1.01, 0.1))


if __name__ == "__main__":
    # define used datasets, and test conditions
    datasets = ("mnist", "fmnist", "cifar10")
    decimals = ("00", "01", "05")

    fig = plt.figure(figsize=(1, 3))
    # iterate over datasets to create respective subplots
    for dataset_iteration, dataset in enumerate(datasets):
        data = [get_data(dataset, decimal) for decimal in decimals]
        make_subplot(dataset_iteration, data[0], data[1], data[2])

    # formatting whole graphic
    fig.subplots_adjust(wspace=0, hspace=0.1)
    plt.subplots_adjust(right=0.3)
    plt.legend(bbox_to_anchor=(1.8, 3.2), fontsize=20)
    plt.show()


