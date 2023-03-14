import pandas as pd
import pathlib
import statistics
import matplotlib.pyplot as plt
import numpy as np


def get_data(dataset, decimal, architecture=""):
    """get data from .csv-files"""
    data = pd.read_csv(pathlib.Path(f"results/{dataset}_{architecture}mode0_{decimal}.csv"), names=["a"])
    return data["a"].values.tolist()


def get_statistics(data):
    """return mean and sd of data"""
    return statistics.mean(data), statistics.stdev(data)


def make_subplot(dataset_iteration, decimal_iteration, data1, data2=None, data3=None):
    """create subplot for given dataset and test condition"""
    ax = plt.subplot(3, 4, dataset_iteration * 4 + decimal_iteration + 1)
    mean, sd = get_statistics(data1)
    # error bar for using no labeled data
    plt.errorbar(0.5, mean, sd, fmt="o", linewidth=4, capsize=8, label="no labeled data", color="orange")
    if data2 is not None:
        # error bar for using only labeled outliers
        plt.errorbar(0.5, mean, sd, fmt="o", linewidth=4, capsize=8, label="only labeled outliers", color="skyblue")
        mean, sd = get_statistics(data2)
        # error bar for using labeled outliers and labeled normal data
        plt.errorbar(1.0, mean, sd, fmt="o", linewidth=4, capsize=8, label="labeled outliers and normal data",
                     color="royalblue")
        mean, sd = get_statistics(data3)
        # error bar for using labeled outliers and labeled normal data
        plt.errorbar(1.5, mean, sd, fmt="o", linewidth=4, capsize=8, color="midnightblue",
                 label="different weights for labeled outliers and normal data")
        ax.yaxis.set_visible(False)
    else:
        ax.set_ylabel(datasets[dataset_iteration], labelpad=10, fontsize=20)
    ax.set(xlim=(0, 2), xticks=[], ylim=(0.5, 1), yticks=np.arange(0.5, 1.01, 0.1))
    ax.tick_params(axis="y", labelsize=15)
    if dataset_iteration == 2:
        ax.set_xlabel(f"0.{decimals[decimal_iteration]} (0.0{decimals[decimal_iteration]})", labelpad=20, fontsize=20)


if __name__ == "__main__":
    # define used datasets, test conditions, and Deep SAD variants
    datasets = ("mnist", "fmnist", "cifar10")
    decimals = ("00", "05", "10", "20")
    architectures = ("standard_", "standard_normal_", "extended_")

    fig = plt.figure(figsize=(4, 3))
    # iterate over datasets and test conditions to create respective subplots
    for dataset_iteration, dataset in enumerate(datasets):
        for decimal_iteration, decimal in enumerate(decimals):
            if decimal_iteration == 0:
                data = get_data(dataset, decimal)
                make_subplot(dataset_iteration, decimal_iteration, data)
            else:
                data = [get_data(dataset, decimal, architecture) for architecture in architectures]
                make_subplot(dataset_iteration, decimal_iteration, data[0], data[1], data[2])

    # formatting whole graphic
    fig.subplots_adjust(wspace=0, hspace=0.1)
    fig.supxlabel("ratio labeled normal data (ratio labeled outliers)", fontsize=25)

    plt.subplots_adjust(right=0.6)
    plt.legend(bbox_to_anchor=(4.0, 3.2), fontsize=20)
    plt.show()


