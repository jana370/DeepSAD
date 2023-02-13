import pandas as pd
import pathlib
import statistics
import matplotlib.pyplot as plt
import numpy as np


def get_data(dataset, decimal, architecture=""):
    data = pd.read_csv(pathlib.Path(f"results/{dataset}_{architecture}mode0_{decimal}.csv"), names=["a"])
    return data["a"].values.tolist()


def get_statistics(data):
    return statistics.mean(data), statistics.stdev(data)


def make_subplot(dataset_iteration, decimal_iteration, data1, data2=None, data3=None):
    ax = plt.subplot(3, 4, dataset_iteration * 4 + decimal_iteration + 1)
    mean, sd = get_statistics(data1)
    ax.errorbar(0.5, mean, sd, fmt="o", linewidth=2, capsize=6, label="only labeled outliers", color="skyblue")
    if data2 is not None:
        mean, sd = get_statistics(data2)
        plt.errorbar(1.0, mean, sd, fmt="o", linewidth=2, capsize=6, label="labeled outliers and normal data",
                     color="royalblue")
        mean, sd = get_statistics(data3)
        plt.errorbar(1.5, mean, sd, fmt="o", linewidth=2, capsize=6, color="midnightblue",
                 label="different weights for labeled outliers and normal data")
        ax.yaxis.set_visible(False)
    else:
        ax.set_ylabel(datasets[dataset_iteration], labelpad=10, fontsize=15)
    ax.set(xlim=(0, 2), xticks=[], ylim=(0.5, 1), yticks=np.arange(0.5, 1.01, 0.1))
    if dataset_iteration == 2:
        ax.set_xlabel(f"0.{decimals[decimal_iteration]}", labelpad=10, fontsize=15)


if __name__ == "__main__":
    datasets = ("mnist", "fmnist", "cifar10")
    decimals = ("00", "05", "10", "20")
    architectures = ("standard_", "standard_normal_", "extended_")

    fig = plt.figure(figsize=(4, 3))
    for dataset_iteration, dataset in enumerate(datasets):
        for decimal_iteration, decimal in enumerate(decimals):
            if decimal_iteration == 0:
                data = get_data(dataset, decimal)
                make_subplot(dataset_iteration, decimal_iteration, data)
            else:
                data = [get_data(dataset, decimal, architecture) for architecture in architectures]
                make_subplot(dataset_iteration, decimal_iteration, data[0], data[1], data[2])

    fig.subplots_adjust(wspace=0, hspace=0.1)
    fig.supxlabel("ratio of labeled normal data", fontsize=20)

    plt.subplots_adjust(right=0.8)
    plt.legend(bbox_to_anchor=(2.1, 3.2))
    plt.show()


