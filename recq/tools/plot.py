import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
import numpy as np
from recq.tools.io import mkdir


def plot_save(dirname,
              filename,
              data,
              x,
              y,
              hue=None,
              size=2,
              kind="scatter",
              xticks=None,
              ylim=None,
              title=None):
    sns.set(style="whitegrid", font_scale=1.1)
    if kind == "line":
        g = sns.lineplot(data=data, x=x, y=y, hue=hue, linewidth=2.5)
    else:
        g = sns.scatterplot(data=data, x=x, y=y, hue=hue, s=size)
    if xticks is not None:
        g.set_xticks(xticks)
    if ylim is not None:
        g.set(ylim=ylim)
    if title is not None:
        g.set_title(title)
    dirpath = os.path.join("output", "figures", dirname)
    mkdir(dirpath)
    plt.savefig(os.path.join(dirpath, filename + ".png"),
                dpi=300,
                bbox_inches='tight')
    plt.clf()


def epoch_plot(dirname, filename, data, y, hue=None, title=None):
    sns.set(style="whitegrid", font_scale=1.1)
    colors = np.array(sns.color_palette())
    g = sns.lineplot(
        data=data,
        x="Epoch",
        y=y,
        hue=hue,
        linewidth=2.5,
        palette=colors[[0, 2, 4, 8, 9, 1, 3]],
    )
    if title is not None:
        g.set_title(title)
    dirpath = os.path.join("output", "figures", dirname)
    mkdir(dirpath)
    plt.savefig(os.path.join(dirpath, filename + ".png"),
                dpi=300,
                bbox_inches='tight')
    plt.clf()


def degree_plot(datasets, dirname):
    # datasets: dict, key: name of dataset, value: dataframe
    sns.set(style="whitegrid")
    for name, dataset in datasets.items():

        plot = dataset["item"].value_counts().to_frame("Item Degree")
        bins = 50
        if plot["Item Degree"].max() < 100:
            bins = 'auto'
        sns.histplot(data=plot,
                     x="Item Degree",
                     bins=bins,
                     log_scale=(False, True))
        plt.savefig(os.path.join(dirname, name + "_i_degree.png"),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        plot = dataset["user"].value_counts().to_frame("User Degree")
        bins = 50
        if plot["User Degree"].max() < 100:
            bins = 'auto'
        sns.histplot(data=plot,
                     x="User Degree",
                     bins=bins,
                     log_scale=(False, True))
        plt.savefig(os.path.join(dirname, name + "_u_degree.png"),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()


def group_plot(df, y, dirname, filename, title=None):
    sns.set(style="whitegrid", font_scale=1.1)
    _, ax = plt.subplots()
    ax.set_title(title)
    colors = np.array(sns.color_palette())
    sns.lineplot(
        data=df[(df["Kind"] == "Train set") | (df["Kind"] == "Test set") |
                (df["Kind"] == "Equal probability")],
        x="Group",
        y=y,
        hue="Kind",
        palette=colors[[1, 6]],
        style="Kind",
        markers=True,
        linewidth=2.7,
        ax=ax)
    sns.barplot(data=df[(df["Kind"] != "Train set")
                        & (df["Kind"] != "Test set") &
                        (df["Kind"] != "Equal probability")],
                x="Group",
                y=y,
                hue="Kind",
                palette=colors[[0, 2, 4, 8, 9, 3]],
                ax=ax)
    plt.savefig(os.path.join(dirname, filename + ".png"),
                dpi=300,
                bbox_inches='tight')
    plt.clf()
