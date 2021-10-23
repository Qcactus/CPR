import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np

np.random.seed(0)
from recq.tools.io import mkdir
from recq.tools.plot import degree_plot

DATASET = "movielens_10m"
# DATASET = "netflix"
# DATASET = "ifashion"


DATA_DIR = os.path.join("data", DATASET)
TEMP_DIR = os.path.join(DATA_DIR, "temp")
FIG_DIR = os.path.join("output", "figures", "degree_distribution", DATASET)
mkdir(DATA_DIR)
mkdir(TEMP_DIR)
mkdir(FIG_DIR)

if DATASET == "movielens_10m":
    with open("/Users/qi/Documents/data/movielens_10m/ratings.dat") as f:
        df = pd.read_csv(
            f,
            sep="::",
            names=["user", "item", "rating", "timestamp"],
            usecols=["user", "item", "rating"],
        )

    df = df[df["rating"] == 5][["user", "item"]]
    degree_plot({"implicit": df}, FIG_DIR)
    df.to_csv(os.path.join(TEMP_DIR, "implicit.csv"), index=False)
elif DATASET == "netflix":
    data = {}
    for filename in os.listdir("/Users/qi/Documents/data/netflix/training_set"):
        with open(
            os.path.join("/Users/qi/Documents/data/netflix/training_set", filename)
        ) as f:
            i = f.readline().strip(":\n")
            for l in f:
                splits = l.split(",")
                if int(splits[1]) == 5:
                    u = splits[0]
                    if u not in data:
                        data[u] = [i]
                    else:
                        data[u].append(i)
    # Reduce dataset.
    uni_users = list(data.keys())
    uni_users = np.random.choice(uni_users, 50000, replace=False)
    data = {key: data[key] for key in uni_users}
    users = []
    items = []
    for u, i_list in data.items():
        users.extend([u] * len(i_list))
        items.extend(i_list)
    df = pd.DataFrame({"user": users, "item": items})
    df.to_csv(os.path.join(TEMP_DIR, "implicit.csv"), index=False)
elif DATASET == "ifashion":
    data = {}
    with open("/Users/qi/Documents/data/ifashion/user_data.txt") as f:
        for l in f:
            u, _, i = l.strip("\n").split(",")
            if u not in data:
                data[u] = [i]
            else:
                data[u].append(i)
    # Reduce dataset.
    uni_users = list(data.keys())
    uni_users = np.random.choice(uni_users, 300000, replace=False)

    data = {key: data[key] for key in uni_users}

    users = []
    items = []
    for u, i_list in data.items():
        users.extend([u] * len(i_list))
        items.extend(i_list)

    df = pd.DataFrame({"user": users, "item": items})
    df.to_csv(os.path.join(TEMP_DIR, "implicit.csv"), index=False)
else:
    raise ValueError("Unsupported dataset.")


def get_df_info(df: pd.DataFrame):
    n_user = len(df["user"].unique())
    n_item = len(df["item"].unique())
    n_interact = len(df)
    density = n_interact / n_user / n_item
    print(
        f"#User: {n_user}, #Item: {n_item}, #Interaction: {n_interact}, Density: {density:.5f}"
    )
    return n_user, n_item, n_interact, density


def filter_degree(df: pd.DataFrame, min_u_d, min_i_d):
    print("Before being filtered by degree:")
    get_df_info(df)
    while 1:
        df = df.groupby("user").filter(lambda x: len(x) >= min_u_d)
        df = df.groupby("item").filter(lambda x: len(x) >= min_i_d)
        u_degree = df["user"].value_counts()
        i_degree = df["item"].value_counts()
        if u_degree.min() == min_u_d and i_degree.min() == min_i_d:
            break
    print("After being filtered by degree:")
    get_df_info(df)
    return df


df = pd.read_csv(os.path.join(TEMP_DIR, "implicit.csv"))
df = filter_degree(df, 3, 3)

degree_plot({"full": df}, FIG_DIR)
df.to_csv(os.path.join(TEMP_DIR, "full.csv"), index=False)

df = pd.read_csv(os.path.join(TEMP_DIR, "full.csv"))
if DATASET == "movielens_10m":
    max_ips = 1 / 60
elif DATASET == "netflix":
    max_ips = 1 / 60
elif DATASET == "ifashion":
    max_ips = 1 / 12
df["ips"] = df.groupby(["item"]).transform(lambda x: min(1 / len(x), max_ips))
unbias = df.sample(frac=0.3, weights="ips", random_state=0)[["user", "item"]]
train = df.drop(unbias.index)[["user", "item"]]
# Move users and items that do not appear in train back to train.
users = train["user"].unique()
items = train["item"].unique()
unbias = unbias[unbias["user"].isin(users) & unbias["item"].isin(items)]
train = df.drop(unbias.index)[["user", "item"]]
print(len(train))
print(len(unbias))

# Reindex.
users = train["user"].unique()
items = train["item"].unique()
id2idx_u = dict(zip(users, range(len(users))))
id2idx_i = dict(zip(items, range(len(items))))
for idx in range(len(unbias)):
    unbias.values[idx] = (
        id2idx_u[unbias.values[idx][0]],
        id2idx_i[unbias.values[idx][1]],
    )
for idx in range(len(train)):
    train.values[idx] = id2idx_u[train.values[idx][0]], id2idx_i[train.values[idx][1]]

test = unbias.sample(frac=2 / 3, random_state=0)
valid = unbias.drop(test.index)

print(len(train))
print(len(valid))
print(len(test))
degree_plot({"train": train, "valid": valid, "test": test}, FIG_DIR)

train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
valid.to_csv(os.path.join(DATA_DIR, "valid.csv"), index=False)
test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

df = pd.DataFrame({"id": list(id2idx_u.keys()), "idx": list(id2idx_u.values())})
df.to_csv(os.path.join(DATA_DIR, "id2idx_u.csv"), index=False)

df = pd.DataFrame({"id": list(id2idx_i.keys()), "idx": list(id2idx_i.values())})
df.to_csv(os.path.join(DATA_DIR, "id2idx_i.csv"), index=False)


# Create train sets with different degrees of bias.
def subsample(frac, pow):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train[str(pow)] = train.groupby(["item"]).transform(lambda x: len(x) ** pow)
    train_skew = train.sample(frac=frac, weights=str(pow), random_state=0)[
        ["user", "item"]
    ]
    users = train_skew["user"].unique()
    items = train_skew["item"].unique()
    train_missing = train[~(train["user"].isin(users) & train["item"].isin(items))]
    train_skew = pd.concat([train_skew, train_missing])
    degree_plot({"train_" + str(pow): train_skew}, FIG_DIR)
    train_skew.to_csv(os.path.join(DATA_DIR, "train_" + str(pow) + ".csv"), index=False)


subsample(0.7, 0.5)
subsample(0.7, -0.5)
