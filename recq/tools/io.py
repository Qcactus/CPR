import os
import pickle
import pandas as pd
from recq.tools.dataformat import df2dict


def print_seperate_line():
    print("=" * 140)


def save_pkl(obj, dir, filename):
    with open(os.path.join(dir, filename + ".pkl"), "wb") as f:
        pickle.dump(obj, f)


def load_pkl(dir, filename):
    with open(os.path.join(dir, filename + ".pkl"), "rb") as f:
        return pickle.load(f)


def load_csv_2dict(dir, filename):
    df = pd.read_csv(os.path.join(dir, filename + ".csv"))
    d = df2dict(df, "user", "item")
    return d


def write_u_i_to_file(u_i_list, path):
    """Write user-item interaction list to a file.

    Args:
        u_i_list (list): 2D list. u_i_list[i] contains items interacted with ith user.
        path (str): File path to store the list with format user_id item_id item_id ... item_id
    """
    with open(path, "w") as f:
        f.write(
            "\n".join(
                [
                    " ".join(["%d" % u] + [str(x) for x in i_list])
                    for u, i_list in enumerate(u_i_list)
                ]
            )
        )


def read_u_i_from_file(path):
    """Read user-item interaction list to a file.

    Args:
        path (str): File path to restore the list with format user_id item_id item_id ... item_id

    Returns:
        list: 2D list. u_i_list[i] contains items interacted with ith user.
    """
    u_i_list = []
    with open(path) as f:
        lines = f.read().split("\n")
        u_i_list = [[int(x) for x in l.strip("\n").split(" ")[1:]] for l in lines]
    return u_i_list


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lines(filepath, first, last, after="", drop_first=False, drop_last=False):
    """Get lines after one line starting with `after` that meet the requirements:
    the first line starts with `first`, and the last line
    starts with `last`.

    Args:
        filename (string)
        first_prefix (string)
        last_prefix (string)

    Return:
        list of lines.
    """
    with open(filepath) as f:
        begin = False
        choose = False
        for l in f:
            l = l.strip("\n")
            if begin:
                if choose:
                    if l.startswith(last):
                        choose = False
                        if not drop_last:
                            lines.append(l)
                        return lines
                    else:
                        lines.append(l)
                elif l.startswith(first):
                    choose = True
                    lines = [] if drop_first else [l]
            elif l.startswith(after):
                begin = True


def get_filepaths(dir, prefix="", suffix="", filter=""):
    paths = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if f.startswith(prefix) and f.endswith(suffix) and filter in f
    ]
    return [path for path in paths if os.path.isfile(path)]


def get_all_lines(filepath, first, last, after="", drop_first=False, drop_last=False):
    """Get lines after one line starting with `after` that meet the requirements:
    the first line starts with `first`, and the last line
    starts with `last`.

    Args:
        filename (string)
        first_prefix (string)
        last_prefix (string)

    Return:
        list(of lists of strings): Each sub-list contains consecutive
        lines that meet the requirements.
    """
    all_lines = []
    with open(filepath) as f:
        begin = False
        choose = False
        for l in f:
            l = l.strip("\n")
            if begin:
                if choose:
                    if l.startswith(last):
                        begin = False
                        choose = False
                        if not drop_last:
                            lines.append(l)
                        all_lines.append(lines)
                    else:
                        lines.append(l)
                elif l.startswith(first):
                    choose = True
                    lines = [] if drop_first else [l]
            elif l.startswith(after):
                begin = True
    return all_lines


def find_from_lines(lines, substr):
    for l in lines:
        if substr in l:
            return True
    return False


def create_df_from_lines(lines):
    import pandas as pd

    columns = lines[0].split()
    data = [[float(x) for x in l.split()] for l in lines[1:]]
    return pd.DataFrame(data, columns=columns)


def get_metric_from_lines(lines, metric, k):
    for l in lines:
        splits = l.replace(":", " ").split()
        if splits[0] == metric and splits[1] == "@" + str(k):
            return float(splits[2])


def get_metric_in_group_from_lines(lines, metric, k):
    for l in lines:
        splits = l.split()
        if splits[0] == metric and splits[1] == "@" + str(k):
            return [float(x) for x in splits[4:]]
