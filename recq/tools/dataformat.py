import pandas as pd


def invert_dict(d, sort=False):
    inverse = {}
    for key in d:
        for value in d[key]:
            if value not in inverse:
                inverse[value] = [key]
            else:
                inverse[value].append(key)
    return inverse


def df2dict(df, key_name, value_name):
    d = df.groupby(key_name)[value_name].apply(list).to_dict()
    return d
