from cmath import e, log
import numpy as np

from Information_Extraction.Entropy import calculate_the_columns_entropy


def indentify_the_best_column_attribute(attributes_list, dataset):
    pass


def information_gain(column):
    entropy_values, parent_entropy = calculate_the_columns_entropy(column)
    information_gain_values = [
        {"key": entropy_key, "information-gain": parent_entropy - entropy_value}
        for entropy_key, entropy_value in zip(entropy_values.keys(), entropy_values.values())
    ]
    return [information_gain_values, parent_entropy]


def gain_ratio(column):
    information_gain_values, parent_entropy = information_gain(column)

    gain_ratio_values = [
        {
            "key": info["key"],
            "information-gain": info["information-gain"],
            "gain-ratio": info["information-gain"] / parent_entropy if parent_entropy != 0 else 0
        }
        for info in information_gain_values
    ]

    return [gain_ratio_values]
