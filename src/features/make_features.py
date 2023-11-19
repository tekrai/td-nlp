import ast
import string
from ast import literal_eval
from typing import List

import numpy as np
import pandas as pd


def make_features(df, task):
    match task:
        case "is_comic_video":
            train_x, train_y = make_video_name_features(df)
        case "is_name":
            train_x, train_y = make_features_is_name(df)
        case "find_comic_name":
            train_x, train_y = make_features_find_comic_name(df)
        case _:
            raise ValueError("Unknown task, impossible to extract features")

    return train_x, train_y


def make_video_name_features(df: pd.DataFrame):
    return df["video_name"], df["is_comic"]


def make_features_is_name(df: pd.DataFrame):
    df['is_name'] = df['is_name'].apply(ast.literal_eval)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)

    def extract_features(tokens: List[str], labels: List[bool]):
        """This function extract the features for our model. 'NONE' is for DictVect"""
        return [
            {
                'word': token,
                'is_capitalized': token[0].isupper(),
                'is_punctuation': token in string.punctuation,
                'is_starting_word': index == 0,
                'is_final_word': index == len(tokens) - 1,
                'previous_word': tokens[index - 1] if index > 0 else 'NONE',
                'next_word': tokens[index + 1] if index < len(tokens) - 1 else 'NONE',
                'is_name': label,
            } for index, (token, label) in enumerate(zip(tokens, labels)) if token
        ]

    df["features"] = df.apply(lambda row: extract_features(row['tokens'], row['is_name']), axis=1)

    features = pd.DataFrame([feature for sublist in df['features'] for feature in sublist])

    train_x = features.drop("is_name", axis=1).to_dict(orient='records')

    train_y = features["is_name"]

    return train_x, train_y


def make_features_find_comic_name(df: pd.DataFrame):
    return df["video_name"], df["comic_name"]
