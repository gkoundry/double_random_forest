import os
import time
from dataclasses import dataclass
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ucimlrepo import fetch_ucirepo

from double_random_forest import (
    DoubleRandomForestClassifier,
    DoubleRandomForestRegressor,
)


@dataclass
class TestSummary:
    dataset_name: str
    mean_rf_score: float
    mean_drf_score: float


def preprocess_data(dataset):
    x = dataset.data.features.copy()
    for colname in list(x.columns):
        if x[colname].dtype is np.dtype(object):
            x.loc[:, f"encoded_{colname}"] = OrdinalEncoder(
                encoded_missing_value=-1
            ).fit_transform(x[[colname]])
            x.pop(colname)
        elif x[colname].isnull().sum():
            x.loc[:, f"fillna_{colname}"] = x[colname].fillna(x[colname].median())
            x.pop(colname)
    if dataset.data.targets.ndim == 2 and dataset.data.targets.shape[1] > 1:
        y = dataset.data.targets.iloc[:, 0]
    else:
        y = dataset.data.targets.values.ravel()
    if y.dtype is np.dtype(object):
        y = OrdinalEncoder().fit_transform(y.astype(str).reshape(-1, 1)).ravel()
    x["__target__"] = y
    return x


def fetch_data(dataset_name):
    filename = f"{dataset_name}.csv"
    if os.path.isfile(filename):
        x = pd.read_csv(filename)
    else:
        dataset = fetch_ucirepo(name=dataset_name)
        x = preprocess_data(dataset)
        x.to_csv(filename, index=False)
    y = x.pop("__target__").values
    return x.values, y


def test_data(x, y, rs, is_regression, model_class):
    if is_regression:
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=0.3, random_state=rs
        )
        valid_train_x, valid_test_x, valid_train_y, valid_test_y = train_test_split(
            train_x, train_y, test_size=0.3, random_state=rs
        )
    else:
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=0.3, stratify=y, random_state=rs
        )
        valid_train_x, valid_test_x, valid_train_y, valid_test_y = train_test_split(
            train_x, train_y, test_size=0.3, stratify=train_y, random_state=rs
        )
    n_features = train_x.shape[1]
    best_score = -np.inf
    for max_features_power in (0.25, 0.5, 0.75):
        max_features = int(n_features**max_features_power)
        for min_samples_leaf in (1, 5, 25):
            rf = model_class(
                n_estimators=200,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            )
            rf.fit(valid_train_x, valid_train_y)
            if is_regression:
                pred = rf.predict(valid_test_x)
                score = spearmanr(valid_test_y, pred).correlation
            else:
                pred = rf.predict_proba(valid_test_x)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    pred = pred[:, 1]
                score = roc_auc_score(
                    valid_test_y, pred, multi_class="ovr" if pred.ndim > 1 else "raise"
                )
            if score > best_score:
                best_score = score
                best_mfp = max_features_power
                best_msl = min_samples_leaf

    rf = model_class(
        n_estimators=200,
        min_samples_leaf=best_msl,
        max_features=int(n_features**best_mfp),
    )
    print(
        f"Best params: {model_class.__name__} max_features={best_mfp}, min_samples_leaf={best_msl}"
    )
    rf.fit(train_x, train_y)
    if is_regression:
        pred = rf.predict(test_x)
        return spearmanr(test_y, pred).correlation
    else:
        pred = rf.predict_proba(test_x)
        if pred.ndim == 2 and pred.shape[1] == 2:
            pred = pred[:, 1]
        return roc_auc_score(
            test_y, pred, multi_class="ovr" if pred.ndim > 1 else "raise"
        )


def test_data_loop(x, y, ds_name, is_regression):
    with Pool(4) as pool:
        rf_model_class = (
            RandomForestRegressor if is_regression else RandomForestClassifier
        )
        arg_gen = ((x, y, rs, is_regression, rf_model_class) for rs in range(10))
        rf_results = pool.starmap(test_data, arg_gen)
        drf_model_class = (
            DoubleRandomForestRegressor
            if is_regression
            else DoubleRandomForestClassifier
        )
        arg_gen = ((x, y, rs, is_regression, drf_model_class) for rs in range(10))
        drf_results = pool.starmap(test_data, arg_gen)

    mean_rf_score = np.mean(rf_results)
    mean_drf_score = np.mean(drf_results)
    return TestSummary(
        ds_name,
        mean_rf_score,
        mean_drf_score,
    )


def create_bar_chart(summaries):
    # Extract data from the summaries list
    names = [s.dataset_name for s in summaries]
    rf_scores = [s.mean_rf_score for s in summaries]
    drf_scores = [s.mean_drf_score for s in summaries]

    # Set the width of the bars and the positions
    num_datasets = len(names)
    x = np.arange(num_datasets)  # x-coordinates for the groups
    bar_width = 0.35  # width of each bar

    _, ax = plt.subplots()

    ax.bar(x - bar_width / 2, rf_scores, bar_width, label="RF Score")
    ax.bar(x + bar_width / 2, drf_scores, bar_width, label="DRF Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Mean RF vs DRF Scores (AUC or SpearmanR) by Dataset")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("benchmark_chart.png")
    plt.close()


def run():
    results = []
    for short_name, full_name, is_regression in (
        ("SUPPORT2", "SUPPORT2", False),
        ("Cancer", "Breast Cancer", False),
        ("Digits", "Optical Recognition of Handwritten Digits", False),
        ("Yeast", "Yeast", False),
        ("Shuttle", "Statlog (Shuttle)", False),
        ("Page", "Page Blocks Classification", False),
        ("Image", "Image Segmentation", False),
        ("Spam", "Spambase", False),
        ("Landsat", "Statlog (Landsat Satellite)", False),
        ("Magic", "MAGIC Gamma Telescope", False),
        ("Wine", "Wine Quality", True),
        ("Abalone", "Abalone", True),
        ("BikeSharing", "Bike Sharing", True),
    ):
        start_time = time.time()
        x, y = fetch_data(full_name)
        test_summary = test_data_loop(x, y, short_name, is_regression)
        results.append(test_summary)
        print(test_summary, flush=True)
        create_bar_chart(results)
        print(f"Time: {time.time() - start_time} {short_name}", flush=True)


if __name__ == "__main__":
    run()
