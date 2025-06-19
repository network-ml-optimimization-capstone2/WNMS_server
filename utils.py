import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def get_data_from_csv(data_path, normalize=False, train_start=0, test_start=0,
                      max_train_size=None, max_test_size=None, mode='split_csv', months=2):
    """
    months 인풋 -> 마지막 1달을 Test 용으로 사용 (if mode = split_csv)

    """
    modes = ['split_csv', 'two_csv']
    if mode not in modes:
        raise ValueError("unknown mode " + str(mode))

    df = pd.read_csv(data_path)
    non_feature_str = ['date', 'time', 'number', 'id', 'indicator', 'label']
    label_str = ['label']
    feature_cols = [c for c in df.columns if not any(key in c for key in non_feature_str)]
    label_cols = [c for c in df.columns if any(label_key in c for label_key in label_str)]
    date_col = [c for c in df.columns if 'date' in c][0]
    df[date_col] = pd.to_datetime(df[date_col], format='%Y/%m/%d')
    df["__period"] = df[date_col].dt.to_period('M')

    periods = sorted(df["__period"].unique())
    if len(periods) < months:
        raise ValueError(f'Invalid period length {len(periods)}. Input months = {months}')

    sel = periods[-months:]
    test_period = sel[-1]
    train_periods = sel[:-1]

    print(f"periods: {periods}")
    print(f"test_period: {test_period}")
    print(f"train_periods: {train_periods}")

    train = df[df["__period"].isin(train_periods)].drop(columns="__period")
    test = df[df["__period"] == test_period].drop(columns="__period")

    # train.to_csv('train.csv', index=False)
    # test.to_csv('test.csv', index=False)

    train_data = train[feature_cols].to_numpy()
    test_data = test[feature_cols].to_numpy()
    test_label = test[label_cols].any(axis=1).astype(int).to_numpy()
    try:
        train_label = train[label_cols].any(axis=1).astype(int).to_numpy()
    except (KeyError, ValueError, IndexError, RuntimeError):
        train_label = None

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)

    print("train set shape: ", train_data.shape)
    print("train set label shape: ", None if train_label is None else train_label.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", None if test_label is None else test_label.shape)
    return (train_data, train_label), (test_data, test_label)


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    # No validation set case
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # Make validation set case
    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
