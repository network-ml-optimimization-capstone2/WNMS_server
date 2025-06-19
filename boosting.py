import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import yaml
from utils import *
from tqdm import tqdm, trange
import pickle as pkl
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels

    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if label is None:
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict.astype(int)

# 1) Sliding window dataset
class SlidingWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, window_size: int, labels: np.ndarray=None):
        """
        series: np.ndarray, shape (T, k)
        labels: np.ndarray, shape (T,) or None
        """
        assert len(series) == len(labels)
        self.series = series
        self.labels = labels
        self.window_size = window_size
    def __len__(self):
        return len(self.series) - self.window_size
    def __getitem__(self, idx):
        x = self.series[idx:idx+self.window_size]     # (window_size, k)
        if self.labels is not None:
            y = self.labels[idx+self.window_size-1]   # 윈도우 끝 시점의 라벨
            return torch.from_numpy(x).float(), int(y)
        else:
            return torch.from_numpy(x).float(), -1

def loader_to_numpy(loader):
    """
    DataLoader에서 (x_win, y) 배치들을 받아
     - x_flat: (n_samples, window_size * n_features)
     - y_arr : (n_samples,)
    로 변환해 리턴
    """
    X_list, y_list = [], []
    for x_batch, y_batch in loader:
        # x_batch: torch.Tensor (B, window_size, n_features)
        # y_batch: torch.Tensor (B,)
        B, W, K = x_batch.shape
        x_flat = x_batch.reshape(B, W*K).cpu().numpy()
        y_np   = y_batch.cpu().numpy()
        X_list.append(x_flat)
        y_list.append(y_np)
    x = np.vstack(X_list)
    y = np.hstack(y_list)
    return x, y

def ada_boost(x_train, y_train, x_test, y_test, output_dir, n_estimator=100, random_state=42):
    adb = AdaBoostClassifier(n_estimators=n_estimator, random_state=random_state)
    result_dict = dict(
        y_label=y_test.tolist(),
        y_pred=[],
        y_pred_pa=[]
    )

    tqdm.write(f'Fitting AdaBoost classifier on {len(x_train)} samples')
    adb.fit(x_train, y_train)
    tqdm.write(f'Finish adb.fit -> predicting ...')
    y_pred = adb.predict(x_test)
    y_pred.astype(int)
    print(f'y_pred type: {type(y_pred)}, y_pred shape: {y_pred.shape}, y_pred_1count {np.sum(y_pred==1)}')
    # Save output
    output_csv_path = os.path.join(output_dir, f'ada_boost.csv')
    output_f1_path = os.path.join(output_dir, f'ada_f1_scores.yaml')
    # Calculate Scores
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Original pred counts:", np.unique(y_pred, return_counts=True))
    result_dict['y_pred'] = y_pred.tolist()
    y_pred_pa = adjust_predicts(None, y_test, None, pred=y_pred)
    y_pred_pa = y_pred_pa.astype(int)
    print("Adjusted pred counts:", np.unique(y_pred_pa, return_counts=True))
    result_dict['y_pred_pa'] = y_pred_pa.tolist()

    f1_pa = f1_score(y_test, y_pred_pa)
    tn_pa, fp_pa, fn_pa, tp_pa = confusion_matrix(y_test, y_pred_pa).ravel()
    f1_dict = dict(
        estimator=n_estimator,
        random_state=random_state,
        f1=float(f1),
        tn=int(tn),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        f1pa=float(f1_pa),
        tn_pa=int(tn_pa),
        tp_pa=int(tp_pa),
        fp_pa=int(fp_pa),
        fn_pa=int(fn_pa),
    )
    with open(output_f1_path, 'w', encoding='utf-8') as f:
        yaml.dump(f1_dict, f, allow_unicode=True, sort_keys=False)


    df = pd.DataFrame(result_dict)
    df.to_csv(output_csv_path, index=True, index_label='id')
    tqdm.write(f'AdaBoost Done')

def xg_boost(x_train, y_train, x_test, y_test, output_dir, n_estimator=100, random_state=42):
    xgb = XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
        n_estimators=n_estimator, random_state=random_state)

    result_dict = dict(
        y_label=y_test.tolist(),
        y_pred=[],
        y_pred_pa=[]
    )

    # Learn & Inference
    tqdm.write(f'Fitting XGBoost classifier on {len(x_train)} samples')
    xgb.fit(x_train, y_train)
    tqdm.write(f'Finish xgb.fit -> predicting ...')
    y_pred = xgb.predict(x_test)
    y_pred.astype(int)
    print(f'y_pred type: {type(y_pred)}, y_pred shape: {y_pred.shape}, y_pred_1count {np.sum(y_pred==1)}')

    # Save output
    output_csv_path = os.path.join(output_dir, f'xgboost.csv')
    output_f1_path = os.path.join(output_dir, f'xg_f1_scores.yaml')
    # Calculate Scores
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Original pred counts:", np.unique(y_pred, return_counts=True))
    result_dict['y_pred'] = y_pred.tolist()
    y_pred_pa = adjust_predicts(None, y_test, None, pred=y_pred)
    y_pred_pa = y_pred_pa.astype(int)
    print("Adjusted pred counts:", np.unique(y_pred_pa, return_counts=True))
    result_dict['y_pred_pa'] = y_pred_pa.tolist()

    f1_pa = f1_score(y_test, y_pred_pa)
    tn_pa, fp_pa, fn_pa, tp_pa = confusion_matrix(y_test, y_pred_pa).ravel()
    f1_dict = dict(
        estimator=n_estimator,
        random_state=random_state,
        f1=float(f1),
        tn=int(tn),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        f1pa=float(f1_pa),
        tn_pa=int(tn_pa),
        tp_pa=int(tp_pa),
        fp_pa=int(fp_pa),
        fn_pa=int(fn_pa),
    )
    with open(output_f1_path, 'w', encoding='utf-8') as f:
        yaml.dump(f1_dict, f, allow_unicode=True, sort_keys=False)


    df = pd.DataFrame(result_dict)
    df.to_csv(output_csv_path, index=True, index_label='id')
    tqdm.write(f'XGBoost Done')

def light_gbm_boost(x_train, y_train, x_test, y_test, output_dir, n_estimator=100, random_state=42):
    # 1) 모델 초기화
    lgbm = LGBMClassifier(
        n_estimators=n_estimator,
        random_state=random_state
    )

    # 2) 결과 저장용 dict
    result_dict = dict(
        y_label=y_test.tolist(),
        y_pred=[],
        y_pred_pa=[]
    )

    # 3) 학습
    tqdm.write(f'Fitting LightGBM classifier on {len(x_train)} samples')
    lgbm.fit(x_train, y_train)
    tqdm.write('Finish lgbm.fit -> predicting ...')

    # 4) 예측
    y_pred = lgbm.predict(x_test).astype(int)
    print(f'y_pred type: {type(y_pred)}, shape: {y_pred.shape}, count1: {np.sum(y_pred == 1)}')

    # 5) CSV / YAML 경로 설정
    output_csv_path = os.path.join(output_dir, 'lightgbm.csv')
    output_f1_path = os.path.join(output_dir, 'lgbm_f1_scores.yaml')

    # 6) 성능 계산
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    result_dict['y_pred'] = y_pred.tolist()

    # 7) adjust_predicts 적용
    y_pred_pa = adjust_predicts(None, y_test, None, pred=y_pred).astype(int)
    print("Adjusted pred counts:", np.unique(y_pred_pa, return_counts=True))
    result_dict['y_pred_pa'] = y_pred_pa.tolist()

    f1_pa = f1_score(y_test, y_pred_pa)
    tn_pa, fp_pa, fn_pa, tp_pa = confusion_matrix(y_test, y_pred_pa).ravel()

    # 8) YAML 로 F1, confusion matrix 저장
    f1_dict = dict(
        estimator=n_estimator,
        random_state=random_state,
        f1=f1,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        f1pa=f1_pa,
        tn_pa=int(tn_pa), fp_pa=int(fp_pa), fn_pa=int(fn_pa), tp_pa=int(tp_pa),
    )
    with open(output_f1_path, 'w', encoding='utf-8') as f:
        yaml.dump(f1_dict, f, allow_unicode=True, sort_keys=False)

    # 9) 예측 결과를 DataFrame 으로 저장
    df = pd.DataFrame(result_dict)
    df.to_csv(output_csv_path, index=True, index_label='id')

    tqdm.write('LightGBM Done')

if __name__ == "__main__":
    global id
    id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="None")
    parser.add_argument('--estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--shuffle_dataset', type=str2bool, default=True)
    parser.add_argument("--lookback", type=int, default=50)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=256)
    args = parser.parse_args()

    dataset = args.dataset
    normalize = args.normalize
    window_size = args.lookback
    batch_size = args.bs
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    n_estimator = args.estimators
    random_state = args.random_state

    fname_ext = os.path.basename(dataset)
    fname, _ = os.path.splitext(fname_ext)
    # Load data
    # output_path = f'output_ada/{id}_{n_estimator}_{random_state}/'
    output_path = f'output_ada/handover_march/{fname}/'
    os.makedirs(output_path, exist_ok=True)
    # (x_train, y_train), (x_test, y_test) = get_data(dataset, normalize=normalize)
    (x_train, y_train), (x_test, y_test) = get_data_from_csv(dataset, normalize=normalize)

    param_dict = dict(
        n_estimators=n_estimator,
        random_state=random_state,
    )

    # n_features = x_train.shape[1]
    # target_dims = None

    train_dataset = SlidingWindowDataset(x_train, window_size, y_train)
    test_dataset = SlidingWindowDataset(x_test, window_size, y_test)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    X_tr, y_tr = loader_to_numpy(train_loader)
    # X_val, y_val = loader_to_numpy(val_loader)
    X_te, y_te = loader_to_numpy(test_loader)

    print("X_tr shape:", X_tr.shape, "y_tr sum:", y_tr.sum())
    print("X_te shape:", X_te.shape, "y_te sum:", y_te.sum())

    xg_boost(X_tr, y_tr, X_te, y_te, output_path, n_estimator, random_state)
    # ada_boost(X_tr, y_tr, X_te, y_te, output_path, n_estimator, random_state)
    light_gbm_boost(X_tr, y_tr, X_te, y_te, output_path, n_estimator, random_state)
