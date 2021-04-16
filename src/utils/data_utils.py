import os
import numpy as np
import pandas as pd

def init_dir(dir_path):
    """Create dir if not exists.
    Parameters
    ----------
        dir_path: dir path
    Returns
    None
    -------
    """
    os.makedirs(dir_path,exist_ok=True)

def train_dev_test_split(df, train_size=0.8):
    """Split data to train,dev,test.Train_size can be int or float in (0,1).
    Parameters
    ----------
        df: df need to split.
        train_size:can be int or float in (0,1).
    Returns
    -------
        df_train:train data
        df_dev:dev data
        df_test:test data
    """
    df = df.sample(frac=1, random_state=0).copy()
    if train_size < 1:
        train_size = int(train_size*df.shape[0])
    num = df.shape[0]
    dev_size = (num-train_size)//2
    df_train = df[:train_size]
    df_dev = df[train_size:dev_size+train_size]
    df_test = df[dev_size+train_size:]
    return df_train, df_dev, df_test

def split_3_save_data(save_dir,df,train_size=0.8):
    """Split data to train,dev,test.Than save data to savedir.Train_size can be int or float in (0,1).
    Parameters
    ----------
        save_dir: where to save data
        df: df need to split.
        train_size:can be int or float in (0,1).
    Returns
    -------
        df_train:train data
        df_dev:dev data
        df_test:test data
    """
    df_train,df_dev,df_test = train_dev_test_split(df,train_size)
    init_dir(save_dir)
    df_train.to_csv(os.path.join(save_dir,"train.csv"),index=False)
    df_dev.to_csv(os.path.join(save_dir,"dev.csv"),index=False)
    df_test.to_csv(os.path.join(save_dir,"test.csv"),index=False)
    return df_train, df_dev, df_test


def load_df(path):
    """load dataframe data,support csv path/xlsx path or df object
    Parameters
    ----------
        path: csv path/xlsx path/df object
    Returns
    -------
        df:df object
    """
    if isinstance(path,str):
        basename = os.path.basename(path)
        if '.csv' in basename:
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    else:
        df = path
    df['label'] = df['label'].apply(int)
    df = df.fillna("")
    return df


def get_one_data_report(path, name=""):
    df = load_df(path)
    report = df['label'].value_counts().to_dict()
    report['总量'] = df.shape[0]
    report['数据集'] = name
    raw_report_norm = df['label'].value_counts(normalize=True).to_dict()
    report_norm = {}
    for key, value in raw_report_norm.items():
        report_norm["{}占比".format(key)] = round(value, 3)
    report.update(report_norm)
    return report

def get_data_report(train_path, dev_path, test_path):
    """get report of all data
    Parameters
    ----------
        train_path: train_path
        dev_path: dev_path
        test_path: test_path
    Returns
    -------
        df_data_report:df_data_report
    """
    all_report = [get_one_data_report(train_path, "train"),
                  get_one_data_report(dev_path, "dev"),
                  get_one_data_report(test_path, "test")]
    df_data_report = pd.DataFrame(all_report)
    all_cols = df_data_report.columns.tolist()
    head_cols = ["数据集","总量"]
    other_cols = [x for x in all_cols if x not in head_cols]
    df_data_report = df_data_report[head_cols+other_cols]
    return df_data_report