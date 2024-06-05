import os
import yaml
import argparse
import pandas as pd

def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml")
    opt = parser.parse_args()

    with open(opt.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def read_files(config):
    data_path = config["data_path"]
    dict_path = config["dict_path"]

    train_fname = config["train_fname"]
    val_fname = config["val_fname"]
    dict_fname = config["dict_fname"]

    train_contexts = pd.read_csv(os.path.join(data_path, train_fname))
    val_contexts = pd.read_csv(os.path.join(data_path, val_fname))
    dictionary = pd.read_json(os.path.join(dict_path, dict_fname))

    return train_contexts, val_contexts, dictionary