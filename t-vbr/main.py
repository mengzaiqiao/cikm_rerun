import os
import sys
import time

sys.path.append("../")
sys.path.append("../../")

import numpy as np
import pandas as pd
import pickle
import argparse
import GPUtil
import psutil
from tqdm import tqdm
from livelossplot import PlotLosses

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from utils import logger, data_util
from utils.monitor import Monitor
from utils.sampler import Sampler
from src.t_vbr import TVBR


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run DVBCR.")
    parser.add_argument("--PERCENT", nargs="?", type=float, default=1, help="PERCENT")
    parser.add_argument(
        "--DATASET", nargs="?", type=str, default="dunnhumby", help="DATASET"
    )
    parser.add_argument(
        "--ITEM_FEA_TYPE", nargs="?", type=str, default="random", help="ITEM_FEA_TYPE"
    )
    parser.add_argument(
        "--N_SAMPLE", nargs="?", type=int, default=10000, help="N_SAMPLE"
    )
    parser.add_argument("--MODEL", nargs="?", type=str, default="VAE_D", help="MODEL")
    parser.add_argument("--EMB_DIM", nargs="?", type=int, default=60, help="EMB_DIM")
    parser.add_argument("--LAT_DIM", nargs="?", type=int, default=512, help="LAT_DIM")
    parser.add_argument(
        "--INIT_LR", nargs="?", type=float, default=0.0025, help="INIT_LR"
    )
    parser.add_argument(
        "--BATCH_SIZE", nargs="?", type=int, default=500, help="BATCH_SIZE"
    )
    parser.add_argument(
        "--TIME_STEP", nargs="?", type=int, default=10, help="TIME_STEP"
    )
    parser.add_argument("--OPTI", nargs="?", type=str, default="RMSprop", help="OPTI")
    parser.add_argument("--ALPHA", nargs="?", type=float, default=0.05, help="ALPHA")
    parser.add_argument("--EPOCH", nargs="?", type=int, default=120, help="EPOCH")
    parser.add_argument(
        "--REMARKS", nargs="?", type=str, default="test", help="REMARKS"
    )
    parser.add_argument(
        "--CPU", nargs="?", type=str2bool, default="False", help="use cpu only?"
    )
    parser.add_argument(
        "--SAVE_LOG",
        nargs="?",
        type=str2bool,
        default="True",
        help="Save all the logs?",
    )
    parser.add_argument(
        "--LOAD_SAVED",
        nargs="?",
        type=str2bool,
        default="False",
        help="Load data and sample from file?",
    )
    parser.add_argument(
        "--RESULT_FILE",
        nargs="?",
        type=str,
        default="result_DVBCR_test.csv",
        help="RESULT_FILE",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATASET = args.DATASET
    PERCENT = args.PERCENT
    N_SAMPLE = args.N_SAMPLE
    TIME_STEP = args.TIME_STEP
    MODEL = args.MODEL
    EMB_DIM = args.EMB_DIM
    LAT_DIM = args.LAT_DIM
    INIT_LR = args.INIT_LR
    BATCH_SIZE = args.BATCH_SIZE
    OPTI = args.OPTI
    ALPHA = args.ALPHA
    EPOCH = args.EPOCH
    ITEM_FEA_TYPE = args.ITEM_FEA_TYPE
    RESULT_FILE = args.RESULT_FILE
    REMARKS = args.REMARKS
    USE_CPU = args.CPU
    SAVE_LOG = args.SAVE_LOG
    LOAD_SAVED = args.LOAD_SAVED
    timestamp_str = str(int(time.time()))

    para_str = (
        DATASET
        + "_"
        + str(PERCENT)
        + "_"
        + str(N_SAMPLE)
        + "_"
        + ITEM_FEA_TYPE
        + "_"
        + str(MODEL)
        + "_"
        + str(TIME_STEP)
        + "_"
        + str(EMB_DIM)
        + "_"
        + str(ALPHA)
        + "_"
        + str(INIT_LR)
        + "_"
        + OPTI
    )
    print(args)
    model_str = "T_VBR_2020_" + para_str + timestamp_str
    output_dir = "./result/" + DATASET + "/"
    sample_dir = "./sample/" + DATASET + "/"
    embedding_dir = "./embedding/" + DATASET + "/"
    log_dir = "./log/" + DATASET + "/"

    model_save_dir = embedding_dir + model_str + ".pt"
    sample_file = sample_dir + "triple_" + str(PERCENT) + "_" + str(N_SAMPLE)
    log_file = log_dir + model_str

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    """
    Logging
    """
    if SAVE_LOG:
        logging = logger.init_std_logger(log_file)

    print(args)

    DEVICE_ID_LIST = GPUtil.getAvailable(
        order="memory", limit=3
    )  # get the fist gpu with the lowest load

    if len(DEVICE_ID_LIST) < 1 or USE_CPU:
        gpu_id = None
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_str = "cpu"
    else:
        gpu_id = DEVICE_ID_LIST[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        device_str = "cuda:" + str(DEVICE_ID_LIST[0])
    print("get divece: " + device_str)

    """
    file paths to be saved
    """
    print("timestamp_str:", timestamp_str)
    print("REMARKS:", REMARKS)

    result_file = output_dir + RESULT_FILE
    print("result will be saved in file:", result_file)
    model_save_dir = embedding_dir + model_str + ".pt"
    print("model will be saved in file:", model_save_dir)
    embedding_save_dir = embedding_dir + model_str
    print("embedding will be saved in file: ", embedding_save_dir)
    sample_file = sample_dir + "triple_" + str(PERCENT) + "_" + str(N_SAMPLE)
    print("python version:", sys.version)
    print("pytorch version:", torch.__version__)

    """
    Loading dataset
    """

    train, test, validate = data_util.load_dataset(data_str=DATASET, percent=PERCENT)
    dump_data_file = (
        sample_dir + "data_" + ITEM_FEA_TYPE + "_" + str(PERCENT) + ".pickle"
    )

    if LOAD_SAVED and os.path.exists(dump_data_file):
        with open(dump_data_file, "rb") as handle:
            print("loading data:", dump_data_file)
            data = pickle.load(handle)

        triple_sampler = Sampler(data.train, PERCENT, N_SAMPLE, sample_dir)
        triple_df = triple_sampler.load_triples_from_file(TIME_STEP)

    else:
        data = data_util.Dataset(
            data_str=DATASET,
            train=train,
            validate=validate,
            test=test,
            item_fea_type=ITEM_FEA_TYPE,
        )
        with open(dump_data_file, "wb") as handle:
            pickle.dump(data, handle)

        triple_sampler = Sampler(data.train, PERCENT, N_SAMPLE, sample_dir)
        triple_df = triple_sampler.sample_by_time(TIME_STEP)

    n_users = data.n_users
    n_items = data.n_items

    monitor = Monitor(1, gpu_id, os.getpid(), live_draw=True)

    """
    init model
    """
    print("init model ", MODEL)
    dvbcr = TVBR(
        triple_df,
        data,
        model_save_dir,
        time_step=TIME_STEP,
        n_neg=5,
        emb_dim=EMB_DIM,
        latent_dim=LAT_DIM,
        batch_size=BATCH_SIZE,
        initial_lr=INIT_LR,
        activator="tanh",
        iteration=EPOCH,
        optimizer_type=OPTI,
        alpha=ALPHA,
        model_str=MODEL,
        show_result=True,
        device_str=device_str,
        monitor=monitor,
    )

    dvbcr.train()
    run_time = monitor.stop()

    """
    Prediction and evalution on test set
    """
    columns = [
        "PERCENT",
        "N_SAMPLE",
        "MODEL",
        "EMB_DIM",
        "INIT_LR",
        "BATCH_SIZE",
        "OPTI",
        "ALPHA",
        "time",
    ]
    result_para = {
        "PERCENT": [PERCENT],
        "N_SAMPLE": [N_SAMPLE],
        "MODEL": [MODEL],
        "ITE_FEA_TYPE": [ITEM_FEA_TYPE],
        "LAT_DIM": [LAT_DIM],
        "EMB_DIM": [EMB_DIM],
        "INIT_LR": [INIT_LR],
        "BATCH_SIZE": [BATCH_SIZE],
        "OPTI": [OPTI],
        "ALPHA": [ALPHA],
        "TIME_STEP": [TIME_STEP],
        "REMARKS": [model_save_dir],
    }

    for i in range(1):
        result = data.evaluate(data.test[i], dvbcr.load_best_model(), t=0)
        print(result)
        result["time"] = [run_time]
        result.update(result_para)
        result_df = pd.DataFrame(result)

        if os.path.exists(result_file):
            print(result_file, " already exists, appending result to it")
            total_result = pd.read_csv(result_file)
            for column in columns:
                if column not in total_result.columns:
                    total_result[column] = "-"
            total_result = total_result.append(result_df)
        else:
            if not os.path.exists(os.path.dirname(result_file)):
                os.mkdir(os.path.dirname(result_file))
            print("create new result_file:", result_file)

            total_result = result_df

        total_result.to_csv(result_file, index=False)