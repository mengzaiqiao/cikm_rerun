import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm


class Sampler(object):
    def __init__(self, df_train, percent, N_SAMPLE, sample_dir, dump=True):
        self.sample_file = sample_dir + "triple_" + str(percent) + "_" + str(N_SAMPLE)
        self.df_train = df_train
        self.sample_dir = sample_dir
        self.percent = percent
        self.N_SAMPLE = N_SAMPLE
        self.dump = dump
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        print("Initialize Sampler!")
        sys.stdout.flush()

    def sample(self):
        print("preparing training triples ... ")
        self.dataTrain = (
            self.df_train.groupby(["order_ids", "user_ids"])["item_ids"]
            .apply(list)
            .reset_index()
        )
        self.dataTrain.rename(
            columns={"user_ids": "UID", "order_ids": "TID", "item_ids": "PID"},
            inplace=True,
        )
        n_orders = self.dataTrain.shape[0]
        sampled_index = np.random.choice(n_orders, size=self.N_SAMPLE)
        sampled_order = self.dataTrain.iloc[sampled_index].reset_index()

        process_bar = tqdm(range(self.N_SAMPLE))
        res = []
        for i in process_bar:
            _index, _tid, _uid, _items = sampled_order.iloc[i]
            _i, _j = np.random.choice(_items, size=2)
            res.append([int(_uid), int(_i), int(_j)])
        print("done!")
        data_dic = {}
        res = np.array(res)
        data_dic["UID"] = res[:, 0]
        data_dic["PID1"] = res[:, 1]
        data_dic["PID2"] = res[:, 2]
        triple_df = pd.DataFrame(data_dic)
        if self.dump:
            triple_file = self.sample_file + ".csv"
            triple_df.to_csv(triple_file, index=False)
        return triple_df

    def sample_by_time(self, time_step):
        if time_step == 0:
            return self.sample()
        print("preparing training triples ... ")
        self.dataTrain = (
            self.df_train.groupby(["order_ids", "user_ids"])["item_ids"]
            .apply(list)
            .reset_index()
        )
        dataTrain_timestep = (
            self.df_train.groupby(["order_ids"])["timestamps"]
            .apply(lambda a: a.mean())
            .reset_index()
        )
        self.dataTrain = self.dataTrain.merge(dataTrain_timestep)
        self.dataTrain = self.dataTrain.sort_values(by="timestamps")
        self.dataTrain.rename(
            columns={"user_ids": "UID", "order_ids": "TID", "item_ids": "PID"},
            inplace=True,
        )
        n_orders = self.dataTrain.shape[0]
        n_orders_per_t = int(n_orders / time_step)
        n_sample_per_t = int(self.N_SAMPLE / time_step)
        process_bar = tqdm(range(time_step))
        rest_baskets = n_orders - time_step * n_orders_per_t
        res = []
        for t in process_bar:
            if t != 0:
                index_start = t * n_orders_per_t + rest_baskets
            else:
                index_start = 0
            index_end = (t + 1) * n_orders_per_t + rest_baskets
            sampled_index = np.random.choice(
                np.arange(index_start, index_end), size=n_sample_per_t,
            )
            sampled_order = self.dataTrain.iloc[sampled_index]
            for _, row in sampled_order.iterrows():
                _uid, _tid, _items = row["UID"], row["TID"], row["PID"]
                _i, _j = np.random.choice(_items, size=2)
                res.append([int(_uid), int(_i), int(_j), int(t)])
        res = np.array(res)
        data_dic = {}
        data_dic["UID"] = res[:, 0]
        data_dic["PID1"] = res[:, 1]
        data_dic["PID2"] = res[:, 2]
        data_dic["T"] = res[:, 3]
        triple_df = pd.DataFrame(data_dic)
        if self.dump:
            triple_file = self.sample_file + "_" + str(time_step) + ".csv"
            triple_df.to_csv(triple_file, index=False)
        return triple_df

    def load_triples_from_file(self, time_step):
        if time_step == 0:
            triple_file = self.sample_file + ".csv"
            print("load_triples_from_file:", triple_file)
        else:
            triple_file = self.sample_file + "_" + str(time_step) + ".csv"
            print("load_triples_from_file:", triple_file)
        return pd.read_csv(triple_file)