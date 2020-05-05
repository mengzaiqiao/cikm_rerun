import sys

sys.path.append("../")

import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import torch
import time
from beta_rec.train_engine import TrainEngine
from beta_rec.models.gcn import GCN_SEngine
from beta_rec.models.mlp_gnn import MLPEngine
from beta_rec.models.ncf_gnn import NeuMFEngine
from beta_rec.utils.common_util import update_args
from beta_rec.datasets.nmf_data_utils import SampleGenerator
from beta_rec.utils.constants import MAX_N_UPDATE
from beta_rec.utils.monitor import Monitor


def parse_args():
    """ Parse args from command line

        Returns:
            args object.
    """
    parser = argparse.ArgumentParser(description="Run NGCF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/ncf_gnn_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument("--lr", nargs="?", type=float, help="Initialize learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    return parser.parse_args()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GCN_train(TrainEngine):
    """ An instance class from the TrainEngine base class

    """

    def __init__(self, config):
        """Constructor

        Args:
            config (dict): All the parameters for the model
        """

        self.config = config
        super(GCN_train, self).__init__(self.config)
        self.load_dataset()
        self.build_data_loader()
        common_config = self.config.copy()
        common_config.pop("gcn_config")
        common_config.pop("mlp_config")
        common_config.pop("neumf_config")
        self.config["gcn_config"].update(common_config)
        self.config["mlp_config"].update(common_config)
        self.config["neumf_config"].update(common_config)

    def check_early_stop(self, engine, model_dir, epoch):
        """ Check if early stop criterion is triggered
        Save model if previous epoch have already obtained better result

        Args:
            epoch (int): epoch num

        Returns:
            True: if early stop criterion is triggered
            False: else

        """
        if epoch > 0 and self.eval_engine.n_no_update == 0:
            # save model if previous epoch have already obtained better result
            engine.save_checkpoint(model_dir=model_dir)

        if self.eval_engine.n_no_update >= MAX_N_UPDATE:
            # stop training if early stop criterion is triggered
            print(
                "Early stop criterion triggered, no performance update for {:} times".format(
                    MAX_N_UPDATE
                )
            )
            return True
        return False

    def build_data_loader(self):
        # ToDo: Please define the directory to store the adjacent matrix
        user_fea_norm_adj, item_fea_norm_adj = self.dataset.make_fea_sim_mat()
        self.sample_generator = SampleGenerator(ratings=self.dataset.train)
        self.config["user_fea_norm_adj"] = sparse_mx_to_torch_sparse_tensor(
            user_fea_norm_adj
        )
        self.config["item_fea_norm_adj"] = sparse_mx_to_torch_sparse_tensor(
            item_fea_norm_adj
        )
        self.config["num_batch"] = self.dataset.n_train // config["batch_size"] + 1
        self.config["n_users"] = self.dataset.n_users
        self.config["n_items"] = self.dataset.n_items

    def _train(self, engine, train_loader, save_dir):
        self.eval_engine.flush()
        epoch_bar = tqdm(range(self.config["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            if self.check_early_stop(engine, save_dir, epoch):
                break
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            self.eval_engine.train_eval(
                self.dataset.valid[0], self.dataset.test[0], engine.model, epoch
            )

    def train(self):
        self.monitor = Monitor(
            log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
        )

        # Train GCN
        self.engine = GCN_SEngine(self.config["gcn_config"])
        train_loader = self.dataset
        self.gcn_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["gcn_config"]["save_name"]
        )
        self._train(
            engine=self.engine, train_loader=train_loader, save_dir=self.gcn_save_dir
        )
        while self.eval_engine.n_worker:
            print(f"Wait 15s for the complete of eval_engine.n_worker")
            time.sleep(15)  # wait the
        # Train MLP
        train_loader = self.sample_generator.instance_a_train_loader(
            self.config["num_negative"], self.config["batch_size"]
        )
        self.engine = MLPEngine(
            self.config["mlp_config"], gcn_config=self.config["gcn_config"]
        )
        self.mlp_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["mlp_config"]["save_name"]
        )
        self._train(
            engine=self.engine, train_loader=train_loader, save_dir=self.mlp_save_dir
        )

        while self.eval_engine.n_worker:
            print(f"Wait 15s for the complete of eval_engine.n_worker")
            time.sleep(15)  # wait the

        # Train ncf
        self.engine = NeuMFEngine(
            self.config["neumf_config"],
            mlp_config=self.config["mlp_config"],
            gcn_config=self.config["gcn_config"],
        )
        self.neumf_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["neumf_config"]["save_name"]
        )
        self._train(
            engine=self.engine, train_loader=train_loader, save_dir=self.neumf_save_dir,
        )

        while self.eval_engine.n_worker:
            print(f"Wait 15s for the complete of eval_engine.n_worker")
            time.sleep(15)  # wait the
        self.config["run_time"] = self.monitor.stop()

    def test(self):
        self.engine.resume_checkpoint(model_dir=self.neumf_save_dir)
        super(GCN_train, self).test()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    config_file = args.config_file
    with open(config_file) as config_params:
        config = json.load(config_params)
    update_args(config, args)
    gcn = GCN_train(config)
    gcn.train()
    gcn.test()
