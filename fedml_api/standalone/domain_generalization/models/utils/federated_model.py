import copy

import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
import torch
import torchvision
from argparse import Namespace
from fedml_api.standalone.domain_generalization.utils.conf import get_device
from fedml_api.standalone.domain_generalization.utils.conf import checkpoint_path
from fedml_api.standalone.domain_generalization.utils.util import create_if_not_exists
import os


class FederatedModel(nn.Module):
    """
    Federated learning model.
    """

    NAME = None
    N_CLASS = None

    def __init__(
        self,
        nets_list: list[torch.nn.modules.Module],
        args: Namespace,
        transform: torchvision.transforms,
    ) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform

        # For Online
        self.random_state = np.random.RandomState(args.seed)
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net: torch.nn.modules.Module | None = None
        self.device = get_device(device_id=self.args.device_id)

        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.trainloaders: list[torch.utils.data.DataLoader] | None = None
        self.testlodaers: list[torch.utils.data.DataLoader] | None = None

        self.epoch_index = 0  # Save the Communication Index

        self.checkpoint_path = checkpoint_path() + self.args.dataset + "/" + "/"
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def col_update(self, communication_idx, publoader):
        pass

    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, "pretrain")
                save_path = os.path.join(pretrain_path, str(j) + ".ckpt")
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            prev_net = prev_nets_list[net_id]
            prev_net.load_state_dict(net_para)

    @torch.no_grad()
    def aggregate_nets(self, freq=None):
        global_net = self.global_net
        nets_list = self.nets_list

        online_clients = self.online_clients
        global_w = {k: v.clone() for k, v in self.global_net.state_dict().items()}

        if self.args.averaing == "weight":
            online_clients_dl = [
                self.trainloaders[online_clients_index]
                for online_clients_index in online_clients
            ]
            online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
            online_clients_all = np.sum(online_clients_len)
            freq = online_clients_len / online_clients_all
        else:
            # if freq == None:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]

        first = True
        for index, net_id in enumerate(online_clients):
            net = nets_list[net_id]

            net_para = net.state_dict()
            if first:
                first = False
                for key in net_para:
                    if "mask" in key:
                        continue
                    global_w[key] = net_para[key].clone() * freq[index]
            else:
                for key in net_para:
                    if "mask" in key:
                        continue
                    global_w[key] += net_para[key].clone() * freq[index]

        missmatch = global_net.load_state_dict(global_w, strict=False)
        if missmatch.missing_keys:
            print("Missing keys in global model:", missmatch.missing_keys)
        if missmatch.unexpected_keys:
            print("Unexpected keys in global model:", missmatch.unexpected_keys)
        # check if the global model is equal to global_w
        # equal = self.models_equal(global_net, global_w)
        # print(f"Global model equal to global_w: {equal}")

        for i, net in enumerate(nets_list):
            missmatch = net.load_state_dict(global_net.state_dict(), strict=False)
            if missmatch.missing_keys:
                print(f"Missing keys in model {i}:", missmatch.missing_keys)
            if missmatch.unexpected_keys:
                print(f"Unexpected keys in model {i}:", missmatch.unexpected_keys)
            # with torch.no_grad():
            #     for global_param, net_param in zip(
            #         self.global_net.parameters(), net.parameters()
            #     ):
            #         net_param[:] = global_param.detach().clone()
            equal = self.models_equal(self.nets_list[i], global_net)

    def models_equal(self, model_a: nn.Module, model_b: nn.Module) -> bool:
        """
        Check if two models have equal weights (all parameters are equal).
        """
        state_dict_a = (
            model_a.state_dict() if isinstance(model_a, nn.Module) else model_a
        )
        state_dict_b = (
            model_b.state_dict() if isinstance(model_b, nn.Module) else model_b
        )
        # iterate first over keys in state_dict_a
        for key in state_dict_a:
            if key not in state_dict_b:
                print(f"Key {key} not found in model_b")
                return False
            if not torch.allclose(state_dict_a[key], state_dict_b[key], atol=1e-6):
                print(
                    f"Parameters for key {key} are not equal: "
                    f"model_a {state_dict_a[key].shape}, model_b {state_dict_b[key].shape}"
                )
                return False
        # iterate over keys in state_dict_b to check for extra keys
        for key in state_dict_b:
            if key not in state_dict_a:
                print(f"Key {key} not found in model_a")
                return False
        return True

    def compare_model_structure(self, model_a: nn.Module, model_b: nn.Module):
        """
        Compare the structure (parameter names and shapes) of two models.
        Prints differences in keys and shapes.
        """
        state_dict_a = model_a.state_dict()
        state_dict_b = model_b.state_dict()

        keys_a = set(state_dict_a.keys())
        keys_b = set(state_dict_b.keys())

        only_in_a = keys_a - keys_b
        only_in_b = keys_b - keys_a
        common_keys = keys_a & keys_b

        if only_in_a:
            print("Parameters only in model_a:")
            for k in only_in_a:
                print(f"  {k} : {state_dict_a[k].shape}")
        if only_in_b:
            print("Parameters only in model_b:")
            for k in only_in_b:
                print(f"  {k} : {state_dict_b[k].shape}")

        for k in common_keys:
            if state_dict_a[k].shape != state_dict_b[k].shape:
                print(
                    f"Shape mismatch for '{k}': model_a {state_dict_a[k].shape}, model_b {state_dict_b[k].shape}"
                )
