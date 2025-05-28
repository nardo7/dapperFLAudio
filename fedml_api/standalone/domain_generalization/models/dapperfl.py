from typing import cast
import torch
from torch.nn.modules import Module
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from tqdm import tqdm
import copy
from thop import profile
from torchstat import stat
from fedml_api.standalone.domain_generalization.utils.args import *
from fedml_api.standalone.domain_generalization.models.utils.federated_model import (
    FederatedModel,
)


class DapperFL(FederatedModel):
    NAME = "dapperfl"

    def __init__(self, nets_list, args, transform):
        super(DapperFL, self).__init__(nets_list, args, transform)
        self.reg_coeff = args.reg_coeff
        self.alpha_0 = args.alpha
        self.alpha_min = args.alpha_min
        self.epsilon = args.epsilon
        self.pr_strategy = args.pr_strategy
        self.pr_ratios = args.pr_ratios
        self.prune_prob = {
            # Origin model:
            "0": [0, 0, 0, 0],
            "AD": [0, 0, 0, 0],
            "0.1": [0.1, 0.1, 0.1, 0.1],
            "0.2": [0.2, 0.2, 0.2, 0.2],
            "0.3": [0.3, 0.3, 0.3, 0.3],
            "0.4": [0.4, 0.4, 0.4, 0.4],
            "0.5": [0.5, 0.5, 0.5, 0.5],
            "0.6": [0.6, 0.6, 0.6, 0.6],
            "0.7": [0.7, 0.7, 0.7, 0.7],
            "0.8": [0.8, 0.8, 0.8, 0.8],
            "0.9": [0.9, 0.9, 0.9, 0.9],
        }

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        # stat(self.global_net.cpu(), (3, 28, 28))
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        # Aggregation
        self.aggregate_nets(None)

        return None

    def aggregate_nets(self, freq=None):
        global_w = self.global_net.state_dict()
        global_w_prev = copy.deepcopy(global_w)
        for index, net_id in enumerate(self.online_clients):
            net = self.nets_list[net_id]
            net = cast(Module, net)

            # recovery weights
            for name, module in net.named_modules():
                if isinstance(
                    module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)
                ) and torch_prune.is_pruned(module):
                    mask = list(module.named_buffers())[0][1]
                    module.weight += global_w_prev[name + ".weight"] - (
                        global_w_prev[name + ".weight"] * mask
                    )
                    # remove pruning
                    torch_prune.remove(module, "weight")

        super().aggregate_nets(freq)

    def _run_epoch(self, net, train_loader, criterion, optimizer):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            features = net.features(images)
            outputs = net.classifier(features)

            if self.reg_coeff != 0.0:
                loss = criterion(outputs, labels)
                reg = features.norm(dim=1).mean()
                loss = loss + reg * self.reg_coeff
            else:
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(
            net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5
        )
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for i in iterator:
            loss = self._run_epoch(net, train_loader, criterion, optimizer)
            iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)

            if i == 0:
                # print("labels: ", labels)
                # Co-Pruning
                if self.pr_strategy != "0":
                    # calculate co-weights
                    if self.alpha_0 != 0 and self.epoch_index != 0:
                        alpha_k = (1 - self.epsilon) ** self.epoch_index * self.alpha_0
                        if alpha_k < self.alpha_min:
                            alpha_k = self.alpha_min
                            # self.alpha_0 = 0
                        for [(name0, m0), (name1, m1)] in zip(
                            self.global_net.named_modules(),
                            self.nets_list[index].named_modules(),
                        ):
                            if isinstance(
                                m1, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)
                            ) and torch_prune.is_pruned(m1):
                                m1.weight.data = (
                                    alpha_k * m0.weight.data.clone()
                                    + (1 - alpha_k) * m1.weight.data.clone()
                                )
                    # pruning
                    if "res" in self.nets_list[index].name:
                        if self.args.pr_strategy == "iterative":
                            self.nets_list[index] = self._res_pruning_v2(
                                index,
                                self.nets_list[index],
                                train_loader,
                                criterion,
                                optimizer,
                            )
                        else:
                            self.nets_list[index] = self._prepare_run_pruning(
                                index, self.nets_list[index]
                            )

    def _model_sparsity(model):
        total_zeros = 0
        total_elements = 0
        for _, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if hasattr(module, "weight"):
                    weight = module.weight.data
                    total_zeros += torch.sum(weight == 0).item()
                    total_elements += weight.numel()
        sparsity = 100.0 * total_zeros / total_elements if total_elements > 0 else 0.0
        print(
            f"Model sparsity: {sparsity:.2f}% ({total_zeros}/{total_elements} weights are zero)"
        )
        return sparsity / 100.0

    def _res_pruning_v2(
        self,
        index: int,
        net: nn.Module,
        train_loader,
        criterion,
        optimizer,
        goal_sparsity=0.2,
    ):
        print(f"Pruning local model {index} iteratively")
        model_sparsity = self._model_sparsity(net)
        print(f"Model sparsity: {model_sparsity}")
        while model_sparsity < goal_sparsity:
            net = self._res_pruning(index, net, 0.2)
            model_sparsity = self._model_sparsity(net)
            print(f"Model pruned. Sparsity: {model_sparsity}")
            if model_sparsity >= goal_sparsity:
                break
            print(f"Fine-tuning local model {index} after pruning")
            self._run_epoch(net, train_loader, criterion, optimizer)

        return net

    def _prepare_run_pruning(self, index, net):
        if self.pr_strategy == "AD":
            pr_strategy = self.pr_ratios[
                index % len(self.pr_ratios)
            ]  # get pruning ratio of specific client
            pr_prob = self.prune_prob[pr_strategy]  # get pruning ratios for layers
            self.prune_prob["AD"] = self.prune_prob[
                pr_strategy
            ]  # copy pruning ratios for layers
        else:
            pr_prob = self.prune_prob[self.pr_strategy]  # get pruning ratios for layers

        print("Prune local model %s, pr_ratio = %s" % (index, pr_prob))
        net = self._res_pruning(net, pr_prob)
        return net

    def _res_pruning(self, net, pr_prob=None):
        conv_count = 0
        down_count = 0  # r10's stage1 has no 'downsample' layer
        for name, module in net.named_modules():
            # if isinstance(
            #     module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)
            # ) and torch_prune.is_pruned(module):
            #     torch_prune.remove(module, "weight")
            if isinstance(module, nn.Conv2d):
                if conv_count == 0:  # The first conv layer in resnet.
                    conv_count += 1
                    continue
                if "shortcut" in name:
                    # The first downsample conv layer, only prune 'out_planes'.
                    if down_count == 0:
                        # Use the pruning probability in stage1(pr_prob[0]) to prune 'out_planes'(dim=0).
                        torch_prune.ln_structured(
                            module,
                            name="weight",
                            amount=pr_prob[down_count],
                            n=1,
                            dim=0,
                        )
                        down_count += 1
                    else:  # The other downsample conv layer.
                        torch_prune.ln_structured(
                            module,
                            name="weight",
                            amount=pr_prob[down_count - 1],
                            n=1,
                            dim=1,
                        )
                        torch_prune.ln_structured(
                            module,
                            name="weight",
                            amount=pr_prob[down_count],
                            n=1,
                            dim=0,
                        )
                        down_count += 1
                    conv_count += 1
                    continue
                else:  # Normal conv layers in blocks.
                    # Stage1's 1st conv layer.
                    if conv_count == 1:
                        # Pruning 'out_planes'.
                        torch_prune.ln_structured(
                            module, name="weight", amount=pr_prob[0], n=1, dim=0
                        )
                    # Stage1's other conv layers.
                    else:
                        torch_prune.ln_structured(
                            module, name="weight", amount=pr_prob[0], n=1, dim=1
                        )
                        torch_prune.ln_structured(
                            module, name="weight", amount=pr_prob[0], n=1, dim=0
                        )
                    conv_count += 1
                    continue

            elif isinstance(module, nn.BatchNorm2d):
                # 'conv_count' in nn.BatchNorm2d is 1 bigger than nn.Conv2d.
                if conv_count == 1:  # The 1st bn in resnet.
                    continue
                torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])

            elif isinstance(module, nn.Linear):
                torch_prune.ln_structured(
                    module, name="weight", amount=pr_prob[-1], n=2, dim=1
                )

        # stat(glb_model, (3, 32, 32))
        # print(list(glb_model.named_buffers()))
        return net
