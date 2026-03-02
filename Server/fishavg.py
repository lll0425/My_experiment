from Server.utils.server_methods import ServerMethod
from utils.utils import row_into_parameters
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import torch
import copy
import torch.nn.functional as F
from utils.finch import FINCH


class FishAvg(ServerMethod):
    """
    Ours variant:
    - still uses Fisher-weighted gradient to compute div_score and FINCH clustering
    - but final aggregation is plain FedAvg with reconstructed_freq
    - no Fisher-based parameter correction on clients
    """
    NAME = 'FishAvg'

    def __init__(self, args, cfg):
        super(FishAvg, self).__init__(args, cfg)
        self.learning_rate = self.cfg.OPTIMIZER.local_train_lr
        self.div_score = None
        self.aggregation_weight = None
        self.layer_div_scores = {}

    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        default_net = copy.deepcopy(global_net)
        priloader_list = kwargs['priloader_list']

        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=default_net, freq=freq, except_part=[], global_only=True)

        local_fish_dict = kwargs['local_fish_dict']
        prev_net = copy.deepcopy(global_net)
        vectorize_nets_list = []
        for query_net in nets_list:
            vectorize_net = torch.cat([p.view(-1) for p in query_net.parameters()]).detach()
            vectorize_nets_list.append(vectorize_net)

        prev_vectorize_net = torch.cat([p.view(-1) for p in prev_net.parameters()]).detach()

        grad_list = []
        weight_list = []
        for query_index, _ in enumerate(nets_list):
            grad_list.append((prev_vectorize_net - vectorize_nets_list[query_index]) / self.learning_rate)
            query_fish_dict = local_fish_dict[query_index]
            query_fish = torch.cat([p.view(-1) for p in query_fish_dict.values()]).detach()
            norm_fish = (query_fish - torch.min(query_fish)) / (torch.max(query_fish) - torch.min(query_fish))
            weight_list.append(norm_fish)

        weight_grad_list = []
        for query_index, _ in enumerate(nets_list):
            query_grad = grad_list[query_index]
            query_weight = weight_list[query_index]
            weight_grad_list.append(torch.mul(query_grad, query_weight))

        assert len(weight_grad_list) == len(freq)

        weight_global_grad = torch.zeros_like(weight_grad_list[0])
        for weight_client_grad, client_freq in zip(weight_grad_list, freq):
            weight_global_grad += weight_client_grad * client_freq

        # ===== layer_div_scores (same as Ours) =====
        param_info = []
        idx = 0
        for name, param in prev_net.named_parameters():
            n = param.numel()
            param_info.append((name, idx, idx + n))
            idx += n

        self.layer_div_scores = {}
        for (layer_name, start, end) in param_info:
            layer_scores = []
            for query_index, _ in enumerate(nets_list):
                client_layer = weight_grad_list[query_index][start:end]
                global_layer = weight_global_grad[start:end]
                dist = F.pairwise_distance(
                    client_layer.view(1, -1),
                    global_layer.view(1, -1),
                    p=2
                )
                layer_scores.append(dist.item())
            self.layer_div_scores[layer_name] = layer_scores

        # ===== FINCH clustering on div_score =====
        div_score = []
        for query_index, _ in enumerate(nets_list):
            div_score.append(
                F.pairwise_distance(weight_grad_list[query_index].view(1, -1), weight_global_grad.view(1, -1), p=2))

        div_score = torch.tensor(div_score).view(-1, 1)
        fin = FINCH()
        fin.fit(div_score)

        if len(fin.partitions) == 0:
            reconstructed_freq = freq
        else:
            select_partitions = (fin.partitions)['parition_0']
            evils_center = max(select_partitions['cluster_centers'])
            evils_center_idx = np.where(select_partitions['cluster_centers'] == evils_center)[0]
            evils_idx = select_partitions['cluster_core_indices'][int(evils_center_idx)]
            benign_idx = [i for i in range(len(online_clients_list)) if i not in evils_idx]

            print('benign', benign_idx, 'evil', evils_idx)
            freq[evils_idx] = 0
            reconstructed_freq = freq / sum(freq)

        # ===== final aggregation: plain FedAvg with reconstructed_freq =====
        self.div_score = (div_score)
        self.aggregation_weight = (reconstructed_freq)
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=global_net, freq=reconstructed_freq, except_part=[], global_only=False)
        return freq
