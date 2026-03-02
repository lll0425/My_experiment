import copy
import numpy as np
import torch
import torch.nn.functional as F

from Server.utils.server_methods import ServerMethod


class Equal(ServerMethod):
    NAME = 'Equal'

    def __init__(self, args, cfg):
        super(Equal, self).__init__(args, cfg)
        self.learning_rate = self.cfg.OPTIMIZER.local_train_lr
        self.layer_div_scores = {}

    def weight_calculate(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        freq = [1 / len(online_clients_list) for _ in range(len(online_clients_list))]
        return freq

    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # 分層 FINCH 分數計算（與 Ours 一致，方便比較）
        self.layer_div_scores = {}
        local_fish_dict = kwargs.get('local_fish_dict')
        if local_fish_dict:
            bad_client_rate = self.cfg.attack.bad_client_rate if hasattr(self.cfg, 'attack') else 0.0
            bad_scale = int(self.cfg.DATASET.parti_num * bad_client_rate)
            good_scale = self.cfg.DATASET.parti_num - bad_scale
            client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()

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
                if not client_type[query_index]:
                    query_fish = torch.randn_like(query_fish)

                min_val = torch.min(query_fish)
                max_val = torch.max(query_fish)
                if max_val == min_val:
                    norm_fish = torch.zeros_like(query_fish)
                else:
                    norm_fish = (query_fish - min_val) / (max_val - min_val)
                weight_list.append(norm_fish)

            weight_grad_list = []
            for query_index, _ in enumerate(nets_list):
                query_grad = grad_list[query_index]
                query_weight = weight_list[query_index]
                weight_grad_list.append(torch.mul(query_grad, query_weight))

            weight_global_grad = torch.zeros_like(weight_grad_list[0])
            for weight_client_grad, client_freq in zip(weight_grad_list, freq):
                weight_global_grad += weight_client_grad * client_freq

            param_info = []
            index = 0
            for name, param in prev_net.named_parameters():
                param_number = param.numel()
                param_info.append({'name': name, 'start': index, 'end': index + param_number})
                index += param_number

            for p_info in param_info:
                layer_name = p_info['name']
                start, end = p_info['start'], p_info['end']

                layer_div_score = []
                for query_index, _ in enumerate(nets_list):
                    client_grad_layer = weight_grad_list[query_index][start:end]
                    global_grad_layer = weight_global_grad[start:end]

                    distance = F.pairwise_distance(
                        client_grad_layer.view(1, -1),
                        global_grad_layer.view(1, -1), p=2)
                    layer_div_score.append(distance.item())
                self.layer_div_scores[layer_name] = layer_div_score

        # FedAvg 聚合（等權重）
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=global_net, freq=freq, except_part=[], global_only=False)
        return freq
