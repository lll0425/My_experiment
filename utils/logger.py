import copy
import os
import csv

from utils.utils import create_if_not_exists

import yaml
from yacs.config import CfgNode as CN

except_args = ['result_path', 'csv_log', 'csv_name', 'device_id', 'seed', 'tensorboard', 'conf_jobnum', 'conf_timestamp', 'conf_host', 'opts']


class CsvWriter:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.model_path = self.model_folder_path()
        self.para_path = self.write_para()
        print(self.para_path)

    def model_folder_path(self):
        if self.args.attack_type == 'None':
            model_path = os.path.join(self.args.result_path, self.args.task, self.args.attack_type, self.args.dataset, str(self.cfg.DATASET.beta), self.args.server,
                                      self.args.optim)
        else:
            model_path = os.path.join(self.args.result_path, self.args.task, self.cfg.attack[self.args.attack_type].evils, str(self.cfg.attack.bad_client_rate),
                                      self.args.dataset, str(self.cfg.DATASET.beta), self.args.server, self.args.optim)
        create_if_not_exists(model_path)
        return model_path

    def write_metric(self, metric_list, epoch_index, name):
        metric_path = os.path.join(self.para_path, name + '.csv')
        if epoch_index != 0:
            write_type = 'a'
        else:
            write_type = 'w'
        with open(metric_path, write_type) as result_file:
            result_file.write(str(epoch_index) + ':' + '\n')
            for i in range(len(metric_list)):
                result_file.write(str(metric_list[i]) + ',')
            result_file.write('\n')

    def write_layer_metric(self, layer_scores: dict, epoch_index: int, name='layer_div_score'):
        """
        layer_scores: dict[layer_name] -> list of scores per client
        Output format:
        epoch,layer,client_0,client_1,...
        """
        metric_path = os.path.join(self.para_path, name + '.csv')
        write_header = not os.path.exists(metric_path)

        with open(metric_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # infer client count from first entry
            first_layer = next(iter(layer_scores))
            n_clients = len(layer_scores[first_layer])
            if write_header:
                header = ['epoch', 'layer'] + [f'client_{i}' for i in range(n_clients)]
                writer.writerow(header)
            for layer, scores in layer_scores.items():
                writer.writerow([epoch_index, layer] + [float(s) for s in scores])

    def write_layer_metric_split(self, layer_scores: dict, epoch_index: int, prefix='layer'):
        """
        Write one CSV per layer.
        File name: <prefix>_<layer_name>.csv
        Each row: epoch, client_0, client_1, ...
        """
        if not layer_scores:
            return
        for layer, scores in layer_scores.items():
            safe_name = layer.replace('/', '_').replace('\\', '_')
            metric_path = os.path.join(self.para_path, f"{prefix}_{safe_name}.csv")
            write_header = not os.path.exists(metric_path)
            with open(metric_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    header = ['epoch'] + [f'client_{i}' for i in range(len(scores))]
                    writer.writerow(header)
                writer.writerow([epoch_index] + [float(s) for s in scores])

    def write_acc(self, acc, name, mode='ALL'):
        if mode == 'ALL':
            acc_path = os.path.join(self.para_path, name + '_all_acc.csv')
            self.write_all_acc(acc_path, acc)
        elif mode == 'MEAN':
            mean_acc_path = os.path.join(self.para_path, name + '_mean_acc.csv')
            self.write_mean_acc(mean_acc_path, acc)

    def cfg_to_dict(self, cfg):
        d = {}
        for k, v in cfg.items():
            if isinstance(v, CN):
                d[k] = self.cfg_to_dict(v)
            else:
                d[k] = v
        return d

    def dict_to_cfg(self, d):
        cfg = CN()
        for k, v in d.items():
            if isinstance(v, dict):
                cfg[k] = self.dict_to_cfg(v)
            else:
                cfg[k] = v
        return cfg

    def write_para(self) -> None:
        """Always write a fresh args/cfg set.
        - If --csv_name 指定，使用該名稱；若重名則自動加編號後綴。
        - 若未指定，使用 para1/para2/… 依序遞增，不重用舊資料夾。
        """
        args = vars(copy.deepcopy(self.args))
        cfg = copy.deepcopy(self.cfg)

        # 移除不需記錄的欄位並轉成字串
        for cc in except_args:
            if cc in args:
                del args[cc]
        for key, value in args.items():
            args[key] = str(value)

        def unique_path(base_path: str) -> str:
            """如果路徑已存在，自動加 _1, _2... 直到不存在為止。"""
            path = base_path
            idx = 1
            while os.path.exists(path):
                path = base_path + f"_{idx}"
                idx += 1
            return path

        # 決定輸出資料夾
        if self.args.csv_name:
            base = os.path.join(self.model_path, self.args.csv_name)
            path = unique_path(base)
        else:
            i = 1
            while True:
                candidate = os.path.join(self.model_path, f'para{i}')
                if not os.path.exists(candidate):
                    path = candidate
                    break
                i += 1

        create_if_not_exists(path)

        # 寫 args 與 cfg
        columns = list(args.keys())
        args_path = os.path.join(path, 'args.csv')
        cfg_path = os.path.join(path, 'cfg.yaml')
        with open(args_path, 'w', newline='') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            writer.writeheader()
            writer.writerow(args)
        with open(cfg_path, 'w') as f:
            f.write(yaml.dump(self.cfg_to_dict(cfg)))

        return path

    def write_mean_acc(self, mean_path, acc_list):
        if os.path.exists(mean_path):
            with open(mean_path, 'a') as result_file:
                for i in range(len(acc_list)):
                    result = acc_list[i]
                    result_file.write(str(result))
                    if i != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
        else:
            with open(mean_path, 'w') as result_file:
                for epoch in range(self.cfg.DATASET.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
                for i in range(len(acc_list)):
                    result = acc_list[i]
                    result_file.write(str(result))
                    if i != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

    def write_all_acc(self, all_path, all_acc_list):
        if os.path.exists(all_path):
            with open(all_path, 'a') as result_file:
                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(key + ',')
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')
        else:
            with open(all_path, 'w') as result_file:
                result_file.write('domain,')
                for epoch in range(self.cfg.DATASET.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(key + ',')
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')
