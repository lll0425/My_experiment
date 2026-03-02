import numpy as np
from Attack.backdoor.utils import backdoor_attack
from Attack.byzantine.utils import attack_dataset
from Server import get_server_method, Server_NAME
from Datasets.federated_dataset.single_domain import single_domain_dataset_name, get_single_domain_dataset
from Optims import Fed_Optim_NAMES, get_fed_method
from utils.conf import set_random_seed, config_path, log_path
from Datasets.federated_dataset.multi_domain import multi_domain_dataset_name, get_multi_domain_dataset
from Backbones import get_private_backbones
from utils.cfg import CFG as cfg, simplify_cfg, show_cfg
from utils.utils import ini_client_domain
from argparse import ArgumentParser
from utils.training import train
import datetime
import socket
import uuid
import setproctitle
import argparse
import os


def parse_args():
    parser = ArgumentParser(description='Federated Learning', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--task', type=str, default='label_skew')
    '''
    label_skew:   fl_cifar10 fl_mnist fl_fashionmnist
    '''
    parser.add_argument('--dataset', type=str, default='fl_fashionmnist',
                        help='Which scenario to perform experiments on.')

    parser.add_argument('--attack_type', type=str, default='backdoor')

    '''
    Federated Method:  fedfish
    '''
    parser.add_argument('--optim', type=str, default='fedfish',
                        help='Federated Method name.', choices=Fed_Optim_NAMES)

    parser.add_argument('--rand_domain_select', type=bool, default=False, help='The Local Domain Selection')
    '''
    Aggregations Strategy Hyper-Parameter
    '''
    parser.add_argument('--server', type=str, default='Ours', choices=Server_NAME, help='The Option for averaging strategy')

    parser.add_argument('--seed', type=int, default=0, help='The random seed.')

    parser.add_argument('--csv_log', action='store_true', default=True, help='Enable csv logging')
    parser.add_argument('--result_path', default=log_path(), help='Enable csv logging')
    parser.add_argument('--csv_name', type=str, default=None, help='Predefine the csv name')
    parser.add_argument('--save_checkpoint', action='store_true', default=False)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    # 路徑保持不動
    cfg_dataset_path = os.path.join(config_path(), 'Datasets', args.task, args.dataset, 'Default.yaml')
    cfg.merge_from_file(cfg_dataset_path)

    cfg.merge_from_list(args.opts)

    particial_cfg = simplify_cfg(args, cfg)

    show_cfg(args, particial_cfg)
    if args.seed is not None:
        set_random_seed(args.seed)

    '''
    Loading the dataset
    '''
    if args.dataset in multi_domain_dataset_name:
        private_dataset = get_multi_domain_dataset(args, particial_cfg)
    elif args.dataset in single_domain_dataset_name:
        private_dataset = get_single_domain_dataset(args, particial_cfg)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ====== 關鍵修正：先建立 data loaders（train/val/test）再做 attack ======
    # 依不同 task / dataset 類型決定怎麼建立 loaders
    # - label_skew: 一般 single-domain 的 get_data_loaders() 會建立 train_loaders/val_loader/test_loader
    # - domain_skew: 常見設計是要傳 client_domain_list 或使用 dataset.domain_list
    # - multi-domain: 通常也要 client_domain_list 才能正確切 domain
    client_domain_list = None

    if args.task == 'label_skew':
        # 不需要 domain 指派，直接建 loaders
        if hasattr(private_dataset, "get_data_loaders"):
            private_dataset.get_data_loaders()
        else:
            raise AttributeError(f"{type(private_dataset).__name__} has no method get_data_loaders()")

    else:
        # domain_skew 或其他需要 domain 指派的 task
        # 先準備 client_domain_list，再把它帶入 get_data_loaders（若該資料集支援）
        domains_list = list(range(particial_cfg.DATASET.parti_num))

        # 優先用 dataset.domain_list（若存在），否則用 0..parti_num-1
        if hasattr(private_dataset, "domain_list") and private_dataset.domain_list is not None:
            domains_for_init = private_dataset.domain_list
        else:
            domains_for_init = domains_list

        client_domain_list = ini_client_domain(args.rand_domain_select, domains_for_init, particial_cfg.DATASET.parti_num)

        if hasattr(private_dataset, "get_data_loaders"):
            try:
                private_dataset.get_data_loaders(client_domain_list)
            except TypeError:
                # 有些資料集 get_data_loaders 不收參數
                private_dataset.get_data_loaders()
        else:
            raise AttributeError(f"{type(private_dataset).__name__} has no method get_data_loaders()")

    # 防呆：避免你這次遇到的「train_loaders 空」悄悄發生
    if not hasattr(private_dataset, "train_loaders") or private_dataset.train_loaders is None or len(private_dataset.train_loaders) == 0:
        raise RuntimeError(
            "private_dataset.train_loaders is empty after get_data_loaders(). "
            "Please check dataset.get_data_loaders() and partition logic."
        )
    # ====== 修正結束 ======

    bad_scale = int(particial_cfg.DATASET.parti_num * particial_cfg['attack'].bad_client_rate)
    good_scale = particial_cfg.DATASET.parti_num - bad_scale
    client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()

    # attack_type 分流：先確保 loaders 已存在，再做資料汙染/攻擊處理
    if args.attack_type == 'backdoor':
        backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=True)
        backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=False)
    elif args.attack_type == 'byzantine':
        # 依你原本的 import：Attack.byzantine.utils.attack_dataset
        # 這通常會在資料或更新層做處理（視你專案實作）
        attack_dataset(args, particial_cfg, client_type, private_dataset)
    else:
        raise ValueError(f"Unknown attack_type: {args.attack_type}")

    '''
    Loading the Private Backbone
    '''
    priv_backbones = get_private_backbones(particial_cfg)

    '''
    Loading the Federated Optimizer
    '''
    # 如果前面 task != label_skew 已經算過 client_domain_list，這裡沿用
    if client_domain_list is None:
        domains_list = list(range(particial_cfg.DATASET.parti_num))
        client_domain_list = ini_client_domain(args.rand_domain_select, domains_list, particial_cfg.DATASET.parti_num)

    fed_method = get_fed_method(priv_backbones, client_domain_list, args, particial_cfg)

    ''''
    Loading the Federated Aggregation
    '''
    fed_server = get_server_method(args, particial_cfg)

    train(fed_method, fed_server, private_dataset, args, particial_cfg)


if __name__ == '__main__':
    main()
