from yacs.config import CfgNode as CN
from utils.utils import log_msg


# simplify cfg
def simplify_cfg(args, cfg):
    dump_cfg = CN()
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.OPTIMIZER = cfg.OPTIMIZER
    if args.server in list(cfg['Server'].keys()):
        dump_cfg['Server'] = CN()
        dump_cfg['Server'][args.server] = CN()
        dump_cfg['Server'][args.server] = cfg['Server'][args.server]
    if args.optim in list(cfg['Optim'].keys()):
        dump_cfg['Optim'] = CN()
        dump_cfg['Optim'][args.optim] = CN()
        dump_cfg['Optim'][args.optim] = cfg['Optim'][args.optim]

    if args.attack_type != 'None':
        dump_cfg['attack'] = CN()
        dump_cfg['attack'].bad_client_rate = cfg['attack'].bad_client_rate
        dump_cfg['attack'].noise_data_rate = cfg['attack'].noise_data_rate
        dump_cfg['attack'][args.attack_type] = cfg['attack'][args.attack_type]
    return dump_cfg


def show_cfg(args, cfg):
    print(log_msg("CONFIG:\n{}".format(cfg.dump()), "INFO"))
    return None


CFG = CN()
'''Federated dataset'''
CFG.DATASET = CN()
CFG.DATASET.dataset = "fl_cifar10"  #
CFG.DATASET.communication_epoch = 50
CFG.DATASET.n_classes = 10

CFG.DATASET.parti_num = 10
CFG.DATASET.online_ratio = 1.0
CFG.DATASET.train_val_domain_ratio = 0.9
CFG.DATASET.val_scale = 256
CFG.DATASET.backbone = "resnet18"
CFG.DATASET.pretrained = False
CFG.DATASET.aug = "weak"
CFG.DATASET.beta = 1.0

'''attack'''
CFG.attack = CN()
CFG.attack.bad_client_rate = 0.3
CFG.attack.noise_data_rate = 0.5

CFG.attack.byzantine = CN()
CFG.attack.byzantine.evils = 'GaussianNoise'  # PairFlip SymFlip AddNoise RandomNoise None MEAN GaussianNoise
CFG.attack.byzantine.dataset_type = 'single_domain'

CFG.attack.byzantine.dev_type = 'std'
CFG.attack.byzantine.lamda = 10.0
CFG.attack.byzantine.threshold_diff = 1e-5

CFG.attack.backdoor = CN()
# evils 可選: base_backdoor, semantic_backdoor, dba_backdoor
CFG.attack.backdoor.evils = 'base_backdoor'  
CFG.attack.backdoor.backdoor_label = 2
CFG.attack.backdoor.trigger_position = [
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 5], [0, 0, 6],
    [0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 4], [0, 2, 5], [0, 2, 6],
    [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 4], [1, 0, 5], [1, 0, 6],
    [1, 2, 0], [1, 2, 1], [1, 2, 2], [1, 2, 4], [1, 2, 5], [1, 2, 6],
    [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 0, 4], [2, 0, 5], [2, 0, 6],
    [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 2, 4], [2, 2, 5], [2, 2, 6],
]
CFG.attack.backdoor.trigger_value = [1.0] * len(CFG.attack.backdoor.trigger_position)
CFG.attack.backdoor.semantic_backdoor_label = 3

# 新增：DBA (Distributed Backdoor Attack) 專用參數
# 此參數決定全域觸發器要拆分成幾份局部觸發器分配給惡意客戶端
CFG.attack.backdoor.dba_parts = 4  

'''Federated OPTIMIZER'''
CFG.OPTIMIZER = CN()
CFG.OPTIMIZER.type = 'SGD'
CFG.OPTIMIZER.momentum = 0.9
CFG.OPTIMIZER.weight_decay = 1e-5
CFG.OPTIMIZER.local_epoch = 10
CFG.OPTIMIZER.local_train_batch = 64
CFG.OPTIMIZER.local_test_batch = 64
CFG.OPTIMIZER.val_batch = 64
CFG.OPTIMIZER.local_train_lr = 1e-3

''''''
CFG.Server = CN()

'''Optim'''
CFG.Optim = CN()

CFG.Optim.FedFish = CN()
