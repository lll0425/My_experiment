import copy
import torch

# Placeholder byzantine attack hooks to keep the pipeline runnable.
# They currently behave as no-ops; replace with real logic if needed.


def attack_dataset(args, cfg, private_dataset):
    """Return dataset unchanged (stub)."""
    return private_dataset


def attack_net_para(net_para, attack_type='none', lamda=1.0):
    """Return parameters unchanged (stub)."""
    return copy.deepcopy(net_para)

