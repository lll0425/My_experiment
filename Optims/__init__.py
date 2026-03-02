from Optims.fedfish import FedFish

# Registry of available federated optimizers
Fed_Optim_NAMES = ['fedfish']

def get_fed_method(priv_backbones, client_domain_list, args, cfg):
    name = args.optim.lower()
    if name == 'fedfish':
        return FedFish(priv_backbones, client_domain_list, args, cfg)
    raise ValueError(f'Unsupported optim method: {args.optim}')

