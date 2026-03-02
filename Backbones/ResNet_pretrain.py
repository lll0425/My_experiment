from Backbones.ResNet import resnet18

# Minimal fallback: reuse vanilla resnet18 (no actual pretrain weights here).
def resnet18_pretrained(cfg):
    return resnet18(cfg)

