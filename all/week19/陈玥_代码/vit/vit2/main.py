import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,    # 图像大小
    patch_size = 32,     # patch大小（分块的大小）
    num_classes = 1000,  # imagenet数据集1000分类
    dim = 1024,          # position embedding的维度
    depth = 6,           # encoder和decoder中block层数是6
    heads = 16,          # multi-head中head的数量为16
    mlp_dim = 2048,
    dropout = 0.1,       # 
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
