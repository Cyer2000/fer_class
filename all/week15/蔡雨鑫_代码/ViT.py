import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange,repeat
from einops.layers.torch import Rearrange

#判断t是否是元组，如果是，直接返回t；如果不是，则将t复制为元组(t, t)再返回
def pair(t):
    return t if isinstance(t,tuple) else (t,t)

#对应最底下的Norm层。dim是维度，fn是要预先要进行的处理函数
class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x),**kwargs)

#FeedForward层由全连接层，配合激活函数GELU和Dropout实现，对应模型中的MLP。
#参数dim和hidden_dim分别是输入输出的维度和中间层为维度，dropout是dropout操作的概率参数p
class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

#transformer中的核心部件，对应Multi-Head Attention
#参数heads是多头自注意力的头的数目，dim_head是每个头的维度
class Attention(nn.Module):
    def __init__(self,dim,heads = 8, dim_head =64 , dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads   #想问一下这个维度指什么阿，感觉没有直观的概念
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        #开根号
        self.scale = dim_head ** -0.5

        #对某维度的每一行进行softmax
        self.attend = nn.Softmax(dim = -1)
        #全连接层
        self.to_qkv = nn.Linear(dim,inner_dim * 3,bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self,x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q, k ,v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d ', h=h),qkv)
        #在查询q和键值k之间执行矩阵乘法，即最后一个轴上的求和
        dots = einsum('b h i d, b h j d -> b h i j',q,k) * self.scale

        attn = self.attend(dots)
        #在q和k得到的结果后，与v
        out = einsum('b h i j ,b h j d -> b h i d',attn,v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self,dim, depth, heads, dim_head, mlp_dim, dropout= 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim,heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim,mlp_dim,dropout=dropout))
            ]))

    def forward(self,x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, * ,image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, pool='cls', channels=3, dim_head=64, dropout= 0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height == 0 and image_width % patch_width ==0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls','mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim,dim)
        )

        #这里加一是因为要加入位置矩阵
        #nn.Parameter会自动被认为是module的可训练参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim,depth,heads,dim_head,mlp_dim,dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        x = self.to_patch_embedding(img)

        # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值
        b,n,_ = x.shape

        # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # 将cls_token拼接到patch token中去       (b, 65, dim)
        x = torch.cat((cls_tokens,x), dim=1)
        # 加位置嵌入（直接加）      (b, 65, dim)
        x += self.pos_embedding[:,:(n+1)]
        x = self.dropout(x)

        # (b, 65, dim)
        x = self.transformer(x)

        # (b, dim)
        x= x.mean(dim =1 ) if self.pool == 'mean' else x[:,0]

        # Identity (b, dim)
        x = self.to_latent(x)
        print(x.shape)

        #  (b, num_classes)
        return self.mlp_head(x)

def main():
    model_vit = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(16, 3, 256, 256)

    preds = model_vit(img)

    print(preds.shape)  # (16, 1000)


if __name__ == "__main__":
    main()

