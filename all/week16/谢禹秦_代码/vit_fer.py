import os
import torch

from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class RAFDB(Dataset):
    labels_num2str = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Neutral"]
    labels_str2num = {v: k for k, v in enumerate(labels_num2str)}

    def __init__(
        self,
        path: str,
        mode: str,
        transform=transforms.Compose(
            (
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            )
        ),
    ):
        self._label_map = dict()
        with open(os.path.join(path, "EmoLabel", "list_patition_label.txt")) as f:
            for line in f.read().splitlines():
                filename, label = line.split(" ")
                label = int(label) - 1
                self._label_map[filename.split(".")[0]] = label

        img_dir = os.path.join(path, "Image", "aligned")
        self._image_paths = []
        for image_name in os.listdir(img_dir):
            if image_name.startswith(mode):
                self._image_paths.append(os.path.join(img_dir, image_name))

        self._transform = transform

    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        image_name = "_".join(os.path.split(image_path)[1].split("_")[:2])
        label = self._label_map[image_name]
        data = Image.open(image_path)
        data = self._transform(data)
        return data, label

    def __len__(self):
        return len(self._image_paths)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.parameter.Parameter(
            torch.randn(1, num_patches + 1, dim)
        )
        self.cls_token = nn.parameter.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def main():
    dataset_path = r"D:\学习\大学\大创\王艳\RAF-DB"
    train_set = RAFDB(dataset_path, "train")
    test_set = RAFDB(dataset_path, "test")
    device = "cuda"

    model = ViT(
        image_size=100,
        patch_size=5,
        num_classes=7,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=256,
        dropout=0.01,
        emb_dropout=0.01,
    )

    criterion = nn.CrossEntropyLoss()
    epochs = 10
    batch_size = 32
    test_batch_size = 32
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for every step, this scheduler times the lr by gamma
    # step_size if how many steps before it times the lr by gamma
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=True, num_workers=4
    )

    test_results = []
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x.to(device))
            loss = criterion(output, y.to(device))
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"epoch: {epoch}, i: {batch_idx*len(x)}/{len(train_loader.dataset)}, loss: {loss.item()}"
                )

        scheduler.step()
        model.eval()
        correct_count = 0
        for batch_idx, (x, y) in enumerate(test_loader):
            output = model(x.to(device))
            correct_count += output.argmax(dim=-1).eq(y.to(device)).count_nonzero()
        print(
            f"correct: {correct_count}/{len(test_loader.dataset)}, {correct_count/len(test_loader.dataset)*100:.2f}%"
        )
        test_results.append(float(correct_count/len(test_loader.dataset)))

    print(f"test_results: {test_results}")

if __name__ == "__main__":
    main()
