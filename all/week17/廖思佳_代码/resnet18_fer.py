import os
import torch

from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision


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


class Resnet18(nn.Module):
    def __init__(self, num_classes=7, drop_rate=0.1):
        super().__init__()
        resnet = torchvision.models.resnet18(False) # true 预训练
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(
            *list(resnet.children())[:-1] # resnet.children() 拿出所有层 *号展开
        )  # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention

        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(fc_in_dim, num_classes)

    def forward(self, x):
        x = self.features(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        out = self.fc(x)
        return out


def main():
    dataset_path = r"D:\学习\大学\大创\王艳\RAF-DB"
    train_set = RAFDB(dataset_path, "train")
    test_set = RAFDB(dataset_path, "test")
    device = "cuda"

    model = Resnet18()

    criterion = nn.CrossEntropyLoss()
    epochs = 10
    batch_size = 64
    test_batch_size = 64
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for every step, this scheduler times the lr by gamma
    # step_size if how many steps before it times the lr by gamma
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.4)
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
        test_results.append(float(correct_count / len(test_loader.dataset)))

    print(f"test_results: {test_results}, max: {max(test_results)}")


if __name__ == "__main__":
    main()
