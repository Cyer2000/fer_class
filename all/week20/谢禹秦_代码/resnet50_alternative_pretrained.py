import os
import pickle
import torch
import torchvision

from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class RAFDB(Dataset):
    labels_num2str = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Neutral"]
    labels_str2num = {v: k for k, v in enumerate(labels_num2str)}

    def __init__(
        self,
        path: str,
        mode: str,
        transform=transforms.Compose(
            (
                transforms.Resize((224, 224)),
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


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def main():
    dataset_path = r"D:\学习\大学\大创\王艳\RAF-DB"
    train_set = RAFDB(dataset_path, "train")
    test_set = RAFDB(dataset_path, "test")
    device = "cuda"

    model = torchvision.models.resnet50(False)
    model.fc = nn.Linear(model.fc.in_features, 8631)
    load_state_dict(model, r"D:\学习\大学\大创\王艳\checkpoints\resnet50_ft_weight.pkl")
    model.fc = nn.Linear(model.fc.in_features, 7)

    # model = torchvision.models.resnet50(True)
    # model.fc = nn.Linear(model.fc.in_features, 7)

    criterion = nn.CrossEntropyLoss()
    epochs = 40
    batch_size = 32
    test_batch_size = 32
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=True, num_workers=4
    )

    test_results = []
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x.to(device))
            loss = criterion(output, y.to(device))
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"epoch: {epoch}, i: {batch_idx*len(x)}/{len(train_loader.dataset)}, loss: {loss.item()}",
                    end=f"{' '*10}\r",
                )
        print()

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
