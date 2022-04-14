import torch.utils.data as data
import pandas as pd
import os
import image_utils
import cv2
from torchvision import transforms
import torch
import random
import torchvision
from IPython.display import display
from einops.layers.torch import Rearrange
from torch import layer_norm, nn
import torch
import einops
from torchvision import transforms
pil=torchvision.transforms.ToPILImage()#将tensor数据转换成图片


class MyDataSet(data.Dataset):
    def __init__(self,phase="train",transform=None,basic_aug=True):
        self.phase=phase
        self.transform=transform
        self.basic_aug=basic_aug
        
        NAME_COLUMN=0 #图片名字
        LABEL_COLUMN=1 #图片标签

        df=pd.read_csv("E:/desktop/WY/MY/15/datasets/raf-basic/EmoLabel/list_patition_label.txt",sep=' ',header=None)
        if phase=="train":
            dataset=df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset=df[df[NAME_COLUMN].str.startswith('test')]
        file_names=dataset.iloc[:,NAME_COLUMN].values    #所有行获得第一列的文件名字
        self.label=dataset.iloc[:,LABEL_COLUMN].values-1 #  0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths=[]#使用对齐过的图片进行训练、测试

        for f in file_names:
            f=f.split(".")[0] #只要文件名字，不要后缀名
            f=f+"_aligned.jpg" #使用对齐后的图片
            path=os.path.join('E:/desktop/WY/MY/15/datasets/raf-basic/Image/aligned',f)
            self.file_paths.append(path)

        self.aug_func=[image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)   #返回一共有多少张图片

    def __getitem__(self,idx):   
        path=self.file_paths[idx]
        image=cv2.imread(path)
        # display(pil(image))
        image = image[:, :, ::-1]  #读取该张图片并将其转换为RGB格式，原格式为#BGR
        label=self.label[idx]  #得到该张图片对应的标签

        #如果是训练阶段，就对图片进行随机增强
        if self.phase=="train":
            if self.basic_aug and random.uniform(0,1)>0.5:
                #即如果basic_aug为真并且随机数大于0.5，就对该张图片进行随机增强
                index=random.randint(0,1) #随机取0或者取1
                image=self.aug_func[index](image) #取增强中的随机一种方法进行图片增强
        
            #然后再对图片进行预处理
        if self.transform is not None:
            #一系列变换，根据均值标准差进行归一化,随机翻转、随机遮挡
            image=self.transform(image)

        return image,label,idx   #返回处理过后的图片数据、图片标签以及图片对应的位置


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积1  （b 3 224 224） -> (b 64 56 56)
        self.proj1 = nn.Conv2d(
            3, 64,
            kernel_size=7,
            stride=4,
            padding=2
        )
        self.drop = nn.Dropout(0.00001)
        # 卷积2  (b 64 56 56) -> (b 128 28 28)
        self.proj2 = nn.Conv2d(
            64, 128,
            kernel_size=5,
            stride=2,
            padding=2
        )
        # embedding  (b 128 28 28) -> (c )
        self.patch_embedding=nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),#将得到的每个小图片的长乘以宽的像素展平，得到196*256的向量
            nn.LayerNorm(128)
        )#w,h分别为每个patch的宽和高

        self.fc=nn.Linear(128,128)
        self.class_token=nn.parameter.Parameter(torch.randn(1,1,128))
        self.position_encodings=nn.parameter.Parameter(
            torch.randn((1,28*28+1,128))
        )
        # 设置transformer encoder
        self.transformer_encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,nhead=16,dim_feedforward=128,batch_first=True
            ),
            3,
        )
        self.mlp_head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,7))

    def forward(self,x):
        x=self.proj1(x)
        x=self.drop(x)
        x=self.proj2(x)
        x=self.patch_embedding(x)
        batch_size, _ , _=x.shape
        x=self.fc(x)
        class_token=einops.repeat(
            self.class_token,
            "() h  w -> repeat h w",
            repeat=batch_size
        )
        x = torch.cat((class_token,x),dim=-2)
        x += self.position_encodings 
        x=self.transformer_encoder(x)
        x=x[:,0]
        x=self.mlp_head(x)
        return x

def test(model,test_loader):
    correct=0
    for idx, (x, y, _) in enumerate(test_loader):
        output=model(x)
        _,max_result=output.max(axis=1)
        if max_result.equal(y):
            correct+=1
    print(f"acc:{correct/len(test_loader.dataset)}")


def train(model,optimizer,scheduler,criterion,epochs,train_loader,test_loader):
    model.train()
    for epoch in range(epochs):
        print(f"***************************epoch:{epoch}*****************************************")
        for batch_idx, (x, y, _) in enumerate(train_loader):
            optimizer.zero_grad()
            output=model(x)
            loss=criterion(output,y)
            loss.backward()
            optimizer.step()

            if batch_idx%10==0:
                print(f"epoch:{epoch},i:{batch_idx*len(x)}/{len(train_loader.dataset)},loss:{loss.item()}")
        scheduler.step()   
        test(model,test_loader)                                     


def run_training():
    #构建数据集、参数等
    train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02,0.25))])
    train_dataset=MyDataSet(phase='train',transform=train_transforms,basic_aug=False)
    print(train_dataset.__len__())
    train_loader=torch.utils.data.DataLoader(train_dataset,
                                                batch_size=16,
                                                num_workers=4,
                                                shuffle=True,
                                                pin_memory=False)
    test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])        
    test_dataset=MyDataSet(phase='test',transform=test_transforms,basic_aug=False)
    print(test_dataset.__len__())
    test_loader=torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            num_workers=4,
                                            shuffle=True,
                                            pin_memory=False)
    vision_transformer_model=VisionTransformer()
    epochs=10
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(vision_transformer_model.parameters(),lr=0.005)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    train(vision_transformer_model, optimizer, scheduler, criterion, epochs, train_loader,test_loader)

if __name__ == "__main__":                    
    run_training()