import cv2
import torch
from models import swin_transformer
from torchvision import transforms
import random
import torch.utils.data as data
import os
import pandas as pd
import image_utils

class MyDataSet(data.Dataset):
    def __init__(self,phase="train",basic_aug=True):
        self.phase=phase
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
        #一系列变换，根据均值标准差进行归一化
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        image=transform(image)

        return image,label,idx   #返回处理过后的图片数据、图片标签以及图片对应的位置


def test(model,test_loader):
    correct=0
    correct_rate=0
    for idx, (x, y, _) in enumerate(test_loader):
        output=model(x)
        _,max_result=output.max(axis=1)
        if max_result.equal(y):
            correct+=1
        correct_rate=correct/(idx+1)
        print(f"all:{idx+1}，corrct number:{correct}，corrct_rate:{100*correct_rate}%")


def train(model,optimizer,scheduler,criterion,epochs,train_loader,test_loader):
    model.train()
    for epoch in range(epochs):
        print(f"***************************{epoch}*****************************************")
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
    

def main():
    train_set=MyDataSet("train",True)
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=8,num_workers=4, shuffle=True)
    test_set=MyDataSet("test",True)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=1,num_workers=4, shuffle=True)
    model=swin_transformer.SwinTransformer(num_classes=7)
    model = torch.nn.DataParallel(model)
    checkPoint=torch.load("swin_base_patch4_window7_224_22k.pth",map_location="cpu")
    model.load_state_dict(checkPoint,False)
    epochs=10
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    train(model,optimizer,scheduler,criterion,epochs,train_loader,test_loader)

if __name__=="__main__":
    main()