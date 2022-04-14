import cv2
import numpy as np

def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)#random.normal(均值，标准差，大小)，此概率分布下的生成的数组元素个数，也可以是矩阵，这一步将数组放大两倍
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)#clip()，使用第二个参数代替第一个参数即数组里面小于该数的数据，使用第三个参数代替数组里面大于该参数的数据，修改后即0为最小值，255为最大值
    #uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255，astype是将数组类型转换为uint8型   这里为什么不等比例放缩到0-255？
    return noisy_img_clipped    #返回修剪之后（数据位于0-255之间）的图片二倍数据

def flip_image(image_array):          #这一步是进行图片翻转
    return cv2.flip(image_array, 1)   #将图片数组进行水平翻转，0是垂直翻转，-1是水平垂直翻转

def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)   #颜色空间转换函数，将Image_array从RGB格式转换为灰度图片
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d   #将三个色彩通道转换成了灰度图像
