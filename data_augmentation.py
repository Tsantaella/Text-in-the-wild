import cv2 as cv
import numpy as np
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt

def data_augmentation_one_image(image,target):
    final_train_data = []
    final_target_train = []
    sigma=0.2
    final_train_data.append(image)
    a=rotate(image, angle=45, mode = 'wrap')
    b=rotate(image, angle=-45, mode = 'wrap')
    final_train_data.append(a)
    final_train_data.append(b)
    final_train_data.append(random_noise(image,var=sigma**2))
    final_train_data.append(random_noise(a,var=sigma**2))
    final_train_data.append(random_noise(b,var=sigma**2))
    #final_train_data.append(np.fliplr(image))
    #final_train_data.append(np.flipud(image))
    kernel = np.ones((5,5),np.float32)/25
    final_train_data.append(cv.filter2D(image,-1,kernel))#convolution
    final_train_data.append(cv.GaussianBlur(image,(5,5),0))#gaussian_blurring
    final_train_data.append(cv.bilateralFilter(image,9,75,75))#bilateral_filtering
    final_train_data.append(cv.medianBlur(image,5))#median_blurring
    
    for j in range(len(final_train_data)):
        final_target_train.append(target)
    #final_train = np.array(final_train_data)
    #final_target_train = np.array(final_target_train)
    if (len(final_train_data)==len(final_target_train)): 
        return final_train_data, final_target_train
    else: 
        print("Error in data aumengtation!!!!")
        
        
def data_augmentation(data,label):
    train_images=[]
    train_labels=[]
    aux=0
    for img,tar in zip(data,label):
        x_train_prueba, y_train_prueba = data_augmentation_one_image(img,tar)
        train_images += x_train_prueba
        train_labels += y_train_prueba
        if(aux%10000==0):
            print(aux)
        aux+=1
    return train_images, train_labels