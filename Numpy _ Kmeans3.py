# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:48:35 2021

@author: Pingdu ZHou
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.morphology import square, erosion
from skimage.segmentation import flood_fill

# nii = nib.load('C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass2/t1.nii')
# img = nii.get_data()

# plt.hist(img.ravel(),bins = 100)
# mb = MiniBatchKMeans(n_clusters = 3)
# mb.fit(np.expand_dims(np.ravel(img[:,:,250]),axis = 1))
# labs = np.reshape(mb.labels_,[img.shape[0],img.shape[1]])

# bg = labs==1
# origwm = labs==2
# wm = labs==2
# gmsk = labs==0

# flood = flood_fill(erosion(wm+gmsk, square(3)).astype(float),(101,205),10,tolerance = 0.5)

# for i in np.range(0,10):
#     wm = dilation(wm,square(3))
#     wm = wm * gmsk + origwm

#load
img1 = nib.load('C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass4/t1s/img1.nii.gz')
img2 = nib.load('C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass4/t1s/img2.nii.gz')
img3 = nib.load('C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass4/t1s/img3.nii.gz')
t2 = nib.load('C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass4/t1s/t2.nii.gz')

img1 = img1.get_data()
img2 = img2.get_data()
img3 = img3.get_data()
t2 = t2.get_data()


def IMGsegmentationX(image,sliceindex,title = 'XAxis'):
    img = image.copy()  
    seed = np.where(img[sliceindex,:,:]==np.max(img[sliceindex,:,:][150:200,210:250]))
    mb = KMeans(n_clusters = 3,random_state = 3262)
    mb.fit(np.expand_dims(np.ravel(img[sliceindex,:,:]),axis = 1))
    labs = np.reshape(mb.labels_,[img.shape[1],img.shape[2]])
    a = labs==0
    b= labs==1
    c = labs==2
    labs_list = [a, b, c]
    labs_mean = []
    for i in labs_list:
        labs_mean.append(np.mean(i * img[sliceindex,:,:]))
    total = np.zeros(img[sliceindex,:,:].shape)
    for i in range(len(labs_mean)):
        if labs_mean[i] != min(labs_mean):
            total += labs_list[i]
    flood = flood_fill(erosion(total, square(3)).astype(float),(seed[0][0],seed[1][0]),10,tolerance = 0.5) 
    flood[flood!=10] = 0
    flood[flood==10] = 1  
    plt.subplot(1,2,1)
    plt.imshow(img[sliceindex,:,:],cmap = 'gray')
    plt.xticks([]),plt.yticks([])
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(flood * img[sliceindex,:,:],cmap = 'gray')
    plt.xticks([]),plt.yticks([])
    plt.title(title)
    plt.show()
    
    
def IMGsegmentationY(image,sliceindex,title = 'YAxis'):
    img = image.copy()   
    seed = np.where(img[:,sliceindex,:]==np.max(img[:,sliceindex,:][100:160,200:300]))
    mb = KMeans(n_clusters = 3,random_state = 3262)
    mb.fit(np.expand_dims(np.ravel(img[:,sliceindex,:]),axis = 1))
    labs = np.reshape(mb.labels_,[img.shape[0],img.shape[2]])
    a = labs==0
    b = labs==1
    c = labs==2
    labs_list = [a, b, c]
    labs_mean = []
    for i in labs_list:
        labs_mean.append(np.mean(i * img[:,sliceindex,:]))
    total = np.zeros(img[:,sliceindex,:].shape)
    for i in range(len(labs_mean)):
        if labs_mean[i] != min(labs_mean):
            total += labs_list[i]
    flood = flood_fill(erosion(total, square(3)).astype(float),(seed[0][0],seed[1][0]),10,tolerance = 0.5) 
    flood[flood!=10] = 0
    flood[flood==10] = 1 
    plt.subplot(1,2,1)
    plt.imshow(img[:,sliceindex,:],cmap = 'gray')
    plt.xticks([]),plt.yticks([])
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(flood * img[:,sliceindex,:],cmap = 'gray')
    plt.xticks([]),plt.yticks([])
    plt.title(title)
    plt.show()
    
def IMGsegmentationZ(image,sliceindex,title = 'ZAxis'):
    img = image.copy()
    seed = np.where(img[:,:,sliceindex]==np.max(img[:,:,sliceindex][100:200,100:160]))
    mb = KMeans(n_clusters = 3,random_state = 3262)
    mb.fit(np.expand_dims(np.ravel(img[:,:,sliceindex]),axis = 1))
    labs = np.reshape(mb.labels_,[img.shape[0],img.shape[1]])
    a = labs==0
    b= labs==1
    c = labs==2
    labs_list = [a, b, c]
    labs_mean = []
    for i in labs_list:
        labs_mean.append(np.mean(i * img[:,:,sliceindex]))
    total = np.zeros(img[:,:,sliceindex].shape)
    for i in range(len(labs_mean)):
        if labs_mean[i] != min(labs_mean):
            total += labs_list[i]
    flood = flood_fill(erosion(total, square(3)).astype(float),(seed[0][0],seed[1][0]),10,tolerance = 0.5) 
    flood[flood!=10] = 0
    flood[flood==10] = 1
    plt.subplot(1,2,1)
    plt.imshow(img[:,:,sliceindex],cmap = 'gray')
    plt.xticks([]),plt.yticks([])
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(flood * img[:,:,sliceindex],cmap = 'gray')
    plt.xticks([]),plt.yticks([])
    plt.title(title)
    plt.show()
     
    
# mb = KMeans(n_clusters = 3,init='k-means++',random_state=1234)
# mb.fit(np.expand_dims(np.ravel(img1[:,:,250]),axis = 1))
# labs = np.reshape(mb.labels_,[img1.shape[0],img1.shape[1]])
# a = labs==0
# b = labs==1
# c = labs==2
# labs_list = [a, b, c]
# labs_mean = []
# for i in labs_list:
#     labs_mean.append(np.mean(i * img1[:,:,250]))
# total = np.zeros(img1[:,:,250].shape)
# for i in range(len(labs_mean)):
#     if labs_mean[i] != min(labs_mean):
#         total += labs_list[i]
# a0 = np.mean(a*img1[:,:,200])
# b0 = np.mean(b*img1[:,:,200])
# c0 = np.mean(c*img1[:,:,200])
# print(min(a0, b0, c0))
# if a0
# plt.imshow(total)
# flood = flood_fill(erosion(total, square(3)).astype(float),(101,205),10,tolerance = 0.5)   
# plt.imshow(flood)
# flood[flood!=10] = 0
# plt.imshow(img1[:,:,250] * flood)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
