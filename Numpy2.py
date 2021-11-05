# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:49:52 2021

@author: Pingdu Zhou 

Assignment2 
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import sklearn.metrics as skm
import math
import numpy as np
import cv
import cv2
import os


path = "C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass2/Data/"
imgs = os.listdir(path)
images = {}

for img in imgs:
    images[img] = cv2.imread(path+img,0)
    
#Part1
def JointHist(I,J,bins,x,y):
   plt.hist2d(I.flatten(), J.flatten(), bins = bins,cmap = 'jet',cmin = 0,cmax = 1000)
   plt.xlabel(x)
   plt.ylabel(y)
   plt.show()
# Part1_b
a = images['I1.png'].flatten()
h = np.histogram(a, 256,(0, 255))[0].sum()
a.shape
   
JointHist(images['I1.png'], images['J1.png'], 25,"I1","J1")
JointHist(images['I2.jpg'], images['J2.jpg'], 25,"I2","J2")
JointHist(images['I3.jpg'], images['J3.jpg'], 25,"I3","J3")
JointHist(images['I4.jpg'], images['J4.jpg'], 25,"I4","J4")
JointHist(images['I5.jpg'], images['J5.jpg'], 25,"I5","J5")
JointHist(images['I6.jpg'], images['J6.jpg'], 25,"I6","J6")


#Part2_a
def SSD(I,J):
    ssd = np.sum((I-J)**2)
    return ssd

SSD(images['I1.png'],images['J1.png'])
SSD(images['I2.jpg'],images['J2.jpg'])
SSD(images['I3.jpg'],images['J3.jpg'])
SSD(images['I4.jpg'],images['J4.jpg'])
SSD(images['I5.jpg'],images['J5.jpg'])
SSD(images['I6.jpg'],images['J6.jpg'])

#Part2_b
def corr(I,J):
    corr = (((I-I.mean())*(J-J.mean())).sum()) / ((((I-I.mean())**2).sum())**0.5 * (((J-J.mean())**2).sum())**0.5)
    return corr

corr(images['I1.png'],images['J1.png'])
corr(images['I2.jpg'],images['J2.jpg'])
corr(images['I3.jpg'],images['J3.jpg'])
corr(images['I4.jpg'],images['J4.jpg'])
corr(images['I5.jpg'],images['J5.jpg'])
corr(images['I6.jpg'],images['J6.jpg'])

#Part2_c
def MI(I,J):
    i = I.flatten()
    j = J.flatten()
    pi = np.histogram(i, 256,(0, 255))[0] / i.size
    pj = np.histogram(j, 256,(0, 255))[0] / j.size
    hi = - np.sum(pi * np.log(pi + 1e-8)) 
    hj = - np.sum(pj * np.log(pj + 1e-8)) 
    hij = np.histogram2d(i, j, 256, [[0, 255],[0, 255]])[0]
    hij /= (1.0 * i.size) 
    hij = -np.sum(hij * np.log(hij + 1e-8))
    
    mi = hi + hj -hij
    return mi

MI(images['I1.png'],images['J1.png'])
MI(images['I2.jpg'],images['J2.jpg'])
MI(images['I3.jpg'],images['J3.jpg'])
MI(images['I4.jpg'],images['J4.jpg'])
MI(images['I5.jpg'],images['J5.jpg'])
MI(images['I6.jpg'],images['J6.jpg'])

# skm.mutual_info_score(images['I6.jpg'].flatten(),images['J6.jpg'].flatten())

#Part3_a
X = np.linspace(0,20,21)
Y = np.linspace(0,20,21)
Z = np.linspace(0,4,5)
x, y, z = np.meshgrid(X,Y,Z)
a = np.array([x,y,z])
b = a.reshape(3,-1)
matrix = b.copy()
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim3d(0,21)
ax.set_ylim3d(0,21)
ax.set_zlim3d(0,15)
ax.scatter(x, y, z,c = 'black')
 
#Part3_b
def rigid_transform(mat,theta,omega,phi,p,q,r):
     RXaxis = np.mat([[1, 0, 0, 0],
                      [0, np.cos(theta), -np.sin(theta), 0],
                      [0, np.sin(theta), np.cos(theta), 0],
                      [0, 0, 0, 1]])
     
     RYaxis = np.mat([[np.cos(omega), 0, np.sin(omega), 0],
                      [0, 1, 0, 0],
                      [-np.sin(omega), 0, np.cos(omega), 0],
                      [0, 0, 0, 1]])
     
     RZaxis = np.mat([[np.cos(phi), -np.sin(phi), 0, 0],
                      [np.sin(phi), np.cos(phi), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
     
     Transform = np.mat([[1, 0, 0, p],
                         [0, 1, 0, q],
                         [0, 0, 1, r],
                         [0, 0, 0, 1]])
     
     mat = np.r_[mat,np.ones((1,mat.shape[1]))]
     rotatemat = Transform * RYaxis * RXaxis * RZaxis * mat
     x1,y1,z1 = np.meshgrid(rotatemat[0,:],rotatemat[1,:],rotatemat[2,:])
     X = np.linspace(0,20,21)
     Y = np.linspace(0,20,21)
     Z = np.linspace(0,4,5)
     x, y, z = np.meshgrid(X,Y,Z)
     fig = plt.figure()
     ax = Axes3D(fig)
     ax.scatter(x, y, z,c = 'black')
     ax.scatter(x1, y1, z1,c = 'red')
     plt.show()

rigid_transform(matrix,math.pi/4,math.pi/4,math.pi/4,0,0,20)

#Part3_c   
def affine_transform(mat,s,theta,omega,phi,p,q,r):
     RYaxis = np.mat([[np.cos(omega), 0, np.sin(omega), 0],
                      [0, 1, 0, 0],
                      [-np.sin(omega), 0, np.cos(omega), 0],
                      [0, 0, 0, 1]])
     
     RXaxis = np.mat([[1, 0, 0, 0],
                      [0, np.cos(theta), -np.sin(theta), 0],
                      [0, np.sin(theta), np.cos(theta), 0],
                      [0, 0, 0, 1]])
     
     RZaxis = np.mat([[np.cos(phi), -np.sin(phi), 0, 0],
                      [np.sin(phi), np.cos(phi), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
     
     Transform = np.mat([[s, 0, 0, p],
                         [0, s, 0, q],
                         [0, 0, s, r],
                         [0, 0, 0, 1]])
     
     mat = np.r_[mat,np.ones((1,mat.shape[1]))]
     
     rotatemat = Transform * RYaxis * RXaxis * RZaxis * mat
     x2,y2,z2 = np.meshgrid(rotatemat[0,:],rotatemat[1,:],rotatemat[2,:])
     X = np.linspace(0,20,21)
     Y = np.linspace(0,20,21)
     Z = np.linspace(0,4,5)
     x, y, z = np.meshgrid(X,Y,Z)
     fig = plt.figure()
     ax = Axes3D(fig)
     ax.set_xlim3d(0,21)
     ax.set_ylim3d(0,21)
     ax.set_zlim3d(0,15)
     ax.scatter(x, y, z,c = 'black')
     ax.scatter(x2, y2, z2,c = 'red')
     plt.show()

affine_transform(matrix,0.5,math.pi/2,math.pi/8,math.pi/4,10,0,0)

#Part3_d

mat = np.r_[matrix,np.ones((1,matrix.shape[1]))] 
  
M1 = np.mat([[0.9045, -0.3847, -0.1840, 10.0000],
             [0.2939, 0.8750, -0.3847, 10.0000],
             [0.3090, 0.2939, 0.9045, 10.0000],
             [0, 0, 0, 1.0000]])  
  
M2 = np.mat([[-0.0000, -0.2595, 0.1500, -3.0000],
             [0.0000, -0.1500, -0.2598, 1.5000],
             [0.3000, -0.0000, 0.0000, 0],
             [0, 0, 0, 1.0000]]) 

M3 = np.mat([[0.7182, -1.3727, -0.5660, 1.8115],
             [-1.9236, -4.6556, -2.5515, 0.2873],
             [-0.6426, -1.7985, -1.6285, 0.7404],
             [0, 0, 0, 1.0000]])

mat1 = M1.dot(mat)
mat2 = M2.dot(mat)
mat3 = M3.dot(mat)
X = np.linspace(0,20,21)
Y = np.linspace(0,20,21)
Z = np.linspace(0,4,5)
x, y, z = np.meshgrid(X,Y,Z)
x1,y1,z1 = np.meshgrid(mat1[0,:],mat1[1,:],mat1[2,:])
x2,y2,z2 = np.meshgrid(mat2[0,:],mat1[1,:],mat1[2,:])
x3,y3,z3 = np.meshgrid(mat3[0,:],mat1[1,:],mat1[2,:])
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z,c = 'black')
ax.scatter(x1, y1, z1,c = 'red')
ax.scatter(x2, y2, z2,c = 'red')
ax.scatter(x3, y3, z3,c = 'red')
plt.show()


#Part4_a
def translation(I,p,q):
    Img = I.copy().astype('float')
    h = Img.shape[0]
    w = Img.shape[1]
    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    f = interp2d(x+p, y+q, Img, kind='cubic', fill_value=0)
    znew = f(x, y)
    plt.subplot(1,2,1); plt.imshow(Img) 
    plt.subplot(1,2,2); plt.imshow(znew)
    return znew

translation(images['I1.png'],100,50)

#Part4_b
def translationSSD(img1,img2,legend):
    iters = 500
    lr = 1e-7
    cost_history = []
    p,q = 0, 0
    for itr in range(iters):
       current = translation(img2,p,q)
       gy,gx = np.gradient(current)
       dx = -((current - img1) * gx).sum() * 2
       dy = -((current - img1) * gy).sum() * 2
       #du = np.array([dx,dy])
       #cu = lr * du
       p -= lr * dx
       q -= lr * dy
       cost_history.append(SSD(current, img1))
    plt.subplot(1,2,1); plt.imshow(img1) 
    plt.subplot(1,2,2); plt.imshow(current)
    plt.show()
    plt.plot(np.arange(len(cost_history)), cost_history)
    plt.legend([legend])
    plt.show()
    return cost_history

translationSSD(images['BrainMRI_1.jpg'],images['BrainMRI_2.jpg'],'BrainMRI_2')
translationSSD(images['BrainMRI_1.jpg'],images['BrainMRI_3.jpg'],'BrainMRI_3')
translationSSD(images['BrainMRI_1.jpg'],images['BrainMRI_4.jpg'],'BrainMRI_4')

#Part4_c
def rotation_TopLeft(I,theta):   
    h = I.shape[0]
    w = I.shape[1]
    angle = theta * math.pi/180
    cosA = math.cos(angle)
    sinA = math.sin(angle)
    img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x = int(i * cosA - j * sinA)
            y = int(i * sinA + j * cosA)
            if x >= 0 and x < h and y >= 0 and y < w:
                 img[i,j] = I[x, y]
    plt.imshow(img)
    plt.title('theta %f'%theta)
    return img

rotation_TopLeft(images['I1.png'],45)

def rotation(I,theta):
    radius = math.pi * theta / 180
    width, height = I.shape
    im_new = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            new_i = int(
                (i - 0.5 * width) * math.cos(radius) - (j - 0.5 * height) * math.sin(radius) + 0.5 * width)
            new_j = int(
                (i - 0.5 * width) * math.sin(radius) + (j - 0.5 * height) * math.cos(radius) + 0.5 * height)
            if new_i >= 0 and new_i < width and new_j >= 0 and new_j < height:
                im_new[i, j] = I[new_i, new_j]
    plt.imshow(im_new)
    return im_new

#Part4_d
def rotationSSD(img1, img2, legend):
    iters = 500
    lr = 1e-9
    cost_history = []
    t = 0
    x = np.array([[j for i in range(img1.shape[1])] for j in range(img1.shape[0])]) 
    y = np.array([[i for i in range(img1.shape[1])] for j in range(img1.shape[0])])      
    for itr in range(iters):
       current = rotation_TopLeft(img2,t)
       c,s = np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t))
       gy,gx = np.gradient(current)
       dt = -2 * ((current - img1) * (gy*(x*c-y*s) - gx*(x*s+y*c))).sum()
       ct = lr * dt
       t -= ct
       cost_history.append(SSD(current, img1))
    plt.imshow(current)
    plt.title('theta %f'%t)
    plt.show()
    plt.plot(np.arange(len(cost_history)), cost_history)
    plt.legend([legend])
    plt.show()
    return cost_history

rotationSSD(images['BrainMRI_1.jpg'],images['BrainMRI_2.jpg'],'2')
rotationSSD(images['BrainMRI_1.jpg'],images['BrainMRI_3.jpg'],'3')
rotationSSD(images['BrainMRI_1.jpg'],images['BrainMRI_4.jpg'],'4')

#Part4_e
def registration(img1,img2,legend):
    iters = 500
    lr = 1e-9
    cost_history = []
    p,q = 0, 0
    t = 0
    x = np.array([[j for i in range(img1.shape[1])] for j in range(img1.shape[0])]) 
    y = np.array([[i for i in range(img1.shape[1])] for j in range(img1.shape[0])])  
    for itr in range(iters):
       
       current1 = translation(img2,p,q)
       current = rotation_TopLeft(current1,t)
       gy,gx = np.gradient(current)
       c,s = np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t))
       dx = -((current - img1) * gx).sum() * 2
       dy = -((current - img1) * gy).sum() * 2
       dt = -2 * ((current - img1) * (gy*(x*c-y*s) - gx*(x*s+y*c))).sum()
       #du = np.array([dx,dy])
       #cu = lr * du
       ct = lr * dt
       t -= ct
       p -= lr * dx
       q -= lr * dy
       cost_history.append(SSD(current, img1))
    plt.imshow(current)
    plt.title('theta %f'%t)
    plt.show()
    plt.plot(np.arange(len(cost_history)), cost_history)
    plt.legend([legend])
    return cost_history

registration(images['BrainMRI_1.jpg'],images['BrainMRI_2.jpg'],'2')
registration(images['BrainMRI_1.jpg'],images['BrainMRI_3.jpg'],'3')  
registration(images['BrainMRI_1.jpg'],images['BrainMRI_4.jpg'],'4')    
    
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
       