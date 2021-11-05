# -*- coding: utf-8 -*-
"""
Created on Thu May 13 09:39:42 2021
Assignment1 
@author: Pingdu Zhou
"""
import numpy as np
#from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt
import nibabel as nib # loads .nii images
from numpy.fft import fft, ifft, ifft2, fftfreq, fft2, fftshift, fftn, ifftn

#Load Images
img_1 = nib.load("C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass1/cardiac_axial.nii.gz")
img_2 = nib.load("C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass1/cardiac_realtime.nii.gz")
img_3 = nib.load("C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass1/ct.nii.gz")
img_4 = nib.load("C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass1/fmri.nii.gz")
img_5 = nib.load("C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass1/meanpet.nii.gz")
img_6 = nib.load("C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass1/swi.nii.gz")
img_7 = nib.load("C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass1/T1_with_tumor.nii.gz")
img_8 = nib.load("C:/Users/Robert/Desktop/CS516 Volumetric image processing and Computer Vision/Ass1/tof.nii.gz")


#Get images' data
img1 = img_1.get_data()
img2 = img_2.get_data()
img3 = img_3.get_data()
img4 = img_4.get_data()
img5 = img_5.get_data()
img6 = img_6.get_data()
img7 = img_7.get_data()
img8 = img_8.get_data()

img1_freqs = fftshift(fftn(img1))
img1_freqs = img1_freqs.reshape(img1_freqs.shape[0],img1_freqs.shape[1],-1)[:,:,:30]
img2_freqs = fftshift(fftn(img2))
img2_freqs = img2_freqs.reshape(img2_freqs.shape[0],img2_freqs.shape[1],-1)
img3_freqs = fftshift(fftn(img3))
img4_freqs = fftshift(fftn(img4))
img4_freqs = img4_freqs.reshape(img4_freqs.shape[0],img4_freqs.shape[1],-1)[:,:,:153]
img5_freqs = fftshift(fftn(img5))
img6_freqs = fftshift(fftn(img6))
img7_freqs = fftshift(fftn(img7))
img8_freqs = fftshift(fftn(img8))

def filtered(img,sigma,img_freqs,i,n,j,title):
    sz_x = img.shape[0]
    sz_y = img.shape[1]
    sz_z = img.shape[i]
    [X,Y,Z] = np.mgrid[0:sz_x,0:sz_y,0:sz_z]
    xpr = X - int(sz_x) // 2
    ypr = Y - int(sz_y) // 2
    zpr = Z - int(sz_z) // 2
    gaussfilt = np.exp(-((xpr**2+ypr**2+zpr**2)/(2*sigma**2)))/(2*np.pi*sigma**2)
    gaussfilt = gaussfilt / np.max(gaussfilt)
    filterd_freqs = img_freqs * gaussfilt
    filtered = np.abs(ifftn(fftshift(filterd_freqs)))
    plt.subplot(3,3,n)
    plt.imshow(filtered[:,:,j])
    plt.title(title)
    plt.xticks([]),plt.yticks([])
    
def Michelson(img):    
    fmin = np.min(img)
    fmax = np.max(img)
    if fmin < 0:
        imgnor = 255 * (img - fmin)/(fmax - fmin)
        fmin = np.min(imgnor)
        fmax = np.max(imgnor)
    return (fmax - fmin) / (fmax + fmin)
    

def RMS(img):
    img3d = img.reshape(img.shape[0],img.shape[1],-1)
    I = np.mean(img3d)
    return (((img3d - I) ** 2).mean()) ** 0.5

def Entropy(img):
    H = plt.hist(img.flatten(),bins = int(np.max(img)-np.min(img)))
    Hnorm = H[0]/img.size
    Hnorm = Hnorm[np.where(Hnorm != 0)]
    entropy = -(Hnorm * np.log(Hnorm)).sum()
    return entropy

def SNR(img,i,signal,noise):
    img3d = img.reshape(img.shape[0],img.shape[1],-1)
    midimg = img3d[:,:,i]
    Sarea = midimg[signal[0]:signal[1],signal[2]:signal[3]]
    Narea = midimg[noise[0]:noise[1],noise[2]:noise[3]]
    snr = Sarea.mean() / Narea.std()
    return snr

def Noise(img,i,noise):
     img3d = img.reshape(img.shape[0],img.shape[1],-1)
     midimg = img3d[:,:,i]
     Narea = midimg[noise[0]:noise[1],noise[2]:noise[3]]
     return Narea





#part1_a
plt.subplot(3,3,1)
plt.imshow(img1[:,:,0,15],cmap='jet')
plt.title("cardiac_axial")
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,2)
plt.imshow(img2[:,:,0,250],cmap='jet')
plt.title("cardiac_realtime")
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,3)
plt.imshow(img3[:,:,120],cmap='jet')
plt.title("ct")
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,4)
plt.imshow(img4[:,:,0,77],cmap='jet')
plt.title("fmri")
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,5)
plt.imshow(img5[:,:,104],cmap='jet')
plt.title("meanpet")
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,6)
plt.imshow(img6[:,:,250],cmap='jet')
plt.title("swi")
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,7)
plt.imshow(img7[:,:,128],cmap='jet')
plt.title("T1_with_tumor")
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,8)
plt.imshow(img8[:,:,75],cmap='jet')
plt.title("tof")
plt.xticks([]),plt.yticks([])

#part1_b
min_img = img6.copy()
max_img = img8.copy()
min_img = min_img[:,:,200:300]
min_img = np.min(min_img,axis=2)
max_img = np.max(max_img,axis = 2)
plt.subplot(1,2,1)
plt.imshow(min_img,cmap = 'jet')
plt.title("SWI MIP")
plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(max_img,cmap = 'jet',vmin = 0, vmax = 300)
plt.title("TOF MIP")
plt.xticks([]),plt.yticks([])

#part2
michelson1 = Michelson(img1)
michelson2 = Michelson(img2)
michelson3 = Michelson(img3)
michelson4 = Michelson(img4)
michelson5 = Michelson(img5)
michelson6 = Michelson(img6)
michelson7 = Michelson(img7)
michelson8 = Michelson(img8)
rms1 = RMS(img1)
rms2 = RMS(img2)
rms3 = RMS(img3)
rms4 = RMS(img4)
rms5 = RMS(img5)
rms6 = RMS(img6)
rms7 = RMS(img7)
rms8 = RMS(img8)
entropy1 = Entropy(img1)
entropy2 = Entropy(img2)
entropy3 = Entropy(img3)
entropy4 = Entropy(img4)
entropy5 = Entropy(img5)
entropy6 = Entropy(img6)
entropy7 = Entropy(img7)
entropy8 = Entropy(img8)

plt.subplot(3,3,1)
plt.imshow(img1[:,:,0,15],cmap='jet')
plt.title("Michelson: %d"%michelson1)
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,2)
plt.imshow(img2[:,:,0,250],cmap='jet')
plt.title("Michelson: %d"%michelson2)
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,3)
plt.imshow(img3[:,:,120],cmap='jet')
plt.title("Michelson: %d"%michelson3)
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,4)
plt.imshow(img4[:,:,0,77],cmap='jet')
plt.title("Michelson: %d"%michelson4)
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,5)
plt.imshow(img5[:,:,104],cmap='jet')
plt.title("Michelson: %d"%michelson5)
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,6)
plt.imshow(img6[:,:,250],cmap='jet')
plt.title("Michelson: %d"%michelson6)
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,7)
plt.imshow(img7[:,:,128],cmap='jet')
plt.title("Michelson: %d"%michelson7)
plt.xticks([]),plt.yticks([])

plt.subplot(3,3,8)
plt.imshow(img8[:,:,75],cmap='jet')
plt.title("Michelson: %d"%michelson8)
plt.xticks([]),plt.yticks([])

#part3
snr1 = SNR(img1,60,[100,150,100,150],[250,300,250,300])
snr2 = SNR(img2,250,[20,40,80,100],[20,40,0,20])
snr3 = SNR(img3,120,[200,250,200,250],[0,50,400,450])
snr4 = SNR(img4,2754,[60,80,60,80],[0,20,0,20])
snr5 = SNR(img5,104,[100,150,100,150],[0,50,0,50])
snr6 = SNR(img6,250,[200,250,200,250],[10,60,100,150])
snr7 = SNR(img7,128,[100,150,60,80],[0,50,0,20])
snr8 = SNR(img8,75,[200,250,200,250],[0,50,0,50])

img1_noise = Noise(img1,60,[250,300,250,300])
img2_noise = Noise(img2,250,[20,40,0,20])
img3_noise = Noise(img3,120,[0,50,400,450])
img4_noise = Noise(img4,2754,[0,20,0,20])
img5_noise = Noise(img5,104,[0,50,0,50])
img6_noise = Noise(img6,250,[10,60,100,150])
img7_noise = Noise(img7,128,[0,50,0,20])
img8_noise = Noise(img8,75,[0,50,0,50])

plt.subplot(3,3,1)
plt.hist(img1_noise)
plt.title("SNR: %.3f"%snr1)
plt.subplot(3,3,2)
plt.hist(img2_noise)
plt.title("SNR: %.3f"%snr2)
plt.subplot(3,3,3)
plt.hist(img3_noise)
plt.title("SNR: %.3f"%snr3)
plt.subplot(3,3,4)
plt.hist(img4_noise)
plt.title("SNR: %.3f"%snr4)
plt.subplot(3,3,5)
plt.hist(img5_noise)
plt.title("SNR: %.3f"%snr5)
plt.subplot(3,3,6)
plt.hist(img6_noise)
plt.title("SNR: %.3f"%snr6)
plt.subplot(3,3,7)
plt.hist(img7_noise)
plt.title("SNR: %.3f"%snr7)
plt.subplot(3,3,8)
plt.hist(img8_noise)
plt.title("SNR: %.3f"%snr8)

#part4
filtered(img1,15,img1_freqs,3,1,15,"cardiac_axial sigma=15")
filtered(img2,15,img2_freqs,3,2,250,"cardiac_realtime sigma=15")
filtered(img3,15,img3_freqs,2,3,120,"ct sigma=15")
filtered(img4,15,img4_freqs,3,4,76,"fmri sigma=15")
filtered(img5,15,img5_freqs,2,5,103,"meanpet sigma=15")
filtered(img6,15,img6_freqs,2,6,250,"swi sigma=15")
filtered(img7,15,img7_freqs,2,7,128,"T1_with_tumor sigma=15")
filtered(img8,15,img8_freqs,2,8,75,"tof sigma=15")