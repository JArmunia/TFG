import tensorflow as tf
import tensorflow.keras as keras
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib
import scipy
import math

def calculate_output_size(in_size, kernel_size, padding, stride):
    return 1 + ((in_size - kernel_size + 2*padding) / stride)

def g(sigma, x):
    """ Returns the value of the gaussian formula in x given a sigma """
    sigma2 = sigma ** 2
    return (1/(np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x**2)/(2 * sigma2)))

def gauss(sigma, size): 
    """ Returns a 1D gaussian kernel """
    start = int(-np.floor(size/2))
    end = int(-start + 1)
    kernel = np.zeros((size,1), dtype='float32')
    for x in range(start, end ):
        kernel[x-start] = g(sigma, x)
    return kernel

def gauss3d(sigma, size):
    """ Returns a 3D gaussian kernel """
    arr1 = gauss(sigma, size).reshape(size, 1, 1)
    arr2 = arr1.reshape(1, size, 1)
    arr3 = arr1.reshape(1, 1, size)
    kernel = arr1 * arr2 * arr3
    return (kernel / kernel.sum()).astype('float32')

def augment(images, labels):
    """
    Data augmentation of an image array
    """
    shifts = tf.constant([[1, 0, 0], [-1, 0, 0],
                          [0, 1, 0], [0, -1, 0],
                          [0, 0, 1], [0, 0, -1]])
    gaussians = tf.constant([gauss3d(0.7,3), gauss3d(0.7, 5), gauss3d(0.6, 7)])
    
    augmented_imgs = np.ndarray((images.shape[0] * 10, images.shape[1], images.shape[2], images.shape[3]), dtype= 'float64')
    augmented_labels = np.ndarray((images.shape[0] * 10, 1), dtype = 'int32')
    
    for i in range(images.shape[0]):
        print("{}/{}".format(i, images.shape[0]))
        img_augmented = []
        count = 1
        idx = i * 10
        augmented_imgs[idx] = images[i]
        augmented_labels[idx] = labels[i]
        for s in shifts:
            augmented_imgs[idx + count] = scipy.ndimage.shift(images[i], s)
            augmented_labels[idx + count] = labels[i]
            print(idx + count)
            count += 1
        for g in gaussians:
            augmented_imgs[idx + count] = scipy.ndimage.convolve(images[i], gauss3d(g[0], g[1]))
            augmented_labels[idx + count] = labels[i]
            print(idx + count)           
            count += 1
        
    return augmented_imgs, augmented_labels

def augment_image(image, label):
    """ Data augmentation of an image """
    
    if len(image.shape) != 3:
        raise Exception("Image should have 3 dimensions, shape found: {}".format(image.shape))
        
    augmented_imgs = np.ndarray((20, image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    augmented_labels = np.full((20, 1),label, dtype='int32')
    
    shifts = [[1, 0, 0], [-1, 0, 0],
              [0, 1, 0], [0, -1, 0],
              [0, 0, 1], [0, 0, -1]]
    gaussians = [gauss3d(0.7,3), gauss3d(0.7, 5), gauss3d(0.6, 7)]
    
    idx = 1
    augmented_imgs[0] = image
    
    for s in shifts:
        augmented_imgs[idx] = scipy.ndimage.shift(image, s)
        idx += 1
            
    for g in gaussians:
        augmented_imgs[idx] = scipy.ndimage.convolve(image, g)
        idx += 1
        
    for i in range(10):
        augmented_imgs[10 + i] = np.flip(augmented_imgs[i,...], 0)
        
    return augmented_imgs, augmented_labels

def random_transformation(image):
    """ Randomly transforms an image """
    #print("image",type(image), image.shape)
    #image = image.reshape(image.shape[0], image.shape[1], image.shape[2])
    shifts = [[1, 0, 0, 0], [-1, 0, 0, 0],
              [0, 1, 0, 0], [0, -1, 0, 0],
              [0, 0, 1, 0], [0, 0, -1, 0]]
    gaussians = [gauss3d(0.7,3), gauss3d(0.7, 5), gauss3d(0.6, 7)]
    
    transformation = np.random.randint(10)
    
    
    if transformation < 6:
        # Imagen con traslación
        transformed_img = scipy.ndimage.shift(image, shifts[transformation])
        #print("shifted {}".format(shifts[transformation]), transformed_img.dtype)
        
    elif 6 <= transformation < 9: 
        # Imagen con filtro gaussiano
        kernel = gaussians[transformation - 6]
        #kernel = kernel.reshape(kernel.shape[0], kernel.shape[1], kernel.shape[2], 1)
        #print("Kernel", kernel.shape)
        transformed_img = scipy.ndimage.convolve(image, kernel.reshape(kernel.shape[0],kernel.shape[0],kernel.shape[0],1))
        #print("smoothed {}".format(transformation), transformed_img.dtype) 
    else:
        # Imagen original
        transformed_img = image
        #print("No transform",transformed_img.dtype)
        
    if np.random.randint(2):
        # Imagen invertida
        transformed_img = np.flip(transformed_img, 0)
        #print("Flipped", transformed_img.dtype)
    
    return transformed_img

def load_img(path):
    return np.nan_to_num(nib.load(os.path.join(path)).get_fdata(), False).astype('float32')

def load_data_from_path(path):
    """ Loads Nifti data from path and returns a list of numpy arrays"""
    data= []
    for img in os.listdir(path):
        data.append(np.nan_to_num(nib.load(os.path.join(path, img)).get_fdata(), False).astype('float32'))
    return data

def load_data(path, data):
    images = []
    for img in data:       
        images.append(np.nan_to_num(nib.load(os.path.join(path, img)).get_fdata(), False).astype('float32'))
    return images

def extend_class(data, size):
    """ Extends the data up to the size provided by randomly duplicating examples"""
    
    prev_size = len(data)
    while(len(data) < size):
        i = np.random.randint(prev_size)
        data.append(data[i])        
    return data




shifts = tf.constant([(1,1), (1,-1), (2,1), (2,-1), (3,1), (3,-1)])

kernel3 = tf.constant(gauss3d(0.7, 3).reshape(3,3,3,1,1))
kernel3 = tf.pad(kernel3, [[2,2],[2,2],[2,2],[0,0],[0,0]]) # Añadimos padding para que todos los filtros tengan el mismo tamaño y se puedan poner en un tensor

kernel5 = tf.constant(gauss3d(0.7, 5).reshape(5,5,5,1,1))
kernel5 = tf.pad(kernel5, [[1,1],[1,1],[1,1],[0,0],[0,0]])

kernel7 = tf.constant(gauss3d(0.6, 7).reshape(7,7,7,1,1))
gaussians = tf.stack([kernel3, kernel5, kernel7])


def transform(tensor, labels):     

    rand = tf.random.uniform([1], 0, 10, dtype='int32')[0]
    
    tensor = tf.reshape(tensor,[1,91,109,91,1])
    #print(len(shifts), len(gaussians))
    result = tf.cond( rand < 9,   
                 lambda : tf.cond(rand < 6,
                           lambda : tf.roll(tensor, shifts[rand, 1],shifts[rand, 0]),                                  
                           lambda : tf.nn.conv3d(tensor, gaussians[rand - 6], strides=[1,1,1,1,1], padding='SAME' )),
                 lambda : tensor)
    
    return tf.reshape(tf.cond(tf.random.uniform([1], 0, 2, dtype = 'int32') == 1, 
                              lambda : result, 
                              lambda: tf.reverse(result, [1])), [91, 109, 91, 1]), labels

def obtain_data_files(path, info_path):
    """ Devuelve 2 diccionarios: AD_CN y groups, en AD_CN están los nombres de los archivos diferenciandose con AD vs CN,
        en groups, estan los nombres de los archivos diferenciandose según la visita del estudio."""
    adni = pd.read_csv(info_path)
    files = os.listdir(path)
    groups = {}
    AD_CN = {"CN":[], "AD":[]}
    for file in files:
        image_id = file[-10:-4]
        record = adni[adni["Image ID"] == int(image_id)]
        group = str(record["Research Group"].values[0]) + " " + str(record["Visit"].values[0])
        AD_CN[str(record["Research Group"].values[0])].append(file)
        if group in groups:
            groups[group].append(file)
        else:
            groups[group] = [file]
    return AD_CN, groups

def filenames_labels(path, info_path):
    adni = pd.read_csv(info_path)
    files = os.listdir(path)
    groups = {}
    AD_CN = {"CN":[], "AD":[]}
    for file in files:
        image_id = file[-10:-4]
        record = adni[adni["Image ID"] == int(image_id)]
        group = str(record["Research Group"].values[0]) + " " + str(record["Visit"].values[0])
        AD_CN[str(record["Research Group"].values[0])].append(file)
        if group in groups:
            groups[group].append(file)
        else:
            groups[group] = [file]
    return AD_CN, groups


def test_val_train_split(CN_imgs, CN_labels, AD_imgs, AD_labels, test_percentage, val_percentaje, batch_size):
    
    test_idx = math.floor(test_percentage * len(CN_imgs))
    val_idx = math.floor(val_percentaje * len(CN_imgs)) + test_idx
    
    
    
    test_labels = np.concatenate((CN_labels[:test_idx], AD_labels[:test_idx]))
    test_size = len(test_labels)
    test_imgs = np.concatenate([CN_imgs[:test_idx], AD_imgs[:test_idx]]).reshape((test_size, 91, 109, 91, 1))
    
    

    val_labels = np.concatenate((CN_labels[test_idx : val_idx] , AD_labels[test_idx : val_idx]))
    val_size = len(val_labels)
    val_imgs = np.concatenate([CN_imgs[test_idx : val_idx], AD_imgs[test_idx : val_idx]]).reshape((val_size, 91, 109, 91, 1))



    train_labels = np.concatenate((CN_labels[val_idx:] , AD_labels[val_idx:]))
    train_size = len(train_labels)
    train_imgs = np.concatenate([CN_imgs[val_idx:], AD_imgs[val_idx:]]).reshape((train_size, 91, 109, 91, 1))
    
    mean = train_imgs.mean()
    std = train_imgs.std()
    
    train_imgs = (train_imgs - mean) / std
    val_imgs = (val_imgs - mean) / std
    test_imgs = (test_imgs - mean) / std
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels)).cache().shuffle(train_size)\
            .map(lambda tensor, labels : transform(tensor,labels), num_parallel_calls=16).batch(batch_size).prefetch(8)

    val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels)).shuffle(val_size).batch(val_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_labels)).shuffle(test_size).batch(test_size)
    print("Train size: ", train_size, "| Val size: ", val_size, "| Test size: ", test_size, "| Total: ", len(AD_imgs) + len(CN_imgs) )   
    print("Train images shape: ", train_imgs.shape, " train labels sum", train_labels.sum() )
    print("Val images shape: ", val_imgs.shape, " Val labels sum", val_labels.sum() )
    print("Test images shape: ", test_imgs.shape, " Test labels sum", test_labels.sum() )
    return {"test":test_ds, "val": val_ds, "train": train_ds, "prev_mean": mean, "prev_std": std, "post_mean": train_imgs.mean(), "post_std": train_imgs.std(),                           "train_size":train_size, "test_size":test_size, "val_size":val_size}


def train_test_split(CN_imgs, CN_labels, AD_imgs, AD_labels, test_percentage):
    """ Devuelve un dataset de entrenamiento y otro de test, sin ninguna modificación"""
    test_idx = math.floor(test_percentage * len(CN_imgs))
    test_labels = np.concatenate((CN_labels[:test_idx], AD_labels[:test_idx]))
    test_size = len(test_labels)
    test_imgs = np.concatenate([CN_imgs[:test_idx], AD_imgs[:test_idx]]).reshape((test_size, 91, 109, 91, 1))
    
    train_labels = np.concatenate((CN_labels[test_idx:] , AD_labels[test_idx:]))
    train_size = len(train_labels)
    train_imgs = np.concatenate([CN_imgs[test_idx:], AD_imgs[test_idx:]]).reshape((train_size, 91, 109, 91, 1))
    mean = train_imgs.mean()
    std = train_imgs.std()
    
    train_imgs = (train_imgs - mean) / std
    test_imgs = (test_imgs - mean) / std
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels)).cache().shuffle(train_size).repeat()    
    test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_labels)).shuffle(test_size)
    
    return { "train": train_ds, "train_size": train_size, 
            "test": test_ds, "test_size": test_size,  
            "train_imgs": train_imgs, "train_labels": train_labels, 
            "test_imgs": test_imgs, "test_labels": test_labels}

def k_fold(all_data, all_data_labels, n_folds, fold):
    if fold >= n_folds:
        raise Exception("fold no puede ser mayor o igual que n_folds ")
    idx = np.random.permutation(all_data.shape[0])
    all_data, all_data_labels = all_data[idx], all_data_labels[idx]
    fold_size = all_data.shape[0]// n_folds
    val_range = [fold * fold_size, (fold + 1) * fold_size]
    
    val_imgs = all_data[val_range[0]: val_range[1]]
    val_labels = all_data_labels[val_range[0]: val_range[1]]
    
    train_imgs = np.concatenate((all_data[:val_range[0]], all_data[val_range[1]:]))
    train_labels = np.concatenate((all_data_labels[:val_range[0]], all_data_labels[val_range[1]:]))
    
    val_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels)).cache().shuffle(train_imgs.shape[0])
    train_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels)).shuffle(val_imgs.shape[0])
                                                                       
    return {"train_ds": train_ds, "val_ds": val_ds, "train_size": train_imgs.shape[0], "val_size": val_imgs.shape[0]}
    
def visualizar(n, s, axis):   
    if axis == 0:        
        plt.imshow(n[s, :, :, 0 ])
    elif axis == 1:        
        plt.imshow(n[:, s, :, 0 ])
    elif axis == 2:        
        plt.imshow(n[:, :, s, 0 ])
        

import ipywidgets as ipyw
import matplotlib.pyplot as plt

class ImageSliceViewer3D:
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    """
    
    def __init__(self, volume, figsize=(8,8), cmap='plasma'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])
