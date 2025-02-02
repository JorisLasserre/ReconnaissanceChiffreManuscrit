import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import os
import random

#importation des images (pour le moment je ne traite pas les + et -)
def importPicture(picture_path):
    pictureList = []
    for file in os.listdir(picture_path):
        if not (file.startswith('-') or file.startswith('+')):    
            path = os.path.join(picture_path, file)  
            label = file[0]  
            picture = ski.io.imread(path)
            pictureList.append((picture, label))

    return pictureList


#redimensionnement des images 
def resizeAllPIcture(pictureList, newSize):
    resizedList = []
    for picture, filename in pictureList:
        resizedPicture = ski.transform.resize(picture, newSize, anti_aliasing=True)
        resizedList.append((resizedPicture, filename))

    return resizedList

#converti les images en noir et blanc "pur" (binarisation)
def blackWhiteConverter(pictureList):
    blackWHitePictureList = []
    for picture, filename in pictureList:
        picture = rgb2gray(picture[..., :3]) 
        
        binaryPicture = np.zeros_like(picture)
        for colonne in range(picture.shape[0]):
            for ligne in range(picture.shape[1]):

                if picture[colonne,ligne] > 128/255:
                    binaryPicture[colonne,ligne] = 255
                else:
                    binaryPicture[colonne,ligne] = 0
                    
        blackWHitePictureList.append((binaryPicture, filename))
    return blackWHitePictureList
    
def erosion(pictureList, structurant_element):
    erosionList = []
    for picture, filename in pictureList:
        erosionPicture = ski.morphology.erosion(picture, ski.morphology.square(structurant_element))
        erosionList.append((erosionPicture, filename))
    return erosionList 

def dilatation(pictureList, structurant_element):
    dilatationList = []
    for picture, filename in pictureList:
        dilatationPicture = ski.morphology.dilation(picture, ski.morphology.square(structurant_element))
        dilatationList.append((dilatationPicture, filename))
    return dilatationList

def all_picture_treatment(pictureList, newSize, structurant_element):
    pictureList = resizeAllPIcture(pictureList, newSize)
    pictureList = blackWhiteConverter(pictureList)
    pictureList = erosion(pictureList, structurant_element)
    pictureList = dilatation(pictureList, structurant_element)
    return pictureList