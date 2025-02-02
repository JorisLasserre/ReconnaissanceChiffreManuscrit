import os
import numpy as np
from pictureTreatment import *



def count_pixel_in_bloc(image, num_blocks):
    hauteur, largeur = image.shape
    block_height = hauteur // num_blocks
    block_width = largeur // num_blocks
    result = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            count = 0
            for row in range(i * block_height, (i + 1) * block_height):
                for col in range(j * block_width, (j + 1) * block_width):
                    if image[row, col] == 0:  
                        count += 1
            result.append(count)

    return result

def add_vector_into_dict(picture, number_block):
    dictio = {}
    i = 0
    for image, filename in picture:
        i+=1
        label = filename[0]
        vector = count_pixel_in_bloc(image, number_block)
        if label not in dictio:
            dictio[label] = []
        dictio[label].append(vector)
    return dictio

def euclidian_distance(vector1, vector2):
    return np.sqrt(np.sum((np.array(vector1) - np.array(vector2)) ** 2))


def calculate_all_vector_distance(input_vector, dict):
    distance_dict = {}
    
    for label, vectors in dict.items():
        distances = []
        for vector in vectors:
            distance = euclidian_distance(input_vector, vector)
            if distance != 0 :
                distances.append(distance)
        distance_dict[label] = distances
    
    return distance_dict

def get_k_nearest_neighbors(distance_dict, k):
    k_nearest_neighbors = []
    for label, distances in distance_dict.items():
        for distance in distances:
            k_nearest_neighbors.append((distance, label))
    
    k_nearest_neighbors.sort(key=lambda x: x[0])
    k_nearest = k_nearest_neighbors[:k]
    
    return k_nearest


def prediction(k_nearest):
    label_count = {}
    for _, label in k_nearest:
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    
    predicted_label = max_dictio(label_count)
    return predicted_label
    

def max_dictio(dictio):
    max = 0
    label = 0
    for key, value in dictio.items():
        if value > max:
            max = value
            label = key
    return label

def matrice_confusion(data_test, data_train, k):
    matrice_confusion = np.zeros((10, 10), dtype=int)
    total = 0
    correct = 0
    
    for label, vectors in data_test.items():
        for vector in vectors:
            distance_dict = calculate_all_vector_distance(vector, data_train)
            k_nearest = get_k_nearest_neighbors(distance_dict, k)
            predicted_label = prediction(k_nearest)
            matrice_confusion[int(label)][int(predicted_label)] += 1
            total += 1
            if label == predicted_label:
                correct += 1
    
    accuracy = correct / total
    return matrice_confusion, accuracy

def print_matrice_confusion(matrice, accuracy):
    
    fig, ax = plt.subplots()

    ax.matshow(matrice, cmap='Oranges')

    for i in range(matrice_confusion.shape[0]):
        for j in range(matrice_confusion.shape[1]):
            c = matrice[j,i]
            ax.text(i, j, str(c), va='center', ha='center')

    plt.show()
    print(f"\nLe taux de reconnaissance est de {accuracy * 100:.2f}%")

if __name__ == "__main__":
    k = 3
    number_block = 5
    structurant_element = 3
    resize_value = (60, 60)
    bdd = os.path.join(os.path.dirname(__file__), 'baseTest')
    
    pictureList = importPicture(bdd)    
    pictureList = all_picture_treatment(pictureList, resize_value, structurant_element)
    pictureList = add_vector_into_dict(pictureList, number_block)

    matrice_confusion, accuracy = matrice_confusion(pictureList, pictureList, k)

    print_matrice_confusion(matrice_confusion, accuracy)