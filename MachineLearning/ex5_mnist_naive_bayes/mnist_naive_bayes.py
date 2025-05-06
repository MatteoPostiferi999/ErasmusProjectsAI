import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
import numpy as np
import math

def class_acc(pred, gt):
    good_predictions = np.sum(pred == gt)
    accuracy = good_predictions / len(gt)
    return accuracy

def main():
    while True:
        dataset_choice = input('Choose a dataset: original/dataset')
        if dataset_choice in ['original', 'fashion']:
            break
        else:
            print("Invalid choice. Please enter 'original' or 'fashion'.")
    if dataset_choice == 'original':
        mnist = tf.keras.datasets.mnist
        print('original selected')
    elif dataset_choice == 'fashion':
        mnist = tf.keras.datasets.fashion_mnist
        print('fashion selected')


    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    new_x_train = x_train.reshape(60000, -1)
    new_x_test = x_test.reshape(10000, -1)
    new_x_train_noise = new_x_train + np.random.normal(loc=0.0, scale=25, size=new_x_train.shape)


    # Print the size of training and test data
    print(f'x_train shape {x_train.shape}')
    print(f'y_train shape {y_train.shape}')
    print(f'x_test shape {x_test.shape}')
    print(f'y_test shape {y_test.shape}')

    # Create a dictionary with 10 empty lists, one for each class
    samples_by_class = {}
    mean_by_class = {}
    variance_by_class = {}
    for i in range(10):
        samples_by_class[i] = []
        mean_by_class[i] = []
        variance_by_class[i] = []

    # Fill in the dictionary
    for i in range(new_x_train.shape[0]):
        label = y_train[i]
        samples_by_class[label].append(new_x_train_noise[i])

    # Find mean and variance vectors
    for i in range(10):
        mean_by_class[i] = np.mean(samples_by_class[i], axis=0)
        variance_by_class[i] = np.var(samples_by_class[i], axis=0)

    # Pre-calculate common values
    log_2pi = math.log(2 * math.pi)
    log_variance_by_class = {c: np.log(variance_by_class[c]) for c in variance_by_class}
    inv_variance_by_class = {c: 1 / (variance_by_class[c]) for c in variance_by_class}
    # Calculate log probabilities using vectorized operations
    log_probabilities = np.zeros((10, 10000))
    for c in range(10):
        log_probabilities[c] = -0.5 * (log_2pi + log_variance_by_class[c] + inv_variance_by_class[c] * (new_x_test - mean_by_class[c])**2).sum(axis=1)

    predicted_classes = np.argmax(log_probabilities, axis=0)

    for i in range(x_test.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1)
            plt.clf()
            plt.imshow(x_test[i], cmap='gray_r')
            plt.title(f"Image {i} label num {y_test[i]} predicted {predicted_classes[i]}")
            plt.pause(1)


    print('Classification accuracy is: ',class_acc(predicted_classes, y_test))

if __name__ == "__main__":
    main()