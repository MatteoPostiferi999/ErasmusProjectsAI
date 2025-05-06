import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
import numpy as np
from scipy.stats import multivariate_normal


def class_acc(pred, gt):
    good_predictions = np.sum(pred.astype(int) == gt.astype(int))
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


    (x_train,y_train), (x_test, y_test) = mnist.load_data()
    new_x_train = x_train.reshape(60000, -1)
    new_x_test = x_test.reshape(10000, -1)
    new_x_train_noise = np.array(new_x_train + np.random.normal(loc=0.0, scale=20, size=new_x_train.shape))


    # Print the size of training and test data
    print(f'x_train shape {new_x_train_noise.shape}')
    print(f'y_train shape {y_train.shape}')
    print(f'x_test shape {new_x_test.shape}')
    print(f'y_test shape {y_test.shape}')

    class_probabilities = np.zeros((10, 10000))
    for cl in range(10):
       mean = new_x_train_noise[y_train == cl].mean(axis=0)
       cov = np.cov(new_x_train_noise[y_train == cl], rowvar=False)
       probability = multivariate_normal.logpdf(new_x_test, mean, cov)
       class_probabilities[cl] = probability

    predicted_classes = np.argmax(class_probabilities, axis=0)

    for i in range(x_test.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1)
            plt.clf()
            plt.imshow(x_test[i], cmap='gray_r')
            # Access the correct label using the index i
            plt.title(f"Image {i} label num {y_test[i]} predicted {predicted_classes[i]}")
            plt.pause(1)

    print('accuracy: ', class_acc(predicted_classes, y_test))

if __name__ == "__main__":
    main()