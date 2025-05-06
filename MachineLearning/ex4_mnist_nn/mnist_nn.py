import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def class_acc(pred, gt):
    good_predictions = np.sum(pred == gt)
    accuracy = good_predictions / len(gt)
    return accuracy


def main():
    while True:
        dataset_choice = input('Choose a dataset: original/fashion --->')
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

    print(f'x_train shape {new_x_train.shape}')
    print(f'y_train shape {y_train.shape}')
    print(f'x_test shape {new_x_test.shape}')
    print(f'y_test shape {y_test.shape}')

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(new_x_train, y_train)
    y_pred = knn.predict(new_x_test)

    for i in range(x_test.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1);
            plt.clf()
            plt.imshow(x_test[i], cmap='gray_r')
            plt.title(f"Image {i} label num {y_test[i]} predicted {y_pred[i]}")
            plt.pause(1)

    print('Classification accuracy is: ', class_acc(y_pred, y_test))


if __name__ == "__main__":
    main()
