import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def main():
    while True:
        dataset_choice = input('Choose a dataset: original/fashion ---> ')
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

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    new_x_train = x_train.reshape(60000, -1)
    new_x_test = x_test.reshape(10000, -1)
    new_x_train = new_x_train / 255.0
    new_x_test = new_x_test / 255.0

    print(f'x_train shape {new_x_train.shape}')
    print(f'y_train shape {y_train.shape}')
    print(f'x_test shape {new_x_test.shape}')
    print(f'y_test shape {y_test.shape}\n')

    # Convert class labels to one-hot encoding
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Create the neural network model
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(784,)))
    model.add(Dense(num_classes, activation='sigmoid'))

    # Compile the model with categorical_crossentropy loss and accuracy metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with verbose set to 1 to record history
    tr_hist = model.fit(new_x_train, y_train, epochs=50, verbose=1)

    # Plot the training loss curve
    plt.plot(tr_hist.history['loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    train_acc = tr_hist.history['accuracy'][-1]  # Get the last accuracy value
    print(f"Final Training Accuracy: {(train_acc * 100):.3f}%")

    test_loss, test_acc = model.evaluate(new_x_test, y_test)
    print(f"Classification accuracy: {(test_acc * 100):.3f}%")


if __name__ == "__main__":
    main()
