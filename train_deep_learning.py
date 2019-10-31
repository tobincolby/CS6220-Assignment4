import tensorflow as tf
from datetime import datetime
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(rgb_images):
    images = []
    for rgb in rgb_images:
        images.append(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))
    return np.array(images)

adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

sgd = optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

adagrad = optimizers.Adagrad(learning_rate=0.01)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

test_images = test_images[:5555]
test_labels = test_labels[:5555]

train_images = train_images.reshape((50000, 32, 32, 3))
test_images = test_images.reshape((5555, 32, 32, 3))

train_images = rgb2gray(train_images)
test_images = rgb2gray(test_images)

train_images = train_images.reshape((50000, 32, 32, 1))
test_images = test_images.reshape((5555, 32, 32, 1))

model_summary = []

for num_epoch in [5, 10, 15, 20]:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    starttime = datetime.now()
    history = model.fit(train_images, train_labels, epochs=num_epoch,
                        validation_data=(test_images, test_labels))
    total_time = (datetime.now() - starttime).total_seconds()
    test_acc = history.history['accuracy'][-1]
    model.save('cifar10-epochs-' + str(num_epoch) + '.h5')
    model_summary.append("Epochs - " + str(num_epoch) + " - " + str(test_acc) + " - " + str(total_time))


    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


for batch_size in [12, 24, 32, 50, 64]:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    starttime = datetime.now()

    history = model.fit(train_images, train_labels, epochs=10, batch_size=batch_size,
                        validation_data=(test_images, test_labels))

    total_time = (datetime.now() - starttime).total_seconds()

    test_acc = history.history['accuracy'][-1]

    model_summary.append("Batches - " + str(batch_size) + " - " + str(test_acc) + " - " + str(total_time))

    model.save('cifar10-batches-' + str(batch_size) + '.h5')


for optimizer in [adam, adagrad, sgd]:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    starttime = datetime.now()

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    test_acc = history.history['accuracy'][-1]
    total_time = (datetime.now() - starttime).total_seconds()


    model.save('cifar10-optimizer-' + str(optimizer) + '.h5')

    model_summary.append("Optimizer - " + str(optimizer) + " - " + str(test_acc) + " - " + str(total_time))

print(model_summary)

for num_convolutional_layers in [1, 2, 3]:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(32, 32, 1)))
    if num_convolutional_layers == 1:
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    elif num_convolutional_layers == 2:
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    elif num_convolutional_layers == 3:
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    elif num_convolutional_layers == 4:
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    starttime = datetime.now()
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))
    test_acc = history.history['accuracy'][-1]
    total_time = (datetime.now() - starttime).total_seconds()

    model.save('cifar10-num-conv-layers-' + str(num_convolutional_layers) + '.h5')

    model_summary.append("CNN - " + str(num_convolutional_layers) + " - " + str(test_acc) + " - " + str(total_time))



for training_set_size in [10000, 20000, 30000, 40000, 50000]:
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    train_images = train_images[:training_set_size]
    train_labels = train_labels[:training_set_size]

    test_images = test_images[:5555]
    test_labels = test_labels[:5555]

    train_images = train_images.reshape((training_set_size, 32, 32, 3))
    test_images = test_images.reshape((5555, 32, 32, 3))

    train_images = rgb2gray(train_images)
    test_images = rgb2gray(test_images)

    train_images = train_images.reshape((training_set_size, 32, 32, 1))
    test_images = test_images.reshape((5555, 32, 32, 1))

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    starttime = datetime.now()

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))
    test_acc = history.history['accuracy'][-1]
    total_time = (datetime.now() - starttime).total_seconds()
    model.save('cifar10-num-conv-layers-' + str(training_set_size) + '.h5')
    model_summary.append("Training Size - " + str(training_set_size) + " - " + str(test_acc) + " - " + str(total_time))


print(model_summary)
