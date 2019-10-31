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

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

test_images = test_images[:5555]
test_labels = test_labels[:5555]

test_images = test_images.reshape((5555, 32, 32, 3))
test_images = rgb2gray(test_images)

test_images = test_images.reshape((5555, 32, 32, 1))

model_summary = []

for num_epoch in [5, 10, 15, 20]:
    model = models.load_model("cifar10-epochs-"+str(num_epoch)+".h5")

    starttime = datetime.now()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    total_time = (datetime.now() - starttime).total_seconds()

    model_summary.append("Epochs - " + str(num_epoch) + " - " + str(test_acc) + " - " + str(test_loss) + " - " + str(total_time))


for batch_size in [12, 24, 32, 50, 64]:
    model = models.load_model("cifar10-batches-"+str(batch_size)+".h5")

    starttime = datetime.now()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    total_time = (datetime.now() - starttime).total_seconds()

    model_summary.append("Batches - " + str(batch_size) + " - " + str(test_acc) + " - " + str(test_loss) + " - " + str(total_time))

for optimizer in ['adam', 'adagrad', 'sgd']:
    model = models.load_model("cifar10-optimizer-"+str(optimizer)+".h5")

    starttime = datetime.now()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    total_time = (datetime.now() - starttime).total_seconds()

    model_summary.append("Optimizer - " + str(optimizer) + " - " + str(test_acc) + " - " + str(test_loss) + " - " + str(total_time))


for num_convolutional_layers in [1, 2, 3]:
    model = models.load_model("cifar10-num-conv-layers-"+str(num_convolutional_layers)+".h5")

    starttime = datetime.now()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    total_time = (datetime.now() - starttime).total_seconds()

    model_summary.append("CNN - " + str(num_convolutional_layers) + " - " + str(test_acc) + " - " + str(test_loss) + " - " + str(total_time))

for training_set_size in [10000, 20000, 30000, 40000, 50000]:
    model = models.load_model("cifar10-num-conv-layers-"+str(training_set_size)+".h5")

    starttime = datetime.now()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    total_time = (datetime.now() - starttime).total_seconds()

    model_summary.append("Training Set Size - " + str(training_set_size) + " - " + str(test_acc) + " - " + str(test_loss) + " - " + str(total_time))

print(model_summary)
