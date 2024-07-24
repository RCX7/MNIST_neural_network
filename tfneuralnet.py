import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import activations
from keras import layers
from keras import models



train_file = r"archive (1)\mnist_train.csv"
test_file = r"archive (1)\mnist_test.csv"

training_set = pd.read_csv(train_file)

training_set = np.array(training_set)

'''
Creating a dev set by randomly shuffling the data and selecting the first 15000/60000
'''
np.random.shuffle(training_set)

dev_set = np.array(training_set[:15000])
training_set = training_set[15000:]

'''
Transposing the data sets to make it matrix multiplication friendly
'''

training_set = np.transpose(training_set)

training_set_y = training_set[0]
training_set_x = training_set[1:]

#training_set_x = tf.stack(training_set_x)
#training_set_y = tf.stack(training_set_y)


dev_set = np.transpose(dev_set)
dev_set_y = dev_set[0]
dev_set_x = dev_set[1:]

training_set_x = training_set_x / 255.0
dev_set_x = dev_set_x / 255.0

n, m = training_set_x.shape

print(f"Training set x shape: {training_set_x.shape}")
print(f"Training set y shape: {training_set_y.shape}")

'''
Training the model for weights to test with forward prop of numpy
'''


model = models.Sequential([
    layers.Dense(units = 15, activation = "relu", input_shape=(784,), name="layer1"),
    layers.Dense(units = 10, activation = "relu", name="layer2"),
    layers.Dense(units = 10, activation = "softmax", name="layer3")
])

model.compile(
    loss= keras.losses.SparseCategoricalCrossentropy,
    optimizer= keras.optimizers.Adam(learning_rate = 0.005))

model.fit(
    np.transpose(training_set_x), training_set_y,
    verbose=1,
    epochs=50
)

def one_hot_enc(set):
    '''
    Inputs: A column vector (y) of labels 0-9
    Outputs: A one-hot encoded version of the same thing; a matrix of size (m, 10)
    '''
    encoded = np.eye(10)[set]
    encoded = encoded.reshape(m, 10)

    return encoded.T

def compute_cost(predicted, actual, epsilon):
    '''
    Inputs: A predicted cost, the actual cost, and epsilon
    '''
 
    actual = one_hot_enc(actual) #Inputs the y labels (shape(45000,1)) and one hot encodes them to (shape(10, 45000))

    predicted = np.clip(predicted, epsilon, 1 - epsilon)

    loss_matrix = actual * -np.log(predicted.T) #Multplies the one hot encoding to the predicted values
    cost = np.sum(loss_matrix)/actual.shape[1] 

    return cost


predictions = model.predict(training_set_x.T)
np.savetxt("tfpredictions.csv", predictions)

print("Cost:", compute_cost(predictions, training_set_y, 1e-8))

model.summary()


first_layer_weights = model.layers[0].get_weights()[0]

first_layer_weights = pd.DataFrame(first_layer_weights, dtype=np.float64)
first_layer_weights.to_csv("layer1weights.csv")
#np.savetxt("layer1weights.csv", first_layer_weights, dtype=np.float64)


first_layer_biases  = model.layers[0].get_weights()[1]

first_layer_biases = pd.DataFrame(first_layer_biases, dtype=np.float64)
first_layer_biases.to_csv("layer1bias.csv")
#np.savetxt("layer1bias.csv", first_layer_biases, dtype=np.float64)


second_layer_weights = model.layers[1].get_weights()[0]

second_layer_weights = pd.DataFrame(second_layer_weights, dtype=np.float64)
second_layer_weights.to_csv("layer2weights.csv")
#np.savetxt("layer2weights.csv", second_layer_weights, dtype=np.float64)

second_layer_biases  = model.layers[1].get_weights()[1]

second_layer_biases = pd.DataFrame(second_layer_biases, dtype=np.float64)
second_layer_biases.to_csv("layer2bias.csv")
#np.savetxt("layer2bias.csv", second_layer_biases, dtype=np.float64)

third_layer_weights = model.layers[2].get_weights()[0]
third_layer_weights = pd.DataFrame(third_layer_weights, dtype=np.float64)
third_layer_weights.to_csv("layer3weights.csv")
#np.savetxt("layer3weights.csv", third_layer_weights, dtype=np.float64)

third_layer_biases  = model.layers[2].get_weights()[1]

third_layer_biases = pd.DataFrame(third_layer_biases, dtype=np.float64)
third_layer_biases.to_csv("layer3bias.csv")
#np.savetxt("layer3bias.csv", third_layer_biases, dtype=np.float64)
