import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax activation function for the output layer (to get probabilities for multi-class classification)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Artificial neural network implementation with 3 hidden layers and softmax output
class PenguinNeuralNetworkThreeHiddenLayers:
    def __init__(self):
        np.random.seed(1)
        
        # Weights and biases for first hidden layer (4 input nodes, 5 hidden neurons)
        self.weights_input_hidden1 = np.random.rand(4, 5)
        self.bias_hidden1 = np.random.rand(1, 5)
        
        # Weights and biases for second hidden layer (5 neurons from first layer, 5 neurons in second layer)
        self.weights_hidden1_hidden2 = np.random.rand(5, 5)
        self.bias_hidden2 = np.random.rand(1, 5)

        ### demo activation function not working
        self.weights_hidden1_hidden2[:, 2] = -500.0  # Large negative weights for the 3rd neuron
        #self.bias_hidden2[0, 2] = -800.0
        
        # Weights and biases for third hidden layer (5 neurons from second layer, 5 neurons in third layer)
        self.weights_hidden2_hidden3 = np.random.rand(5, 5)
        self.bias_hidden3 = np.random.rand(1, 5)
        
        # Weights and biases for output layer (5 neurons from third layer, 3 output neurons for each species)
        self.weights_hidden3_output = np.random.rand(5, 3) 
        self.bias_output = np.random.rand(1, 3) 
    
    def feedforward(self, X):
        # First hidden layer calculations
        z_hidden1 = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        a_hidden1 = sigmoid(z_hidden1)  # Apply activation function
        
        # Second hidden layer calculations
        z_hidden2 = np.dot(a_hidden1, self.weights_hidden1_hidden2) + self.bias_hidden2
        a_hidden2 = sigmoid(z_hidden2)  # Apply activation function
        
        # Third hidden layer calculations
        z_hidden3 = np.dot(a_hidden2, self.weights_hidden2_hidden3) + self.bias_hidden3
        a_hidden3 = sigmoid(z_hidden3)  # Apply activation function
        
        # Output layer calculations
        z_output = np.dot(a_hidden3, self.weights_hidden3_output) + self.bias_output
        a_output = softmax(z_output)  # Apply softmax activation function to get probabilities
        
        return a_hidden1, a_hidden2, a_hidden3, a_output

# Sample input data for penguin (4 features)
X = np.array([[37.3, 16.8, 19.2, 30.0]])  # Bill length, Bill depth, Flipper length, Weight

X_normalized = (X - np.mean(X)) / np.std(X)

# Create a neural network instance
nn = PenguinNeuralNetworkThreeHiddenLayers()

# Feedforward the input through the network
hidden_layer1_output, hidden_layer2_output, hidden_layer3_output, output_layer_output = nn.feedforward(X_normalized)

# Print the outputs
print("First hidden layer output (after activation):\n", hidden_layer1_output)
print("\nSecond hidden layer output (after activation):\n", hidden_layer2_output)
print("\nThird hidden layer output (after activation):\n", hidden_layer3_output)
print("\nOutput layer probabilities (after softmax):\n", output_layer_output)
