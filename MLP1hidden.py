import numpy as np
import matplotlib.pyplot as plt


class MLP_oh:
    def __init__(self, input_size, hidden_size1, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.output_size = output_size

        # Initialize weights and biases for hidden and output layers
        self.weights_hidden1 = np.random.randn(input_size, hidden_size1)
        self.bias_hidden1 = np.zeros((1, hidden_size1))
        self.weights_output = np.random.randn(hidden_size1, output_size)
        self.bias_output = np.zeros((1, output_size))

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def accuracy(self, y_true, y_pred):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        return np.mean(y_true == y_pred)

    def forward(self, X):
        # Forward pass through the network
        self.hidden_input1 = np.dot(X, self.weights_hidden1) + self.bias_hidden1
        self.hidden_output1 = self.sigmoid(self.hidden_input1)
        self.output = self.sigmoid(np.dot(self.hidden_output1, self.weights_output) + self.bias_output)
        return self.output

    def backward(self, X, y, learning_rate):

        # Calculate loss
        loss = self.mse_loss(y, self.output)

        # Compute gradients for output layer
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        # Update weights and biases for output layer
        self.weights_output += learning_rate * np.dot(self.hidden_output1.T, output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        # Compute gradients for hidden layer 1
        hidden_error1 = np.dot(output_delta, self.weights_output.T) * self.sigmoid_derivative(self.hidden_output1)

        # Update weights and biases for hidden layer 1
        self.weights_hidden1 += learning_rate * np.dot(X.T, hidden_error1)
        self.bias_hidden1 += learning_rate * np.sum(hidden_error1, axis=0, keepdims=True)

        return loss

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate, early_stop_patience):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Forward and backward pass for training set
            train_output = self.forward(X_train)
            train_loss = self.backward(X_train, y_train, learning_rate)
            train_losses.append(train_loss)

            # Forward pass for validation set
            val_output = self.forward(X_val)
            val_loss = self.mse_loss(y_val, val_output)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter == early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        # print the train accuracy
        train_acc = self.accuracy(y_train, self.sigmoid(train_output))
        print(f"\nTrain Acc: {train_acc}")

        # Visualize loss
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def test(self, X_test, y_test):
        # Forward pass for test set
        test_output = self.forward(X_test)
        test_loss = self.mse_loss(y_test, test_output)
        test_acc = self.accuracy(y_test, self.sigmoid(test_output))
        print(f"Test Loss: {test_loss}")
        print(f"Test Acc: {test_acc}")
