import os

import numpy as np

import training_generator

MODEL_DIR = "saved_reservoir"

class Reservoir:

    def __init__(
        self,
        input_size=5,
        reservoir_size=100,
        spectral_radius=0.9,
        input_scale=1.0,
    ):

        self.input_size = input_size
        self.reservoir_size = reservoir_size

        # Input weights
        self.Win = (
            np.random.randn(reservoir_size, input_size)
            * input_scale
        )

        # Recurrent weights
        self.W = np.random.randn(
            reservoir_size,
            reservoir_size
        )

        # Scale spectral radius
        eigvals = np.linalg.eigvals(self.W)

        current_radius = np.max(np.abs(eigvals))

        self.W *= spectral_radius / current_radius

        # Initial state
        self.state = np.zeros((reservoir_size, 1))

    def reset(self):
        self.state[:] = 0

    def load_weights(self, model_dir="saved_reservoir"):
        """Load saved reservoir weights from disk."""
        self.Win = np.load(os.path.join(model_dir, "Win.npy"))
        self.W = np.load(os.path.join(model_dir, "W.npy"))
        self.reservoir_size = self.W.shape[0]
        self.input_size = self.Win.shape[1]
        self.reset()

    def step(self, u):
        """
        u shape = (input_size,)
        """

        u = u.reshape(-1, 1)

        self.state = np.tanh(
            self.W @ self.state +
            self.Win @ u
        )

        return self.state
    
    def run_sequence(self, commands):

        states = []

        self.reset()

        for cmd in commands:

            state = self.step(cmd)

            states.append(state.flatten())

        return np.array(states)


def extract_examples(reservoir, commands_list, labels_list):
    X = []
    Y = []
    for commands, labels in zip(commands_list, labels_list):

        states = reservoir.run_sequence(commands)

        for state, label in zip(states, labels):

            X.append(state)
            Y.append(label)

    return np.array(X), np.array(Y)

def train_readout(X, Y):
    Wout = np.linalg.lstsq(X, Y, rcond=None)[0]
    return Wout

def predict(X, Wout):
    scores = X @ Wout

    # Pick the highest-scoring class index
    class_indices = np.argmax(scores, axis=1)

    return class_indices


def predict_sequence(reservoir, commands, Wout):
    """Run one command sequence through the reservoir and return one label list."""
    states = reservoir.run_sequence(commands)
    predictions = predict(states, Wout)
    return predictions.flatten()


if __name__ == "__main__":
    train_commands, train_labels = training_generator.load_data(5000, min_length=2, max_length=10)
    reservoir = Reservoir()

    X_train, Y_train = extract_examples(reservoir, train_commands, train_labels)

    print(X_train.shape, Y_train.shape)

    Wout = train_readout(X_train, Y_train)

    predictions = predict(X_train, Wout)

    predictions = training_generator.predictions_to_labels(predictions)

    # What does it do?
    # accuracy = np.mean(predictions == Y_train)
    # print("Accuracy:", accuracy)

    training_generator.get_accuracy(train_commands, train_labels, reservoir, Wout, debug=False)


    # Save the reservoir and readout weights after training
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_DIR, "Win.npy"), reservoir.Win)
    np.save(os.path.join(MODEL_DIR, "W.npy"), reservoir.W)
    np.save(os.path.join(MODEL_DIR, "Wout.npy"), Wout)
    print(f"Weights saved to {MODEL_DIR}/")

