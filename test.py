import os
import numpy as np
import training_generator
import main

test_commands, test_labels = training_generator.load_data(1000)

reservoir = main.Reservoir()
reservoir.load_weights("saved_reservoir")

print(len(test_commands), len(test_labels))
X_test, Y_test = main.extract_examples(
    reservoir,
    test_commands,
    test_labels
)

print(X_test.shape, Y_test.shape)

Wout = np.load(os.path.join("saved_reservoir", "Wout.npy"))


training_generator.get_accuracy(test_commands, test_labels, reservoir, Wout, debug=True)
