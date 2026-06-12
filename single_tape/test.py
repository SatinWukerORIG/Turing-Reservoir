import os
import numpy as np
import training_generator
import train

test_commands, test_labels = training_generator.load_data(1000)

reservoir = train.Reservoir()
reservoir.load_weights(train.MODEL_DIR)

print(len(test_commands), len(test_labels))
X_test, Y_test = train.extract_examples(
    reservoir,
    test_commands,
    test_labels
)

print(X_test.shape, Y_test.shape)

Wout = np.load(os.path.join(train.MODEL_DIR, "Wout.npy"))


training_generator.get_accuracy(test_commands, test_labels, reservoir, Wout, debug=True)
