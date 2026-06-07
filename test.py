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

# predictions = main.predict(X_test, Wout)
# accuracy = np.mean(predictions == Y_test)

# print(f"Test examples: {X_test.shape[0]}")
# print(f"Accuracy: {accuracy:.4f}")

for commands, labels in zip(test_commands[:5], test_labels[:5]):
    readable_commands = training_generator.commands_to_words(commands)
    predicted_labels = main.predict_sequence(reservoir, commands, Wout)

    print("Commands:    ", readable_commands)
    print("True labels: ", labels.tolist())
    print("Predictions: ", predicted_labels.tolist())
    print()

