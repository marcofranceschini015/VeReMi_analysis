import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from model.MLP import *
#import code #code.interact(local=dict(globals(), **locals()))

# Load data
X = np.load('X.npy')
y = np.load('y.npy')

# Build the model that you prefer
model, X = build_mlp(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Stop when the model is not learning anymore
early_stopping = EarlyStopping(
    monitor='accuracy',      # Monitor model's validation loss
    min_delta=0.001,         # Minimum change to qualify as an improvement
    patience=10,             # Stop after 10 epochs without improvement
    verbose=1,               # Print logs
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Fit the models
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test),  callbacks=[early_stopping] )

# Evaluate the model
performance = model.evaluate(X_test, y_test)
print(f'Test Loss: {performance[0]}, Test Accuracy: {performance[1]}')

