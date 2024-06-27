from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

def build_mlp(X):
    X_flattened, time_steps, features = flatten_data(X)

    #TODO try dropout
    model = Sequential([
    Dense(128, input_shape=(time_steps * features,), activation="relu"),
    Dense(256, activation="relu"),
    Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy']) #TODO try Precision and Recall
    return model, X_flattened


def flatten_data(X):
    num_samples, time_steps, features = X.shape
    X_flattened = X.reshape(num_samples, time_steps * features)
    return X_flattened, time_steps, features