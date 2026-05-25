import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm(input_shape, lstm_units=64, dropout=0.2, lr=0.001):
    inp = layers.Input(shape=input_shape)

    # Adding L2 Regularization to LSTM layers
    regularizer = tf.keras.regularizers.l2(1e-4)

    x = layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizer)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, kernel_regularizer=regularizer)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)
    
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model