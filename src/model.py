import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm(input_shape, lstm_units=64, dropout=0.2, lr=0.001):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)
    
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model