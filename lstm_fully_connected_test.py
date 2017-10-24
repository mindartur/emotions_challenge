from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense
import numpy as np

data_dim = 160
hidden_state_dim = data_dim / 4
timesteps = 200
nb_classes = 6


for timestamp in range(timestamps):
    encoder_a = Sequential()
    encoder_a.add(LSTM(hidden_state_dim, input_shape=(1, data_dim)))

    encoder_b = Sequential()
    encoder_b.add(LSTM(hidden_state_dim, input_shape=(1, data_dim)))

    encoder_c = Sequential()
    encoder_c.add(LSTM(hidden_state_dim, input_shape=(1, data_dim)))

    encoder_d = Sequential()
    encoder_d.add(LSTM(hidden_state_dim, input_shape=(1, data_dim)))

    merge = Sequential()
    merge.add(Merge([encoder_a, encoder_b, encoder_c, encoder_d], mode='concat'))



decoder.add(Dense(32, activation='relu'))
decoder.add(Dense(nb_classes, activation='softmax'))

decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# generate dummy training data
x_train_a = np.random.random((1000, timesteps, data_dim))
x_train_b = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, nb_classes))

# generate dummy validation data
x_val_a = np.random.random((100, timesteps, data_dim))
x_val_b = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, nb_classes))

decoder.fit([x_train_a, x_train_b], y_train,
            batch_size=64, nb_epoch=5, show_accuracy=True,
            validation_data=([x_val_a, x_val_b], y_val))