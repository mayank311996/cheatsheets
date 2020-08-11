#########################################################################################
# Chapter 9
#########################################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM

#########################################################################################
MAXLEN = 400
BATCH_SIZE = 32
EMBEDDING_DIMS = 300
EPOCHS = 2
NUM_NEURONS = 50

model = Sequential()
model.add(
    LSTM(
        NUM_NEURONS,
        return_sequences=True,
        input_shape=(MAXLEN, EMBEDDING_DIMS)
    )
)
model.add(
    Dropout(0.2)
)
model.add(
    Flatten()
)
model.add(
    Dense(1, activation="sigmoid")
)
model.compile(
    "rmsprop",
    "binary_crossentropy",
    metrics=["accuracy"]
)
print(model.summary())

