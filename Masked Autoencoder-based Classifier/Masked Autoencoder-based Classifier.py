import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, ReLU, Softmax
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

results = pd.DataFrame()

num_runs = 20

df = pd.read_excel('datasets.xlsx')

scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(df.drop("category", axis=1))

category_encoder = OneHotEncoder(sparse=False)
category_onehot = category_encoder.fit_transform(df["category"].values.reshape(-1, 1))

mask = np.ones(len(df))
mask[df["category"]==0] = 0

input_dim = data_rescaled.shape[1]
category_dim = category_onehot.shape[1]

for run in range(num_runs):
    print(f"Running model iteration {run+1}/{num_runs}")

    input_layer = Input(shape=(input_dim,))
    category_layer = Input(shape=(category_dim,))
    mask_layer = Input(shape=(1,))
    encoder_input = concatenate([input_layer, category_layer])
    encoder_layer = Dense(3, activation='sigmoid')(encoder_input)
    decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)

    autoencoder = Model(inputs=[input_layer, category_layer, mask_layer], outputs=decoder_layer)

    encoder = Model(inputs=[input_layer, category_layer, mask_layer], outputs=encoder_layer)

    def masked_loss(y_true, y_pred):
        mask_value = 0
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return tf.keras.losses.mean_squared_error(y_true * mask, y_pred * mask)

    autoencoder.compile(optimizer=Adam(), loss=masked_loss)

    autoencoder.fit([data_rescaled, category_onehot, mask], data_rescaled, epochs=100, batch_size=1, shuffle=True)

    data_encoded = encoder.predict([data_rescaled, category_onehot, mask])

    encoder_output_dim = data_encoded.shape[1]

    classifier_input = Input(shape=(encoder_output_dim,))

    hidden_layer = Dense(128)(classifier_input)
    hidden_layer = ReLU()(hidden_layer)

    output_layer = Dense(category_dim)(hidden_layer)
    output_layer = Softmax()(output_layer)

    classifier = Model(inputs=classifier_input, outputs=output_layer)

    classifier.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    classifier.fit(data_encoded[df["category"]!=0], category_onehot[df["category"]!=0], epochs=200, batch_size=16, shuffle=True)

    predictions = classifier.predict(data_encoded[df["category"]==0])

    #zero_index = np.where(category_encoder.categories_[0] == 0)[0][0]
    #predictions[:, zero_index] = 0
    #predictions /= np.sum(predictions, axis=1, keepdims=True)

    predicted_categories_indices = np.argmax(predictions, axis=1)
    predicted_categories = category_encoder.categories_[0][predicted_categories_indices]

    for i, prediction in enumerate(predictions):
        result = {"Run": run+1, "Sample": i+1}
        for category, probability in zip(category_encoder.categories_[0], prediction):
            result[f"Probability of Category {category}"] = probability
        results = results.append(result, ignore_index=True)


with pd.ExcelWriter('Masked Autoencoder-based Classifier-result.xlsx') as writer:
    results.to_excel(writer, index=False)
