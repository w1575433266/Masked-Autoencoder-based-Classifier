import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, matthews_corrcoef
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, ReLU, Softmax
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
df = pd.read_excel('datasets.xlsx')


scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(df.drop("category", axis=1))

category_encoder = OneHotEncoder(sparse=False)
category_onehot = category_encoder.fit_transform(df["category"].values.reshape(-1, 1))

mask = np.ones(len(df))
mask[df["category"] == 0] = 0

input_dim = data_rescaled.shape[1]
category_dim = category_onehot.shape[1]


results = []

for i in range(20):

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

    train_indices = df["category"] != 0
    classifier.fit(data_encoded[train_indices], category_onehot[train_indices], epochs=200, batch_size=16, shuffle=True)

    y_true_train = np.argmax(category_onehot[train_indices], axis=1)
    y_pred_train = np.argmax(classifier.predict(data_encoded[train_indices]), axis=1)

    accuracy_train = accuracy_score(y_true_train, y_pred_train)
    precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_true_train, y_pred_train, average='macro')
    mcc_train = matthews_corrcoef(y_true_train, y_pred_train)

    results.append({
        'Iteration': i + 1,
        'Accuracy': accuracy_train,
        'Precision': precision_train,
        'Recall': recall_train,
        'F1 Score': f1_train,
        'MCC': mcc_train,
    })

results_df = pd.DataFrame(results)

averages = results_df.mean()
averages['Iteration'] = 'Average'
results_df = results_df.append(averages, ignore_index=True)

results_df.to_excel('model_performance.xlsx', index=False)

print(results_df)
