import numpy as np 
import random 
np.random.seed(2017)
import pandas as pd  
from sklearn.model_selection import train_test_split
from subprocess import check_output


train = pd.read_json("/home/gasimov_aydin/Statoil/data/processed/train.json")
test = pd.read_json("/home/gasimov_aydin/Statoil/data/processed/test.json")


train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

sum_score = 0
temp = 0

print("done!")

# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
                             , x_band2[:, :, :, np.newaxis]
                             , ((x_band1 + x_band2) / 2)[:, :, :, np.newaxis]], axis=-1)
X_angle_train = np.array(train.inc_angle)
y_train = np.array(train["is_iceberg"])

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis]
                            , x_band2[:, :, :, np.newaxis]
                            , ((x_band1 + x_band2) / 2)[:, :, :, np.newaxis]], axis=-1)
X_angle_test = np.array(test.inc_angle)

X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train
                                                                                    , X_angle_train, y_train,
                                                                                    random_state=1235, train_size=0.90)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras import regularizers


def get_callbacks(filepath, patience=20):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    r_lr = ReduceLROnPlateau(monitor='val_loss', 
                              patience=10, min_lr=0.00001)
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave, r_lr]


def get_model(i):
    momentum_lr = 0
    lr_rate = 0
    dr_img_1 = 0
    dr_img_2 = 0
    dr_end = 0
    dense_coef = 0
    decay_lr = 0

    if (i == 0):
        momentum_lr = 0.91
        lr_rate = 0.0001
        dr_img_1 = 0.1
        dr_img_2 = 0.1
        dr_end = 0.5
        dense_coef = 64
        decay_lr = 0.0
    if (i == 1):
        momentum_lr = 0.92
        lr_rate = 0.0001
        dr_img_1 = 0.1
        dr_img_2 = 0.1
        dr_end = 0.5
        dense_coef = 64
        decay_lr = 0.0
    if (i == 2):
        momentum_lr = 0.93
        lr_rate = 0.0001
        dr_img_1 = 0.1
        dr_img_2 = 0.1
        dr_end = 0.5
        dense_coef = 64
        decay_lr = 0.0
    if (i == 3):
        momentum_lr = 0.91
        lr_rate = 0.0001
        dr_img_1 = 0.2
        dr_img_2 = 0.2
        dr_end = 0.6
        dense_coef = 64
        decay_lr = 0.0005
    if (i == 4):
        momentum_lr = 0.99
        lr_rate = 0.0001
        dr_img_1 = 0.2
        dr_img_2 = 0.2
        dr_end = 0.5
        dense_coef = 32
        decay_lr = 0.0005

    bn_model = momentum_lr
    p_activation = "elu"
    drop_img1 = dr_img_1
    drop_img2 = dr_img_2

    input_1 = Input(shape=(75, 75, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")

    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(drop_img1)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    #    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(drop_img1)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(drop_img1)(img_1)
    img_1 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(drop_img1)(img_1)
    #  img_1 = Conv2D(256, kernel_size = (3,3), activation=p_activation) (img_1)
    # img_1 = MaxPooling2D((2,2)) (img_1)
    # img_1 = Dropout(0.2)(img_1)
    img_1 = GlobalMaxPooling2D()(img_1)

    # img_2 = Conv2D(16, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    #   img_2 = Conv2D(16, kernel_size = (3,3), activation=p_activation) (img_2)
    #  img_2 = MaxPooling2D((2,2)) (img_2)
    #   img_2 = Dropout(drop_img2)(img_2)
    #img_2 = Conv2D(32, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    #img_2 = Conv2D(32, kernel_size = (3,3), activation=p_activation) (img_2)
    #img_2 = MaxPooling2D((2,2)) (img_2)
    #img_2 = Dropout(drop_img2)(img_2)
    img_2 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2, 2))(img_2)
    img_2 = Dropout(drop_img2)(img_2)
    img_2 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(img_2))
    img_2 = MaxPooling2D((2, 2))(img_2)
    img_2 = Dropout(drop_img2)(img_2)
    img_2 = GlobalMaxPooling2D()(img_2)

    img_concat = (Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))

    dense_ayer = Dropout(dr_end)(
        BatchNormalization(momentum=bn_model)(Dense(dense_coef * 4, activation=p_activation)(img_concat)))
    dense_ayer = Dropout(dr_end)(
        BatchNormalization(momentum=bn_model)(Dense(dense_coef, activation=p_activation)(dense_ayer)))
    output = Dense(1, activation="sigmoid")(dense_ayer)

    model = Model([input_1, input_2], output)
    optimizer = Adam(lr=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_lr)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model



for i in range(0, 5):
    np.random.seed(2017 + 2*i)
    model = get_model(i)
    model.summary()

    file_path = ".model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=50)

    model = get_model(i)

    batch_size = 64

    model.fit([X_train, X_angle_train], y_train, epochs=150
              , validation_data=([X_valid, X_angle_valid], y_valid)
              , batch_size=batch_size
              , shuffle=True
              , callbacks=callbacks)

    model.load_weights(filepath=file_path)

    print("Train evaluate:")
    print(model.evaluate([X_train, X_angle_train], y_train, verbose=1, batch_size=batch_size))
    print("####################")
    print("watch list evaluate:")
    print(model.evaluate([X_valid, X_angle_valid], y_valid, verbose=1, batch_size=batch_size))

    # prediction = model.predict([X_test, X_angle_test], verbose=1, batch_size=batch_size)
    sum_score += model.predict([X_test, X_angle_test], verbose=1, batch_size=batch_size)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': sum_score.reshape((sum_score.shape[0])) / 5})

submission.to_csv("./submission_real.csv", index=False)

