import tensorflow as tf
import zipfile
from tensorflow import keras
import numpy as np
import random
import os
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import skimage
import cv2
import warnings
warnings.filterwarnings('ignore')

# # Load the extension and start TensorBoard

# %load_ext tensorboard
# %tensorboard --logdir logs


TRAIN_PATH = "data/nuclear/stage1_train/"
TEST_PATH = "data/nuclear/stage1_test/"
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
# get list of all subfolders
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# define placeholders (also used to replace NaN in images to resize by 0)
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,
                   IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# # Resize images and masks
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    path = TRAIN_PATH + id_
    print(path + '/images/' + id_ + '.png')

    img = cv2.imread(path + '/images/' + id_ + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = np.zeros_like(img)
    img2[:,:,0] = gray
    img2[:,:,1] = gray
    img2[:,:,2] = gray
    img = img2[:, :, :IMG_CHANNELS]
    img = resize(
        img, (IMG_HEIGHT, IMG_WIDTH),
        mode='constant',
        preserve_range=True
    )

    X_train[n] = img  # fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(
            resize(
                mask_,
                (IMG_HEIGHT, IMG_WIDTH),
                mode='constant',
                preserve_range=True
            ),
            axis=-1
        )
        mask = np.maximum(mask, mask_)

    Y_train[n] = mask

# test images
X_test = np.zeros(
    (
        len(test_ids),
        IMG_HEIGHT,
        IMG_WIDTH,
        IMG_CHANNELS
    ),
    dtype=np.uint8
)

sizes_test = []
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    print(3333,path + '/images/' + id_ + '.png')
    img = cv2.imread(path + '/images/' + id_ + '.png')
    print("patches,shape->>", img.shape)
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(
        img,
        (IMG_HEIGHT, IMG_WIDTH),
        mode='constant',
        preserve_range=True
    )
    if(n == 0):
        print("dtype", img.dtype)

    X_test[n] = img

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)  # normalization

# Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(
    s)  # start with normal distributed weights
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(
    128, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(c6)

u7 = tf.keras.layers.Conv2DTranspose(
    64, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(c7)


u8 = tf.keras.layers.Conv2DTranspose(
    32, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(c8)

u9 = tf.keras.layers.Conv2DTranspose(
    16, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                            kernel_initializer="he_normal", padding="same")(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# Callbacks
checkpoiter = tf.keras.callbacks.ModelCheckpoint(
    'model.h5', verbose=1, save_best_only=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

results = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    batch_size=16,
    epochs=25,
    callbacks=callbacks)
# Predictions
idx = random.randint(0, len(X_train))
model.save("nuclear_model.h5")
print(X_test)
model = keras.models.load_model("nuclear_model.h5")
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
skimage.io.imshow(X_test[0])
skimage.io.show()
preds_test = model.predict(X_test, verbose=1)
print("preds_test,shape->>", X_test[0].shape)
cv2.imwrite("test.png", preds_test[0])
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Sanity check on random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(preds_test[0])
plt.show()

# Sanity check on random val samples
# ix = random.randint(0, len(preds_val_t))
# imshow(X_train[int(X_train.shape[0]*0.9):][ix])
# plt.show()
# imshow(np.squeeze(Y_train[int(X_train.shape[0]*0.9):][ix]))
# plt.show()
# imshow(np.squeeze(preds_val_t[ix]))
# plt.show()
