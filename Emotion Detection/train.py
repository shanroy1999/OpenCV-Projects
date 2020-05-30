from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

classes = 5                 # 5 emotions => Angry, Sad, Happy, Neutral, Surprise
img_rows, img_cols = 48, 48     # Target images size = 48 x 48
batch_size = 64                 # Use 32 images at once for training

train_dir = r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Emotion Detection\train"
validation_dir = r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Emotion Detection\validation"

# Augment the data with random transformations
train_gen_data = ImageDataGenerator(
    rescale = 1/255,                    # Rescale the image RGB values between 0 and 1
    rotation_range = 30,                # Range within which pictures randomly rotate
    shear_range = 0.3,                  # Randomly apply Shear Transformations
    zoom_range = 0.3,                   # Randomly zooming inside pictures
    width_shift_range = 0.4,            # Randomly translate pictures vertically
    height_shift_range = 0.4,           # Randomly translate pictures horizontally
    horizontal_flip = True,             # Randomly flip half of images horizontally
    fill_mode = 'nearest'               # Fill in newly created pixels after rotation/shifting of image
)

validation_gen_data = ImageDataGenerator(rescale=1/255)

train_generator = train_gen_data.flow_from_directory(
    train_dir,
    color_mode = 'grayscale',
    target_size = (img_rows, img_cols),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)

valid_generator = validation_gen_data.flow_from_directory(
    validation_dir,
    color_mode = 'grayscale',
    target_size = (img_rows, img_cols),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

checkpoint = ModelCheckpoint(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Emotion Detection\model.h5",
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=9,
                          verbose=1,
                          restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop, checkpoint, reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

train_samples = 24176
valid_samples = 3006
epochs = 50

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples//batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=valid_generator,
    validation_steps=valid_samples//batch_size
)