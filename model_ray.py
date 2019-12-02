import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import json
import random
import cv2
import queue
import ray
ray.init()

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, InputLayer, Activation
from tensorflow.keras import Model
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import get_custom_objects


def swish(x, beta=1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Activation(swish)})

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# data_path = "/home/nick/data/poc3/2019-10-16T10:10:49.341Z"
# data_path = "/home/nick/data/poc3/2019-10-25T10:36:11.736Z"
# data_path = "/home/nick/data/poc3/2019-10-25T14:55:17.660Z"
# data_path = "/home/nick/data/poc3/2019-10-25T17:03:42.865Z"
# data_path = "/home/nick/data/poc3/2019-11-11T14:14:14.364Z"
# data_path = "/home/nick/data/poc3/2019-11-12T09:34:39.939Z"
data_path = "/home/nick/data/poc3/2019-11-12T17:10:36.477Z"

BATCH_SIZE= 1000
dropout = .40
NUM_EPOCHS = 5
training_size = 80000
val_size = 16000

IMG_WIDTH=256
IMG_HEIGHT=256

AUTOTUNE = tf.data.experimental.AUTOTUNE

cameras = ['left_oid', 'center_oid', 'right_oid']
camera_centre = ['center_oid']
steering_adj_val=0.275
steering_adj = {'left_oid': steering_adj_val, 'center_oid': 0., 'right_oid': -steering_adj_val}


def crop_camera(img, crop_height=66, crop_width=200):
    height = img.shape[0]
    width = img.shape[1]

    y_start = 120
    x_start = int(width/2)-int(crop_width/2)

    return img[y_start:y_start+crop_height, x_start:x_start+crop_width]


def jitter_image_rotation(image, steering):
    rows, cols, _ = image.shape
    transRange = 80
    numPixels = 20
    valPixels = 0.25
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))
    return image, steering


@ray.remote
def process_telemetry_row(logdata, data_path, cameras):
    
    # use one of the cameras randomily
    camera = cameras[random.randint(0, len(cameras)-1)]
    oid = getattr(logdata, camera)
    filename = data_path+'/'+oid+'.jpg'
    steering = getattr(logdata, 'steering') + steering_adj[camera]
    
    # loads image
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # randomily jitter the image and adjust steering
    img, steering = jitter_image_rotation(img, steering)

    img = crop_camera(img)
    
    # adjust brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    bv = .7 + np.random.random()
    hsv[::,2] = hsv[::,2]*bv

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # convert to floats in the [0,1] range.
    img = cv2.normalize(img, None, alpha=0, beta=1,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # flip 50% randomily that are not driving straight

    if random.random() >= .5 and abs(steering) > 0.1:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


def gen_data(data_path=data_path, log_file='vehicle_state.csv', skiprows=0, 
                   cameras=cameras):
    
    # # load the csv log file
    # print("camera images: ", cameras)
    # print("Data path: ", data_path)
    # print("Log file: ", log_file)

    column_names = ["timestamp", "speed", "throttle", "steering",
                    "engine_rpm", "gear", "left_oid", "center_oid", "right_oid"]
    data_df = pd.read_csv(data_path+'/'+log_file, names=column_names,
                          parse_dates=["timestamp"], date_parser=pd.to_datetime,
                          skiprows=skiprows)

    del data_df["timestamp"]

    data_count = len(data_df)

    # print("Log with %d rows." % data_count)
    num_threads = 32
    futures_queue = queue.Queue(maxsize=num_threads)

    while True:
        while not futures_queue.full():
            row = data_df.iloc[np.random.randint(data_count-1)]
            future = process_telemetry_row.remote(row, data_path, cameras)
            futures_queue.put_nowait(future)

        future = futures_queue.get()
        image,steering = ray.get(future)
        futures_queue.task_done()

        yield (image, steering)

# this function used for map to parallelise
def parse_fn(feature, label):
    return feature, label
    
def gen_train_data():
    ds = tf.data.Dataset.from_generator(gen_data,
                                        output_types=(tf.float32, tf.float64),
                                        output_shapes=([66, 200, 3], []))
    # ds = ds.map(parse_fn, num_parallel_calls=AUTOTUNE)
    # ds = ds.shuffle(training_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.repeat()
    ds = ds.batch(batch_size=BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def gen_val_data(): 
    def gen_data_camera_centre():
        return gen_data(cameras=camera_centre)
    ds = tf.data.Dataset.from_generator(gen_data_camera_centre,
                                        output_types=(tf.float32, tf.float64),
                                        output_shapes=([66, 200, 3], []))
    # ds = ds.map(parse_fn, num_parallel_calls=AUTOTUNE)
    # ds = ds.shuffle(val_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.repeat()
    ds = ds.batch(batch_size=BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def build_nvidia_model(img_height=66, img_width=200, img_channels=3,
                       dropout=.4):

    # activation = 'elu'
    # activation = 'selu'
    activation = 'swish'

    # build sequential model
    model = tf.keras.Sequential()

    # normalisation layer
    img_shape = (img_height, img_width, img_channels)


    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

    model.add(InputLayer(input_shape=img_shape, name="image"))

    for l in range(len(nb_filters)):
        conv2Dlayer = Conv2D(nb_filters[l],
                            kernel_size[l],
                            padding=padding[l],
                            strides=strides[l],
                            activation=activation)

        model.add(conv2Dlayer)
        model.add(Dropout(dropout))

    # flatten layer
    model.add(Flatten())

    # fully connected layers with dropout
    neurons = [100, 50, 10]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l], activation=activation))
        model.add(Dropout(dropout))

    # logit output - steering angle
    model.add(Dense(1, activation=activation, name='Out'))

    # optimizer = tf.keras.optimizers.Adam(lr=0.001)
    optimizer = tfa.optimizers.RectifiedAdam(lr=0.001)
    # model.compile(optimizer=optimizer, loss='mse')
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])
    # model.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['mse'])
    return model


def get_callbacks():
    checkpoint = ModelCheckpoint(
        "checkpoints/model-keras-input-{val_loss:.4f}.h5",
        monitor='val_loss', verbose=1, save_weights_only=True,
        save_best_only=True)

    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)

    # return [checkpoint, tensorboard]
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
    reduce_lr = ReduceLROnPlateau(monitor='val_mse', factor=0.9,
                                  patience=1, verbose=1, min_lr=0.0001)

    # earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,
    earlystopping = EarlyStopping(monitor='val_mse', min_delta=0,
                                  patience=3, verbose=1, mode='auto')
    # return [earlystopping, checkpoint]
    return [reduce_lr, checkpoint, earlystopping]

def main():
    model = build_nvidia_model(dropout=dropout)
    model.summary()

    metrics_names = model.metrics_names


    result = model.fit(gen_train_data(),
                        epochs=NUM_EPOCHS,
                        validation_split=0.0,
                        verbose=1,
                        callbacks=get_callbacks(),
                        steps_per_epoch=int(training_size/BATCH_SIZE),
                        validation_data=gen_val_data(),
                        validation_steps=int(val_size/BATCH_SIZE))


    # save weights and model
    model.save('model.h5')
    model.save_weights('model_weights.h5')
    with open('model.json', 'w') as modelfile:
        json.dump(model.to_json(), modelfile)


if __name__ == '__main__':
    main()

