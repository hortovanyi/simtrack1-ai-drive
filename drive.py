import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from io import BytesIO

import tensorflow as tf

from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.backend import sigmoid


def swish(x, beta=1):
    return (x * sigmoid(beta * x))


get_custom_objects().update({'swish': Activation(swish)})

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sio = socketio.Server(async_mode='eventlet')
model = None


def crop_camera(img, crop_height=66, crop_width=200):
    height = img.shape[0]
    width = img.shape[1]

    y_start = 120
    x_start = int(width/2)-int(crop_width/2)

    return img[y_start:y_start+crop_height, x_start:x_start+crop_width]


@sio.event
def connect(sid, environ):
    print("connect", sid)
    send_control(0.0,0.0)


def send_control(steering, throttle):
    sio.emit("drive", data =
        {'steer': steering.__str__(),
         'throttle': throttle.__str__()
         }, skip_sid=True)


@sio.on('telemetry')
def telemetry(sid, data):
    steering = data["steering"]
    throttle = data["throttle"]
    speed = data["speed"]
    engine_rpm = data["engineRPM"]
    gear = data["gear"]
    imageData = data["imageData"]

    image = Image.open(BytesIO(base64.b64decode(imageData))).convert('RGB')
    image.save("last_telemetry_camera.png")
    image_array = np.asarray(image)
    image_array = crop_camera(image_array)
    # im_cropped = Image.fromarray(image_array, mode='RGB')
    # im_cropped.save("last_telemetry_camera_cropped.png")

    image_array = tf.image.convert_image_dtype(image_array, tf.float32)
    transformed_image_array = image_array[None, :, :, :]

    # im = Image.fromarray(tf.image.convert_image_dtype(image_array, tf.uint8).numpy(), mode='RGB')
    # im.save("last_telemetry_camera_image_transformed.jpg")

    steering = float(model.predict(transformed_image_array, batch_size=1))

    throttle = 0.65
    print(steering, throttle)
    send_control(steering, throttle)

if __name__ == '__main__':

    with open("model.json", 'r') as jfile:
        model = tf.keras.models.model_from_json(
            json.loads(jfile.read()), custom_objects={'Activation': Activation(swish)})

    model.compile("adam", "mse")
    model.load_weights("model_weights.h5")

    app = socketio.WSGIApp(sio)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
