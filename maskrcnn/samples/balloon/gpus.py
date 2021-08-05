import os
import tensorflow as tf

# fist method
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# second
tf.config.list_physical_devices(device_type=None)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

for root, dirs, files in os.walk("."):
    for filename in files:
        print(filename)