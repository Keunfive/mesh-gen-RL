import numpy as np
import tensorflow as tf
import os
import random

def my_seed_everywhere(seed):
    np.random.seed(seed) # np
    tf.random.set_seed(seed) # tensorflow
    tf.keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    random.seed(seed)

    