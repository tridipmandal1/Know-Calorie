import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print(tf.__version__)
