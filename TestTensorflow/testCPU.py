import tensorflow as tf
import numpy as np 

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess=tf.Session(config=tf.ConfigProto(
device_count={"CPU":12},
inter_op_parallelism_threads=1,
intra_op_parallelism_threads=1,
))



with tf.Session(config=tf.ConfigProto(
device_count={"CPU":12},
inter_op_parallelism_threads=1,
intra_op_parallelism_threads=1,
)) as sess:
    pass

print('End')


'''
 但是，如果你是用CPU版的TF，有时TensorFlow并不能把所有CPU核数使用到，这时有个小技巧David 9要告诉大家：

    with tf.Session(config=tf.ConfigProto(
    device_count={"CPU":12},
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1,
    gpu_options=gpu_options,
    )) as sess:

在Session定义时，ConfigProto中可以尝试指定下面三个参数：

    device_count, 告诉tf Session使用CPU数量上限，如果你的CPU数量较多，可以适当加大这个值
    inter_op_parallelism_threads和intra_op_parallelism_threads告诉session操作的线程并行程度，如果值越小，线程的复用就越少，越可能使用较多的CPU核数。如果值为0，TF会自动选择一个合适的值。

David 9亲自试验，训练似乎有1倍速度的提高。

另外，有的朋友的服务器上正好都是Intel的CPU，很可能需要把Intel的MKL包编译进TensorFlow中，以增加训练效率。这里David 9把MKL编译进TF的关键点也指出一下。
'''


