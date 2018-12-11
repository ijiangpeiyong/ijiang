import numpy as np
import tensorflow as tf


#sess.run(tf.global_variables_initializer())

const_1=tf.constant([0])
init=tf.global_variables_initializer()

#random_normal_initializer=tf.random_normal_initializer(0., 0.3)

print('-'*20)

print(random_normal_initializer)

with tf.Session() as sess:
    # sess.Constant(0)


    sess.run(init)

    print(sess.run(const_1))

    #print(sess.run(random_normal_initializer))







print('-'*20)
print('End')
