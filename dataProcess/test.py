import tensorflow as tf
import numpy as np

m = [[1, 2], [4, 5], [7, 8]]
m=np.asarray(m)
mm=tf.placeholder(tf.float32, shape=[10, 227,227,3] )
i=mm.get_shape()[0]
print(i)
print(type(i))