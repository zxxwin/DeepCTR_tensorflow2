import tensorflow as tf
from tensorflow.keras.layers import *

class Dice(Layer):
    def __init__(self, axis=-1, epsilon = 1e-10, name=""):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        
    def build(self, input_shape):
        rand = tf.random_normal_initializer()(shape=[input_shape[-1]])
        self.alpha = tf.Variable(rand, dtype=tf.float32, name="alpha")
        
    
    def call(self, _x):
        # 输入数据的各个轴的维度
        input_shape = list(_x.get_shape())

        # 需要进行reduce计算的轴
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        
        # 能进行广播运算所需要的shape
        # shape: (1, _x.shape[axis])
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        
        # 除了axis轴外所有数算均值
        # shape: (_x.shape[axis], )
        mean = tf.reduce_mean(_x, axis=reduction_axes)
        # 然后还原为_x原来的维度，并且在axis轴进行广播
        # shape: (1, _x.shape[axis])
        brodcast_mean = tf.reshape(mean, broadcast_shape)

        # 除了axis轴外所有数算平方差
        # shape: (_x.shape[axis], )
        std = tf.reduce_mean(tf.square(_x - brodcast_mean) + self.epsilon, axis=reduction_axes)
        # 算标准差
        std = tf.sqrt(std)
        # 然后还原为_x原来的维度，并且在axis轴进行广播
        # shape: (1, _x.shape[axis])
        brodcast_std = tf.reshape(std, broadcast_shape)
        
        # 标准化，_x的shape不变
        x_normed = (_x - brodcast_mean) / (brodcast_std + self.epsilon)
        
        # #  以上操作可用下面的一句话代替：
        # x_normed = BatchNormalization(center=False, scale=False)(_x)
        
        # 标准化后使用 sigmoid 函数得到 x_p
        x_p = tf.sigmoid(x_normed)
        return self.alpha * (1.0 - x_p) * _x + x_p * _x