import tensorflow as tf
import numpy as np

x = tf.Variable(np.array([1.0, 2.0, 3.0]))
with tf.GradientTape() as tape:
    y = x ** 3 + 2 * x + 5

print(tape.gradient(y, x))

def training(self, x, y):
    m = len(x)
    with tf.GradientTape() as tape:
        z = self.forpass(x)
        # 정방향 계산해서 손실값 계산
        loss = tf.nn.softmax_cross_entropy_with_logits(y, z)
        # 각 배치에 관한 손실 반환

        loss = tf.reduce_mean(loss)

