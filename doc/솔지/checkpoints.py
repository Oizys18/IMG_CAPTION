import os
import tensorflow as tf
"""
# http://solarisailab.com/archives/2524
SAVER_DIR = "model"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

# 만약 저장된 모델과 파라미터가 있으면 이를 불러오고 (Restore)
# Restored 모델을 이용해서 테스트 데이터에 대한 정확도를 출력하고 프로그램을 종료합니다.
if ckpt and ckpt.model_checkpoint_path:
  saver.restore(sess, ckpt.model_checkpoint_path)
  print("테스트 데이터 정확도 (Restored) : %f" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
  sess.close()
  exit()
"""
# http://jaynewho.com/post/8
# https://hiseon.me/data-analytics/tensorflow/tensorflow-checkpoint/