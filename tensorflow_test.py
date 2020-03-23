import tensorflow as tf
from memory_profiler import profile
import cv2
import time

A_img = cv2.imread('./2001_f.jpg',-1)
B_img = cv2.imread('./2001_t.jpg',-1)

@profile
def cal_ssim_slow(MAXi):
    A_tensor = tf.convert_to_tensor(A_img,dtype=tf.float32)
    B_tensor = tf.convert_to_tensor(B_img,dtype=tf.float32)
    ssim = tf.image.ssim(A_tensor,B_tensor,255)
    s = 0.0
    with tf.Session() as sess:
        s = sess.run(ssim)
    return s

@profile
def cal_ssim_quick(MAXi):
    tf.reset_default_graph()
    A_tensor = tf.convert_to_tensor(A_img, dtype=tf.float32)
    B_tensor = tf.convert_to_tensor(B_img, dtype=tf.float32)
    ssim = tf.image.ssim(A_tensor, B_tensor, 255)
    s = 0.0
    with tf.Session() as sess:
        s= sess.run(ssim)
    tf.get_default_graph().finalize()
    return s


def memory_tensorflow():
    epoch = 50
    # t1 = time.time()
    # for i in range(epoch):
    #     ssim = cal_ssim_slow(255)
    #     print(i)
    # t2 = time.time()
    #
    # tf.reset_default_graph()

    t3 = time.time()
    for i in range(epoch):
        ssim = cal_ssim_quick(255)
        print(i)
    t4 = time.time()

    #print('no reset graph cost time: ',t2-t1)
    print('reset graph cost time: ', t4 - t3)


if __name__ == '__main__':
    memory_tensorflow()


