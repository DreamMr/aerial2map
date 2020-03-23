import numpy as np
import cv2
import tensorflow as tf

true_img = cv2.imread('./2001_t.jpg',-1)
fake_img = cv2.imread('./2001_f.jpg',-1)

true_tensor = tf.convert_to_tensor(true_img,dtype=tf.float32)
fake_tensor = tf.convert_to_tensor(fake_img,dtype=tf.float32)

ssim = tf.image.ssim(true_tensor,fake_tensor,255)
with tf.Session() as sess:
    t=sess.run(ssim)
    print('tensorflow ssim\'s : ',t)

def ssim(img_pred,img_truth,k1=0.01,k2=0.03,maxi=255):
    c1 = (k1*maxi)**2
    c2 = (k2*maxi)**2

    mu_pred = np.mean(img_pred)
    mu_truth = np.mean(img_truth)

    sigma_pred = np.var(img_pred)
    sigma_truth = np.var(img_truth)

    std_pred= np.sqrt(sigma_pred)
    std_truth = np.sqrt(sigma_truth)

    ssim = ((2 * mu_pred * mu_truth + c1) * (2 * std_pred * std_truth + c2)) / ((mu_truth ** 2 + mu_pred ** 2 + c1) * (sigma_truth ** 2 + sigma_pred ** 2 + c2))
    return ssim

print('my implement ssim: ',ssim(fake_img,true_img))

