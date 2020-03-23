import numpy as np
import cv2
import math
import os
from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
import tensorflow as tf
from memory_profiler import profile

class evaluation(ABC):
    def __init__(self):
        pass

    def is_same(self,A_img,B_img):
        if (A_img.shape != B_img.shape):
            return False
        return True

    def read_img(self,A_path,B_path):
        A_img = cv2.imread(A_path)
        B_img = cv2.imread(B_path)
        return A_img,B_img

    @abstractmethod
    def cal_method(self,A_path,B_path):
        pass

    def __call__(self, test_data_file_path,ground_truth_file_path):
        test_file_list = os.listdir(test_data_file_path)
        val = 0.0
        count = 0
        for file_name in test_file_list:
            ground_truth_file = ground_truth_file_path + file_name
            if os.path.exists(ground_truth_file):
                pp = self.cal_method(ground_truth_file, test_data_file_path + file_name)
                val += pp
                count += 1
                # print(file_name,' psnr: ',pp)
            else:
                raise RuntimeError('can not find ground truth file: ', ground_truth_file)
        val /= count
        return val

class psnr(evaluation):
    def __init__(self,MAXi=255):
        evaluation.__init__(self)
        self.MAXi = MAXi

    def cal_method(self,A_path,B_path):
        #print(A_path,B_path)
        A_img,B_img = self.read_img(A_path,B_path)
        if not self.is_same(A_img,B_img):
            raise RuntimeError('The size is not the same !')
        mse = self.mse(A_img,B_img)
        # print(A_path, ': ', mse)
        tmp = (self.MAXi * self.MAXi) / (mse)  # 防止分母为0
        psnr = 10 * math.log10(tmp)
        return psnr

    def mse(self,A_img,B_img):
        mse = np.mean(np.square(A_img - B_img))
        return mse

class ssim(evaluation):
    def __init__(self,MAXi=255,k1=0.01,k2=0.03):
        evaluation.__init__(self)
        self.MAXi = MAXi
        self.k1= k1
        self.k2 = k2

    def cal_method(self,A_path,B_path):
        # print(A_path,B_path)
        A_img, B_img = self.read_img(A_path, B_path)
        tf.reset_default_graph()
        if not self.is_same(A_img, B_img):
            raise RuntimeError('The size is not the same !')
        A_tensor = tf.convert_to_tensor(A_img,dtype=tf.float32)
        B_tensor = tf.convert_to_tensor(B_img,dtype=tf.float32)

        ssim = tf.image.ssim(A_tensor,B_tensor,self.MAXi)
        s = 0.0
        with tf.Session() as sess:
            s = sess.run(ssim)
            print(s)
        tf.get_default_graph().finalize()
        return s



class draw_graph:
    def __init__(self,width=0.3,font_size=7):
        self.width = width
        self.font_size = font_size

    def set_data(self,xlabel,ylabel,x,y,y_names,title):
        '''

        :param xlabel: 横轴标签
        :param ylabel: 纵轴标签
        :param x: 横轴数据
        :param y: 纵轴数据（多维向量，[[],[]]：有两组数据）
        :return:
        '''
        n = len(y)
        xx = range(len(x))
        mmin = 1e10
        mmax = -1
        for i in range(n):
            plt.bar([j+self.width*i for j in xx],y[i],width=self.width,label=y_names[i])
            mmin = min(mmin,min(y[i]))
            mmax = max(mmax,max(y[i]))
        loc_bias = (n-1) * self.width / 2
        plt.xticks([i + loc_bias for i in xx],x,fontsize=self.font_size)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        #plt.ylim(mmin-5, mmax+2)
        plt.ylim(0,1)
        plt.show()

def cal_psnr():
    ground_truth_file_paths = 'F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\datasets\\my_map\\testB\\'
    #epochs = [40,45,50,55,60,65,70,75]
    #test_out_paths = ['F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\test_out\\maps\\','F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\test_out\\cgan_mut\\','F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\test_out\\cgan_orgin\\']

    epochs = [40, 45, 50, 55, 60, 65, 70, 75,85,95,105,115,125,135]
    test_out_paths = ['F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\test_out\\css_gan\\my_map\\']


    ans = {}
    ans_list = []
    for test_out_path in test_out_paths:
        psnrs_list = []
        for epoch in epochs:
            test_fils_path = os.path.join(test_out_path,str(epoch)+'\\')
            p = psnr()
            val = p(test_fils_path,ground_truth_file_paths)
            psnrs_list.append(val)
        ans_list.append(psnrs_list)
        ans[test_out_path]=psnrs_list
    return ans_list,ans

def cal_ssim():
    #ground_truth_file_paths = 'F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\datasets\\maps\\testB\\'
    ground_truth_file_paths = 'F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\datasets\\my_map\\testB\\'
    epochs = [40, 45, 50, 55, 60, 65, 70, 75,85,95,105,115,125,135]
    #test_out_paths = ['F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\test_out\\css_gan\\maps\\','F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\test_out\\cgan_mut\\','F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\test_out\\cgan_orgin\\']
    test_out_paths = ['F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\test_out\\css_gan\\my_map\\']
    # epochs = [40, 45, 50, 55, 60, 65, 70, 75, 85, 95, 105, 115, 125, 135]
    # test_out_paths = ['F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\test_out\\css_gan\\maps\\']

    ans = {}
    ans_list = []
    for test_out_path in test_out_paths:
        psnrs_list = []
        for epoch in epochs:
            test_fils_path = os.path.join(test_out_path, str(epoch) + '\\')
            p = ssim()
            val = p(test_fils_path, ground_truth_file_paths)
            psnrs_list.append(val)
        ans_list.append(psnrs_list)
        ans[test_out_path] = psnrs_list
    return ans_list, ans


if __name__ == '__main__':
    #ground_truth_file_path = 'F:\\bishe\\pytorch-CycleGAN-and-pix2pix\\datasets\\my_map\\trainB\\'
    #p = psnr()
    #ans_list,ans = cal_psnr()
    #ans_list,ans = cal_ssim()
    ans_list = [[0.7834688782691955, 0.7956748068332672, 0.7599925029277802, 0.8099008083343506, 0.5362847328186036, 0.7669207262992859, 0.7921761524677277, 0.7903935301303864, 0.7977597641944886, 0.8014165949821472, 0.7866901898384094, 0.805358567237854, 0.7975343954563141, 0.7943317937850952]]
    pic = draw_graph(width=0.3)
    pic.set_data('epoch','ssim',[40, 45, 50, 55, 60, 65, 70, 75,85,95,105,115,125,135],ans_list,['css'],'')
    #print(ans)
    print(ans_list)