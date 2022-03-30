#=========================================================
#
#  
#
#=========================================================
import os
import numpy as np
from PIL import Image
from os.path import join

def get_path_list(root_path,img_path,label_path):
    tmp_list = [img_path,label_path]
    res = []
    for i in range(len(tmp_list)):
        data_path = join(data_root_path,tmp_list[i])
        filename_list = os.listdir(data_path)
        filename_list.sort()
        res.append([join(data_path,j) for j in filename_list])
    return res

def write_path_list(name_list, save_path, file_name):
    f = open(join(save_path, file_name), 'w')
    for i in range(len(name_list[0])):
        f.write(str(name_list[0][i]) + " " + str(name_list[1][i]) + '\n')
    f.close()

if __name__ == "__main__":
    #------------Path of the dataset -------------------------
    data_root_path = 'datasets'
    # if not os.path.exists(data_root_path): raise ValueError("data path is not exist, Please make sure your data path is correct")
    #train
    img_path = "spine/ultrasound/"
    gt_path = "spine/segmentation/"
    #----------------------------------------------------------
    save_path = "./data_path_list/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    img_list = get_path_list(data_root_path,img_path,gt_path)
    print('Number of imgs:',len(img_list[0]))
    write_path_list(img_list, save_path, 'dataset.txt')

    print("Finish!")
    
