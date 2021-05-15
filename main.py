import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
train_path = 'dataset/THFOOD50-v1/train'
test_path = 'dataset/THFOOD50-v1/val'
#Data Preparation
def data_explor():
    count_data = {i:len(os.listdir(train_path+"/"+i)) for i in os.listdir(train_path)} 
    x = list(count_data.keys())
    y = list(count_data.values())
    plt.bar(x, y)
    plt.xticks(rotation=90)
    # plt.show()

    #after explor
    count_data = {i:len(os.listdir(train_path+"/"+i)) for i in os.listdir(train_path) if len(os.listdir(train_path+"/"+i)) <= 200}
    name = list(count_data.keys())
    count = list(count_data.values())
    plt.bar(name, count)
    plt.xticks(rotation=90)
    # plt.show()
    return name

def data_prep(selected_data):
    #delete folder
    for i in os.listdir(test_path):
        print(i)
        if (i not in selected_data) and (i != '.DS_Store'):
            # shutil.rmtree(train_path+'/'+i)
            print(i)
            shutil.rmtree(test_path+'/'+i)

if __name__ == '__main__':
    selected_data = data_explor()
    data_prep(selected_data)

