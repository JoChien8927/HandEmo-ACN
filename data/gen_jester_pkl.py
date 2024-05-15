import os   
import cv2
import numpy as np
import sys
import pandas as pd
import shutil


if __name__ == '__main__':
    root_frame_path = r'./data/ryks/frames'

    file_list = os.listdir(root_frame_path)
    all_list =[]
    #counter for all the labels
    counter= [0,0,0,0]
    for file_name in file_list:
        label = int(file_name.split("_")[-1])
        if counter[label]<=45:
            counter[label]+=1
            all_list.append(file_name)
        else:
            continue
    print(counter)
    #end of sampling
    #split the list into train and val
    train_list = file_list[:int(len(all_list)*0.8)]
    val_list = file_list[int(len(all_list)*0.8):]
    
    pkl_output_folder = './data/ryks/jester_pkl_csv/'
    #
    if not os.path.exists(pkl_output_folder):
        os.makedirs(pkl_output_folder)

    train_dataset = {'frame':[],'label':[]}
    val_dataset = {'frame':[],'label':[]}

    for folder in os.listdir(root_frame_path):
        if folder in train_list:
        #make a list that store every frame path in os.path.join(root_frame_path,"rgb")
            rgb_frame_list = []
            depth_frame_list = []
            for frame in os.listdir(os.path.join(root_frame_path, folder, "rgb")):
                rgb_frame_list.append(os.path.join(root_frame_path, folder, "rgb", frame))
            label = int(folder.split("_")[-1])
            train_dataset['frame'].append(rgb_frame_list)
            train_dataset['label'].append(label)
            
        elif folder in val_list:
            rgb_frame_list = []
            depth_frame_list = []
            for frame in os.listdir(os.path.join(root_frame_path, folder, "rgb")):
                rgb_frame_list.append(os.path.join(root_frame_path, folder, "rgb", frame))
            label = int(folder.split("_")[-1])
            val_dataset['frame'].append(rgb_frame_list)
            val_dataset['label'].append(label)
            
        # create a train .pkl pkl and write train_dataset into it
        train_df = pd.DataFrame(train_dataset)
        train_df.to_pickle(pkl_output_folder+'train.pkl')
        #to csv
        train_df.to_csv(pkl_output_folder+'train.csv')
        # create a val .pkl pkl and write val_dataset into it
        val_df = pd.DataFrame(val_dataset)
        val_df.to_pickle(pkl_output_folder+'val.pkl')
        #to csv
        val_df.to_csv(pkl_output_folder+'val.csv')
        
    print('done')