import os

frame_path = r'./frames'
labels=[0,0,0,0]
min_frame_num=9999
for folder in os.listdir(frame_path):
    # print(folder)
    label=folder.split("_")[-1]
    if label=="0":
        labels[0]+=1
    elif label=="1":
        labels[1]+=1
    elif label=="2":
        labels[2]+=1
    elif label=="3":
        labels[3]+=1
    else:
        continue
    #check the total frame of each clip
    folder_size=len(os.listdir(os.path.join(frame_path, folder, "rgb")))
    if folder_size==0:
        print(f"empty folder : {folder}")
    if folder_size<min_frame_num :
        min_frame_num=folder_size
        # print(f"min_frame_num : {min_frame_num}")

for i in range(len(labels)):
    print(f"# of label {i} : {labels[i]}")
    
print(f"min_frame_num : {min_frame_num}")
    