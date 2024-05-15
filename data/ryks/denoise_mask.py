import os
import cv2
from tqdm import tqdm

def crop_frame(frame):
    height, width, _ = frame.shape
    #將每張圖的下半部40%以下塗黑
    frame[int(height*0.4):, :] = 0
    #將每張圖的左半部10%塗黑
    frame[:, :int(width*0.1)] = 0
    #將每張圖的右半部20%塗黑
    frame[:, int(width*0.8):] = 0
    
    return frame

#找明度大於200的點 並塗黑
#地板區
def crop_frame_white(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    frame[thresh == 255] = 0
    return frame



if __name__ == "__main__":
    
    #create a folder to store the cropped frames
    input_folder_path = './frames_ori/'
    output_folder_path = './frames_masked/'
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for song_folder in os.listdir(input_folder_path):
        if not os.path.exists(os.path.join(output_folder_path, song_folder)):
            #make sure all output folder exists
            os.makedirs(os.path.join(output_folder_path, song_folder))
            os.makedirs(os.path.join(output_folder_path, song_folder, 'dep'))
            os.makedirs(os.path.join(output_folder_path, song_folder, 'rgb'))
            
        for vids_type in os.listdir(os.path.join(input_folder_path, song_folder)):
            #vids_type: dep or rgb
            #rgb
            depth_folder = os.path.join(input_folder_path, song_folder, 'dep')
            rgb_folder = os.path.join(input_folder_path, song_folder, 'rgb')
                    
            for frame_num in tqdm(os.listdir(depth_folder)): #因為兩個資料夾的frame數量是一樣的 所以只要一個loop就好
                #depth
                dep_frame = cv2.imread(os.path.join(input_folder_path, song_folder, 'dep', frame_num))
                dep_frame = crop_frame(dep_frame)
                dep_frame = crop_frame_white(dep_frame)
               
                #rgb
                rgb_frame = cv2.imread(os.path.join(input_folder_path, song_folder, 'rgb', frame_num))
                # 將depthframe中黑色的部分 也將rgbframe中的對應部分塗黑
                rgb_frame[dep_frame == 0] = 0
                # rgb_frame = crop_frame(rgb_frame)
                # rgb_frame = crop_frame_white(rgb_frame)
                
                
                # show
                # cv2.imshow(str(os.path.join(input_folder_path, song_folder, 'dep', frame_num)),  dep_frame)
                # cv2.imshow(str(os.path.join(input_folder_path, song_folder, 'rgb', frame_num)),  rgb_frame)
                # cv2.waitKey(0)
                
                #output
                cv2.imwrite(os.path.join(output_folder_path, song_folder, 'dep', frame_num), dep_frame)
                cv2.imwrite(os.path.join(output_folder_path, song_folder, 'rgb', frame_num), rgb_frame)    
            # break
            
        # break
                # cv2.imwrite(os.path.join(output_folder_path, song_folder, 'rgb', rgb_frame), frame)
