import pandas as pd
import os
import cv2
from tqdm import tqdm

# read emotion timestamp and clip the video by timestamp
label_folder = './emotion_labels/toProcess/'
processed_folder = './emotion_labels/processed/'
video_folder = './raw_videos/'
label_files = [file for file in os.listdir(label_folder) if file.endswith(".json")]

for file in tqdm(label_files, desc="Processing files", leave=False):
    songName = file.split(".")[0]

    jfile = pd.read_json(label_folder + file)
    jdf = pd.DataFrame(jfile)

    # check if file exists
    print('-----------------------------------')
    print(f"Opening {file}... ")

    # get timestamp
    times = jdf['Time (s)']
    emotion= jdf['Label']
    
    
    # Find the video file named fileName
    rgb_vid = video_folder + songName + "_rgb" + ".mp4"
    dep_vid = video_folder + songName + "_dep" + ".mp4"
    output_folder = './clips/' + songName  # Change the output folder name
    
    # clip the video by timestamp
    # output each clip as a video file
    for i, timestamp in enumerate(times):
        #i=clip_num
        pair_label=emotion[i]
        valence = pair_label[0]
        arousal = pair_label[1]
        strength = pair_label[2]   
        if valence >= 5 and arousal >= 5:
            label=0
        elif valence < 5 and arousal >= 5:
            label=1
        elif valence < 5 and arousal < 5:
            label=2
        elif valence >= 5 and arousal < 5:
            label=3
        print(f'{songName}_clip_{i},(v,a,s) :{valence,arousal,strength} ,label :{label}')
        clip_folder = output_folder +"_clip_"+str(i) +"_emotion_" +str(valence)+"_"+str(arousal)+"_"+str(strength)+"_label_" + str(label)+"/"
        # create a new folder to store the clipped video
        # clip_folder = output_folder +"_clip_"+ str(i) + "/"
        print(f"Creating folder {clip_folder}... ")
        if not os.path.exists(clip_folder):
            os.makedirs(clip_folder)
        
        rgbclip_folder = clip_folder + "rgb/"
        depclip_folder = clip_folder + "dep/"
        
        os.makedirs(rgbclip_folder, exist_ok=True)
        os.makedirs(depclip_folder, exist_ok=True)
        
        # for time = i~i+1
        # Define the time range for the clip (i to i+1)
        start_time = timestamp
        end_time = times[i + 1] if i + 1 < len(times) else timestamp + 1  # Assuming the last timestamp is the end
        print(f"Clipping video from {start_time:.2f} to {end_time:.2f}... ")
        #clip size ==0
        if (start_time == end_time) | (start_time > end_time):
            print(f"== Clip size is 0! Skipping... ==")
            raise ValueError("Clip size is 0!")
            
        print('-----------------------------------')
        
        # Use OpenCV to extract video clips
        cap_rgb = cv2.VideoCapture(rgb_vid)
        cap_dep = cv2.VideoCapture(dep_vid)

        cap_rgb.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        cap_dep.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        # Get video properties
        width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_rgb.get(cv2.CAP_PROP_FPS)

        # Create VideoWriter objects for each channel
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        rgb_video_writer = cv2.VideoWriter(rgbclip_folder + 'output_rgb.mp4', fourcc, fps, (width, height))
        dep_video_writer = cv2.VideoWriter(depclip_folder + 'output_dep.mp4', fourcc, fps, (width, height))

        while cap_rgb.isOpened() and cap_dep.isOpened():
            ret_rgb, frame_rgb = cap_rgb.read()
            ret_dep, frame_dep = cap_dep.read()

            if ret_rgb and ret_dep:
                current_time = cap_rgb.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds

                if current_time >= end_time:
                    break

                # Write frames to video files
                rgb_video_writer.write(frame_rgb)
                dep_video_writer.write(frame_dep)

            else:
                break

        # Release video writers
        rgb_video_writer.release()
        dep_video_writer.release()
        cap_rgb.release()
        cap_dep.release()
        
    print(f"Finishing Clipping {file}... ")
    print(f"num of clips: {i}")
    print(f"output folder: {output_folder}")
    # move the processed file to the processed folder
    os.rename(label_folder + file, processed_folder + file)
    print(f"== Moved {file} to {processed_folder}... ")

print("\n\n== All Video Processed!")