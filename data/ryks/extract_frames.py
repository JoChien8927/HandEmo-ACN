import os
import cv2
from tqdm import tqdm

def extract_frames_from_videos(video_folder, output_folder):
    video_folders = [f.path for f in os.scandir(video_folder) if f.is_dir()]

    for folder in tqdm(video_folders, desc="Processing video folders", leave=False):
        for subfolder in os.scandir(folder):
            if subfolder.is_dir():
                subfolder_name = subfolder.name
                print(f"Processing clip {subfolder_name} of {folder}... ")
                # Get the video files in the subfolder
                rgb_video_file = os.path.join(folder, subfolder_name, "output_rgb.mp4")
                dep_video_file = os.path.join(folder, subfolder_name, "output_dep.mp4")
                # print(f"Opening {rgb_video_file}... ")
                # print(f"Opening {dep_video_file}... ")
                
                
                # Create output folders for frames
                rgb_frame_output_folder = os.path.join(output_folder, os.path.basename(folder), subfolder_name)
                dep_frame_output_folder = os.path.join(output_folder, os.path.basename(folder), subfolder_name)
                os.makedirs(rgb_frame_output_folder, exist_ok=True)
                os.makedirs(dep_frame_output_folder, exist_ok=True)

                # Open RGB video file
                cap_rgb = cv2.VideoCapture(rgb_video_file)

                # Get video properties
                width_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
                height_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_rgb = cap_rgb.get(cv2.CAP_PROP_FPS)

                # Read RGB frames and save as images
                frame_count_rgb = 0
                while True:
                    ret_rgb, frame_rgb = cap_rgb.read()
                    if not ret_rgb:
                        break

                    frame_count_rgb += 1
                    frame_filename_rgb = os.path.join(rgb_frame_output_folder, f"{frame_count_rgb:04d}.jpg")
                    cv2.imwrite(frame_filename_rgb, frame_rgb)

                # Release RGB video capture
                cap_rgb.release()

                # Open Depth video file
                cap_dep = cv2.VideoCapture(dep_video_file)

                # Get video properties
                width_dep = int(cap_dep.get(cv2.CAP_PROP_FRAME_WIDTH))
                height_dep = int(cap_dep.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_dep = cap_dep.get(cv2.CAP_PROP_FPS)

                # Read Depth frames and save as images
                frame_count_dep = 0
                while True:
                    ret_dep, frame_dep = cap_dep.read()
                    if not ret_dep:
                        break

                    frame_count_dep += 1
                    frame_filename_dep = os.path.join(dep_frame_output_folder, f"{frame_count_dep:04d}.jpg")
                    cv2.imwrite(frame_filename_dep, frame_dep)

                # Release Depth video capture
                cap_dep.release()
    print("All frames extracted successfully!")
if __name__ == "__main__":
    # 
    video_folder_path = './clips/'
    output_folder_path = './frames/'

    # 调用函数进行视频帧提取
    extract_frames_from_videos(video_folder_path, output_folder_path)
