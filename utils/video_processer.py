import cv2
import os

def extract_frames(video_path, fps=None, output_dir='./tmp'):
    
    
    os.makedirs(output_dir, exist_ok=True)
    frame_output_dir = output_dir
    os.makedirs(frame_output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    t1=0
    t2=duration
    start_frame = int(t1 * video_fps)
    end_frame = int(t2 * video_fps)
    frame_count = end_frame - start_frame  

    return _extract_by_fps(cap, t1, t2, fps, frame_output_dir, video_fps)

def _extract_by_fps(cap, t1, t2, target_fps, output_dir, video_fps):

    if target_fps >= video_fps:
        step = 1
    else:
        step = int(video_fps / target_fps)
    

    start_frame = int(t1 * video_fps)
    end_frame = int(t2 * video_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    all_images_path = []
    frame_interval = max(1, step)
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_path = os.path.join(output_dir, f"frame_{current_frame:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        all_images_path.append(frame_path)

        for _ in range(frame_interval - 1):
            cap.grab()
            current_frame += 1
            if current_frame > end_frame:
                break
        
        current_frame += 1
    
    return all_images_path


