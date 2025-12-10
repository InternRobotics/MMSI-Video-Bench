import os
import json
import random
from tqdm import tqdm

def interval_sampling_list(a, b):
    step = (a - 1) / (b - 1) if b > 1 else 0
    indices = [int(i * step) for i in range(b)]
    return indices
def proportional_sample_from_lists(num_list, k):
    total_indices =interval_sampling_list(sum(num_list), k)
    
    sum_list = [sum(num_list[:i+1]) for i in range(len(num_list))]
    sum_list = [0] + sum_list
    indices_list = []
    for i in range(len(sum_list)-1):
        raw_indices = [k-sum_list[i] for k in total_indices if k>=sum_list[i] and k<sum_list[i+1]]
        indices_list.append(raw_indices)
    return indices_list


class MMSILOADER():
    def __init__(self,data_root='./data',max_frame=50,task_specific=True):
        self.data_root = data_root
        self.max_frame = max_frame
        self.task_specific = task_specific
        self.annos = json.load(open(os.path.join(data_root,'mmsivideo.json')))
        
    def process_sample(self,sample):
        '''Post-processing for samples, processing the model input based on maximum frame information.
        Args:
            sample (dict): Input sample containing:
                - 'video_list': List of video information dictionaries with 'start', 'end', and 'path' keys
                - 'frames_list': List of lists containing original frame paths
                - 'ref_images': List of reference image paths
                - 'ori_question': Original question string containing '<image>' placeholders
                - 'system_prompt': System instruction text
                - 'task_prompt': Task-specific prompt (if task_specific is True)
                - 'user_prompt': User query text
                - 'format_prompt': Output format specification
        Returns:
            dict: Processed sample with updated:
                - 'input_fps': Adjusted frames per second for video input
                - 'frames_list': Sampled frame paths with updated root directory
                - 'ref_images': Reference image paths with updated root directory
                - 'input_prompt': Fully assembled text prompt
        '''
        # (1) Video
        total_lantency = 0
        base_fps = sample['video_list'][0]['base_fps']
        for video_info in sample['video_list']: 
            start_time = video_info['start']
            end_time = video_info['end']
            video_info['path'] = os.path.join(self.data_root,'videos',video_info['path'])
            total_lantency += end_time - start_time
            
        input_fps = self.max_frame/total_lantency
        if input_fps > base_fps:
            sample['input_fps'] = base_fps
        else:
            sample['input_fps'] = input_fps
            
        # (2) Frames
        sample['max_frame'] = self.max_frame
        total_frames = sum([len(ori_frames) for ori_frames in sample['frames_list']])
        sampled_frames_list = []
        if total_frames > self.max_frame:
            indices_list = proportional_sample_from_lists([len(ori_frames) for ori_frames in sample['frames_list']],self.max_frame)
            for i in range(len(indices_list)):
                if len(indices_list[i])<1:
                    indices_list[i]=[0]
                sampled_frames_list.append([sample['frames_list'][i][j] for j in indices_list[i]])
        else:
            sampled_frames_list = sample['frames_list']
        sample['frames_list'] = [[os.path.join(self.data_root,'frames',frame) for frame in frames] for frames in sampled_frames_list]
      
        # (3) Reference images
        assert len(sample['ref_images'])==sample['ori_question'].count('<image>')
        sample['ref_images'] = [os.path.join(self.data_root,'ref_images',image) for image in sample['ref_images']]
        
        # (4) Text input
        if self.task_specific:
            sample['input_prompt'] = sample["system_prompt"]+'\n'+sample["task_prompt"]+sample['user_prompt']+sample['format_prompt']    
            sample['input_prompt_wo_sys'] = sample["task_prompt"]+sample['user_prompt']+sample['format_prompt'] 
        else:
            sample['input_prompt'] = sample["system_prompt"]+'\n'+sample['user_prompt']+sample['format_prompt']  
            sample['input_prompt_wo_sys'] = sample['user_prompt']+sample['format_prompt']   
        return sample         
    def __len__(self):
        return len(self.annos)
    def __getitem__(self,index):
        return self.process_sample(self.annos[index])
        
if __name__=='__main__':
    test_ex = MMSILOADER()
    a= test_ex[1000]
    print(a)
    