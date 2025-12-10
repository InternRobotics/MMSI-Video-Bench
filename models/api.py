import os
import numpy as np
import time
from tqdm import tqdm
from utils.video_processer import extract_frames
from utils.openai_api import mimic_chat_budget, get_content_groups_from_source_groups,num_tokens_from_string, get_full_response

def split_txt_based_on_images(text,image_paths):
    ret = []
    assert text.count('<image>') == len(image_paths)
    split_text_list = text.split('<image>')
    for index_ in range(len(image_paths)):
        ret.append(split_text_list[index_])
        ret.append(image_paths[index_])
    ret.append(split_text_list[-1])
    ret =[item for item in ret if len(item)>0]
    return ret

def generate_multi_qa_gpt(input_dict,model_name='gpt-4o',postprocess = False):
    system_prompt = input_dict['system_prompt']
    source_groups = input_dict['user_message']
    content_groups = get_content_groups_from_source_groups(source_groups,
                                                           downsample=postprocess) 
 
    
    conversation= mimic_chat_budget(content_groups, system_prompt=system_prompt,
                                    model_name=model_name)    
    raw_annotation = []
    for message in conversation:
        if message["role"] == "assistant":
            raw_annotation.append(message["content"])
    
    return raw_annotation

class api_model():
    def __init__(self,name):
        self.name = name
    def infer_frames(self,sample):
        '''Processes a multimodal sample and generates a response using an external API service.
        
        This method performs the following transformations:
        1. Converts video placeholders (<video>) into multiple image placeholders (<image>)
           based on the number of frames in each video segment.
        2. Extends the image list with both video frames and reference images.
        3. Constructs a structured API input with system prompt and user message.
        4. Due to API limitations on total uploaded image size, the system determines post-processing requirements 
            based on frame count and model type.
        5. Calls the API generation function and returns the response.
        
        Args:
            sample (dict): A dictionary containing multimodal input data with keys:
                - 'input_prompt_wo_sys' (str): Text prompt containing <video> placeholders.
                - 'frames_list' (list): List of lists, each containing frame paths for a video.
                - 'ref_images' (list): List of reference image paths.
                - 'system_prompt' (str): System-level instructions for the API.
                - 'max_frame' (int): Maximum number of frames in the sample
        
        Returns:
            tuple: A tuple containing:
                - results (str): The generated text response from the API.
                - input_ (dict): The structured input sent to the API for debugging/reference.
        
        '''
       
        split_text = sample['input_prompt_wo_sys']
        split_images = []
        assert split_text.count('<video>') == len(sample['frames_list'])
        for frames in sample['frames_list']:
            split_text = split_text.replace('<video>','<image>'*len(frames),1)
            split_images.extend(frames)
        split_images.extend(sample['ref_images'])

        input_ = {'system_prompt':sample['system_prompt'],
                'user_message':[split_txt_based_on_images(split_text,split_images)]}
        
        postprocess =  sample['max_frame']>50 or 'claude' in self.name
            
        results = generate_multi_qa_gpt(input_,self.name,postprocess)[0]
        
        return results,input_
        
            
     
            
        