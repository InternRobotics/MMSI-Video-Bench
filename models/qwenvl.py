import os
import math
import hashlib
import requests
from utils.video_processer import extract_frames
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info

import re
import torch
def parse_special_text(text):
    pattern = r'(<video>|<image>)'
    parts = re.split(pattern, text)

    result = []
    video_count = -1
    image_count = -1
    for part in parts:
        if not part:  
            continue
        elif part == '<video>':
            video_count += 1
            result.append(f'<video>_{video_count}')
        elif part == '<image>':
            image_count += 1
            result.append(f'<image>_{image_count}')
        else:
            result.append(part) 
    return result


class QwenVL2_5():
    def __init__(self,model_path):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.name = model_path.split('/')[-1]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='auto',
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.gen_config = {
            'max_new_tokens': 88096,
        }
    def infer_frames(self,sample):
        """
        Inference sample containing frames, reference images and text to generate a response.
        
        Args:
            sample (dict): A dictionary containing multimodal input data with keys:
                - 'frames_list' (list): List of lists containing frame file paths for each video segment.
                - 'ref_images' (list): List of reference image file paths.
                - 'input_prompt' (str): Text prompt with <video> placeholders for frame insertion, <image> 
                                        placeholders for reference images insertion.
                
        Returns:
            tuple: A tuple containing:
                - response (str): The generated text response from the model.
                - content_list (list[str]): The processed content list.
        """

        content_list = []
        prompt = sample['input_prompt']
        assert prompt.count('<video>') == len(sample['frames_list']),print(prompt,len(sample['frames_list']))
        assert prompt.count('<image>') == len(sample['ref_images']),print(sample['id'])

        
        split_list = parse_special_text(prompt)
        for item in split_list:
            if '<video>' in item:
                index = int(item.split('_')[-1])
                content_list.append({
                    "type": "video",
                    "video":sample['frames_list'][index],
                    "max_pixels": 360 * 420,})
            elif '<image>' in item:
                index = int(item.split('_')[-1])
                content_list.append({
                    "type": "image",
                    "image":sample['ref_images'][index]
                })
            else:
                content_list.append({
                    "type": "text", "text": item
                }) 
    
        messages =[{'role': 'user', 'content': content_list}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        )

        inputs = inputs.to('cuda').to(torch.bfloat16)
        generated_ids = self.model.generate(**inputs, **self.gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]   
        return response,content_list


class QwenVL3():
    def __init__(self,model_path,tmp_dir = './tmp'):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration,Qwen3VLMoeForConditionalGeneration
        self.name = model_path.split('/')[-1]
        if '-A' not in self.name:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path,
                                                                         torch_dtype=torch.bfloat16,
                                                                         device_map="auto")
        else:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path, 
                                                                        torch_dtype=torch.bfloat16,
                                                                            device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.gen_config = {
            'max_new_tokens': 88096,
        }
        self.tmp_dir = tmp_dir
        
    def infer_frames(self,sample):

        """
        Inference sample containing videos, reference images and text to generate a response, caching 
        intermediate sampled frames to a temporary directory (tmp_dir).
        
        Args:
            sample (dict): A dictionary containing multimodal input data with keys:
                - 'video_list' (list): List of video segments.
                - 'ref_images' (list): List of reference image file paths.
                - 'input_prompt' (str): Text prompt with <video> placeholders for frame insertion, <image> 
                                        placeholders for reference images insertion.
                
        Returns:
            tuple: A tuple containing:
                - response (str): The generated text response from the model.
                - content_list (list[str]): The processed content list.
        """
        content_list = []
        prompt = sample['input_prompt']
        assert prompt.count('<video>') == len(sample['frames_list'])
        assert prompt.count('<image>') == len(sample['ref_images']),print(sample['id'])

        video_infos = sample["video_list"]
        split_list = parse_special_text(prompt)
        for item in split_list:
            if '<video>' in item:
                index = int(item.split('_')[-1])
                
                # sample and save
                sample_dir = f'{self.tmp_dir}/{sample["id"]}_{index}_{len(sample["frames_list"][index])}'
                if not os.path.exists(sample_dir):
                    extract_frames(video_infos[index]["path"],
                                fps=sample["input_fps"],output_dir=sample_dir)
                frame_paths = [os.path.join(sample_dir,f) for f in sorted(os.listdir(sample_dir))]
                content_list.append({
                    "type": "video",
                    "video":frame_paths,
                    "max_pixels": 360 * 420,
                    'sample_fps': sample["input_fps"]
                    })
                
            elif '<image>' in item:
                index = int(item.split('_')[-1])
                content_list.append({
                    "type": "image",
                    "image":sample['ref_images'][index]
                })
            else:
                content_list.append({
                    "type": "text", "text": item
                }) 

        messages =[{'role': 'user', 'content': content_list}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, 
                                                                    
                                                                    return_video_metadata=True)
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
            padding=True,
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, 
                                **video_kwargs, padding=True,return_tensors="pt")
        inputs = inputs.to('cuda').to(torch.bfloat16)
        try:
            generated_ids = self.model.generate(**inputs, **self.gen_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]   
            return response,content_list
        except Exception as e:
            return str(e),content_list
        
    