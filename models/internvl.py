import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer,AutoConfig,AutoModelForCausalLM
import math
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def split_model_2(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
def split_model_3(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_image2(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_video(frames_list,input_size=448, max_num=1):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for frame in frames_list:
        img = Image.open(frame).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class InternVL2_5():
    def __init__(self,model_path):
        
        self.name = model_path.split('/')[-1]
        device_map = split_model_2(self.name)
    
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.video_max_num = 1
        self.frame_max_num = 12
        self.generation_config = dict(max_new_tokens=88096, do_sample=False)
        
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
                - input_text (str): The processed input text with frame placeholders replaced.
                If an error occurs during inference, returns ('error', input_text).
        """
        
        pixel_values_list = []
        num_patches_list  = []
        video_input_size = 448
        for frames in sample['frames_list']:
            video_pixel_values, video_num_patches_list = load_video(frames,input_size=video_input_size,
                                                                    max_num=self.video_max_num)
            pixel_values_list.append(video_pixel_values)
            num_patches_list.extend(video_num_patches_list)
        for image in sample['ref_images']:
            image_pixel_value = load_image(image,input_size=video_input_size,max_num=self.frame_max_num)
            pixel_values_list.append(image_pixel_value)
            num_patches_list.append(image_pixel_value.size(0))
        pixel_values = torch.cat(pixel_values_list,dim=0).to(torch.bfloat16).cuda()
        
        input_text = sample['input_prompt']
      
        for frames in sample['frames_list']:
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(frames))])
            input_text = input_text.replace('<video>',video_prefix,1)
        try:
            response, _ = self.model.chat(self.tokenizer, pixel_values, input_text, self.generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
            return response,input_text
        except:
            return 'error',input_text

class InternVL2_5_video():
    def __init__(self,model_path):
        
        self.name = model_path.split('/')[-1]
        device_map = 'auto'
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.video_max_num = 1
        self.frame_max_num = 12
        self.generation_config = dict(max_new_tokens=88096, do_sample=False)
    def infer_frames(self,sample):
        """
        Inference sample containing frames, reference images and text to generate a response.
        
        The InternVL2.5-video model requires the total number of visual tokens to be divisible by 4
        for video input processing. When the token count is not a multiple of 4, blank/empty
        images are appended as padding to meet this requirement.
        
        Args:
            sample (dict): A dictionary containing multimodal input data with keys:
                - 'frames_list' (list): List of lists containing frame file paths for each video segment.
                - 'ref_images' (list): List of reference image file paths.
                - 'input_prompt' (str): Text prompt with <video> placeholders for frame insertion, <image> 
                                        placeholders for reference images insertion.
                
        Returns:
            tuple: A tuple containing:
                - response (str): The generated text response from the model.
                - input_text (str): The processed input text with frame placeholders replaced.
                If an error occurs during inference, returns ('error', input_text).
        """
        
        pixel_values_list = []
        num_patches_list  = []
        for frames in sample['frames_list']:
            video_pixel_values, video_num_patches_list = load_video(frames,input_size=448,
                                                                    max_num=self.video_max_num)
            pixel_values_list.append(video_pixel_values)
            num_patches_list.extend(video_num_patches_list)
        for image in sample['ref_images']:
            image_pixel_value = load_image(image,input_size=448,max_num=self.frame_max_num)
            pixel_values_list.append(image_pixel_value)
            num_patches_list.append(image_pixel_value.size(0))
        pixel_values = torch.cat(pixel_values_list,dim=0).to(torch.bfloat16).cuda()
        
        input_text = sample['system_prompt']+'\n'+sample['prompt']
        
        for frames in sample['frames_list']:
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(frames))])
            input_text = input_text.replace('<video>',video_prefix,1)
            
        num_shape = pixel_values.shape[0]
        to_add = 4- (num_shape % 4)
        
        if to_add!=4:
            ignore_txt = '(Ignore the following white images:'+'<image>'*to_add+')'
            for _ in range(to_add):
                white_img = Image.new('RGB', (448,448), (255, 255, 255))
                image_pixel_value = load_image2(white_img,input_size=448,max_num=self.video_max_num)
                pixel_values_list.append(image_pixel_value)
                num_patches_list.append(image_pixel_value.size(0))
            input_text+= ignore_txt   
            pixel_values = torch.cat(pixel_values_list,dim=0).to(torch.bfloat16).cuda()
        try:
            response, _ = self.model.chat(self.tokenizer, pixel_values, input_text, self.generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
            return response,input_text
        except:
            return 'error',input_text
               

class InternVL3():
    def __init__(self,model_path):
        self.name = model_path.split('/')[-1]
        device_map = split_model_3(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.video_max_num = 1
        self.frame_max_num = 12
        self.generation_config = dict(max_new_tokens=88096, do_sample=False)
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
                - input_text (str): The processed input text with frame placeholders replaced.
                If an error occurs during inference, returns ('error', input_text).
        """
        pixel_values_list = []
        num_patches_list  = []
        video_input_size = 448
        for frames in sample['frames_list']:
            video_pixel_values, video_num_patches_list = load_video(frames,input_size=video_input_size,
                                                                    max_num=self.video_max_num)
            pixel_values_list.append(video_pixel_values)
            num_patches_list.extend(video_num_patches_list)
        for image in sample['ref_images']:
            image_pixel_value = load_image(image,input_size=video_input_size,max_num=self.frame_max_num)
            pixel_values_list.append(image_pixel_value)
            num_patches_list.append(image_pixel_value.size(0))
        pixel_values = torch.cat(pixel_values_list,dim=0).to(torch.bfloat16).cuda()
        
        input_text = sample['system_prompt']+'\n'+sample['prompt']
        
        for frames in sample['frames_list']:
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(frames))])
            input_text = input_text.replace('<video>',video_prefix,1)
        try:
            response, _ = self.model.chat(self.tokenizer, pixel_values, input_text, self.generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
            return response,input_text
        except:
            return 'error',input_text
