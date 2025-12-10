import os
import json
import argparse
try:
    import torch
    torch_flag = True
except:
    torch_flag = False
    
from dataset import MMSILOADER
from tqdm import tqdm  

MAX_LENGTH=300
HIDDEN_KEYS = []
def infer_frame_single_proc(model,data,setting,save_dir = './output'):
    """
    Processsamples sequentially using the specified model and save results.
    
    Iterates through the dataset, skipping already processed samples, and saves
    model responses as JSON files. Includes memory cleanup for GPU models.
    
    Args:
        model: Model instance with infer_frames() method,
        data: MMSI-Video dataloader,
        setting: Uniform or Sufficient,
        save_dir: Directory to save results (default: './output').
    
    Outputs JSON files at: {save_dir}/{setting}/{model.name}/{sample_id}.json
    """
    
    for index in tqdm(range(len(data))):
        sample = data[index]
        max_frame = data.max_frame
        if os.path.exists(f'{save_dir}/{setting}/{model.name}/{sample["id"]}.json'):
            continue
        if hasattr(model,'infer_frames'):
            os.makedirs(f'{save_dir}/{setting}/{model.name}',exist_ok=True)
            sample['response'], INFO = model.infer_frames(sample)
            for key in HIDDEN_KEYS:
                del sample[key]
            with open(f'{save_dir}/{setting}/{model.name}/{sample["id"]}.json','w') as f:
                json.dump(sample,f,indent=4,ensure_ascii=False)
            if torch_flag:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.empty_cache()

def load_model(args):
    
    # API models
    if args.model_name == 'doubao-1-5-vision-think':
        from models.api import api_model
        model = api_model('doubao-1-5-thinking-vision-pro-250428')
    elif args.model_name == 'doubao-seed-1-6':
        from models.api import api_model
        model = api_model('doubao-seed-1-6-vision-250815')
    elif args.model_name == 'GPT-4o':
        from models.api import api_model
        model = api_model('gpt-4o')
    elif args.model_name == 'GPT-5':
        from models.api import api_model
        model = api_model('gpt-5')
    elif args.model_name == 'O4':
        from models.api import api_model
        model = api_model('o4-mini')
    elif args.model_name == 'O3':
        from models.api import api_model
        model = api_model('o3')
    elif args.model_name == 'gemini-2.5':
        from models.api import api_model
        model = api_model('gemini-2.5-flash')
    elif args.model_name == 'gemini-2.5-no-think':
        from models.api import api_model
        model = api_model('gemini-2.5-flash(wo_thinking)')
    elif args.model_name == 'gemini-3':
        from models.api import api_model
        model = api_model('gemini-3-pro-preview-thinking')
    elif args.model_name == 'claude-4-5':
        from models.api import api_model
        model = api_model('claude-haiku-4-5-20251001-thinking')
    
    # InternVL2.5 series
    elif args.model_name == 'InternVL2_5-8B':
        from models.internvl import InternVL2_5
        model = InternVL2_5('/path/to/model')
    elif args.model_name == 'InternVL2_5-38B':
        from models.internvl import InternVL2_5
        model = InternVL2_5('/path/to/model')
    elif args.model_name == 'InternVL2_5-78B':
        from models.internvl import InternVL2_5
        model = InternVL2_5('/path/to/model')
    elif args.model_name == 'InternVL2_5-video-8B':
        from models.internvl import InternVL2_5_video
        model = InternVL2_5_video('/path/to/model')
    
    # InternVL3 series
    elif args.model_name == 'InternVL3-8B':
        from models.internvl import InternVL3
        model = InternVL3('/path/to/model')
    elif args.model_name == 'InternVL3-38B':
        from models.internvl import InternVL3
        model = InternVL3('/path/to/model')
    elif args.model_name == 'InternVL3-78B':
        from models.internvl import InternVL3
        model = InternVL3('/path/to/model')
    
    # QwenVL2.5 series
    elif args.model_name == 'QwenVL2_5-7B':
        from models.qwenvl import QwenVL2_5
        model = QwenVL2_5('/path/to/model')
    elif args.model_name == 'QwenVL2_5-32B':
        from models.qwenvl import QwenVL2_5
        model = QwenVL2_5('/path/to/model')
        from models.qwenvl import QwenVL2_5
    elif args.model_name == 'QwenVL2_5-72B':
        model = QwenVL2_5('/path/to/model')
    
    # QwenVL3 series
    elif args.model_name == 'QwenVL3-8B':
        from models.qwenvl import QwenVL3
        model = QwenVL3('/path/to/model',
                        tmp_dir = './tmp')
    elif args.model_name == 'QwenVL3-30B':
        from models.qwenvl import QwenVL3
        model = QwenVL3('/path/to/model',
                        tmp_dir = './tmp')
    elif args.model_name == 'QwenVL3-30B-Thinking':
        from models.qwenvl import QwenVL3
        model = QwenVL3('/path/to/model',
                        tmp_dir = './tmp')

    # LLaVA-Video series
    elif args.model_name == 'LLaVA-Video-7B':
        from models.llava_video import LLava_Video
        model = LLava_Video('/path/to/model')    
    elif args.model_name == 'LLaVA-Video-72B':
        from models.llava_video import LLava_Video
        model = LLava_Video('/path/to/model')
    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='MMSI-Video-Bench Inference')

    parser.add_argument('--model_name', type=str, default='GPT-4o')
    parser.add_argument('--setting', type=str, default='Uniform-50')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./output')
    args = parser.parse_args()
        
    if 'Uniform' in args.setting:
        max_frame = int(args.setting.split('-')[1])
    elif args.setting=='Sufficient-Coverage':
        max_frame = MAX_LENGTH
        
    test_loader = MMSILOADER(data_root=args.data_root,max_frame=max_frame)
    os.makedirs(args.save_dir, exist_ok=True)
    
    model = load_model(args)
    infer_frame_single_proc(model,test_loader,args.setting,args.save_dir)