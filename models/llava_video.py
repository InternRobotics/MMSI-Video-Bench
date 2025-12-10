import torch
from PIL import Image
from copy import deepcopy
import copy
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
class LLava_Video:
    def __init__(self, model_path):
        self.name = model_path.split("/")[-1]
        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path,
            None,
            model_name,
            torch_dtype="bfloat16",
            device_map="auto"
        )
        self.model.eval()

    def infer_frames(self, sample):
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
        input_text = sample["input_prompt"]
        frame_tensors = []
        modalities = []
        for frames in sample.get("frames_list", []):
            frames1 = [Image.open(frame).convert('RGB') for frame in frames]
            frames1 = self.image_processor.preprocess(frames1, return_tensors="pt")["pixel_values"].to(
                torch.bfloat16).cuda()
            frame_tensors.append(frames1)
            modalities.append('video')

        for f in sample.get("ref_images", []):
            img = Image.open(f).convert("RGB")
            pixel = self.image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(
                torch.bfloat16).cuda()
            frame_tensors.append(pixel)
            modalities.append('image')

        conv_template = "qwen_1_5"  
        input_text = input_text.replace('<video>','<image>')
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        try:
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, 
                                              return_tensors="pt").unsqueeze(0).cuda()
            cont = self.model.generate(
                input_ids,
                images=frame_tensors,
                modalities= modalities,
                do_sample=False,
                temperature=0,
                max_new_tokens=88096,
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            return text_outputs, input_text
        except:
            return "error",input_text