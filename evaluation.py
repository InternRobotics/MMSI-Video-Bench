import os
import json
import argparse

def clear_words(text):
    """Remove spaces, quotes, newlines, and colons from text."""
    return  text.replace(' ','').replace('\"','').replace("\'",'').replace('\n','').replace(':','')

def extract_answer(response):
    """
        Extract answer from model response.
        
        Supports multiple formats: boxed{}, JSON, 'answer is', 'Answer:', and single-letter answers.
        Returns 'O' if no answer found.
    """
    response = response.replace('<answer>','').replace('</answer>','')
    if response is None or 'no answer' in response:
        return 'O'
    if 'boxed{' in response:
        split_text = response.split('boxed{')[1].split('}')[0]
        split_text = clear_words(split_text)
        return split_text
        
    if '\"answer\":' in response:
        split_text = response.split('\"answer\":')[-1]
        split_text = split_text.split(',')[0].split('.')[0]
        split_text = clear_words(split_text)
        return split_text
    elif 'answer is' in response:
        split_text = response.split('answer is')[-1]
        split_text = split_text.split(',')[0].split('.')[0]
        split_text = clear_words(split_text)
        return split_text
    elif 'Answer: ' in response:
        split_text = response.split('Answer: ')[-1]
        split_text = split_text.split(',')[0].split('.')[0]
        split_text = clear_words(split_text)
        return split_text
    elif clear_words(response.split('.')[0]) in ['A','B','C','D','E','F']:
        return clear_words(response.split('.')[0])
    else:
        return 'O'

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='MMSI-Video-Bench Evaluation')

    parser.add_argument('--eval_dir', type=str, default='./output/Uniform-50/Qwen3-VL-8B-Instruct')
    parser.add_argument('--bench', 
                   choices=['main', 'robot_bench', 'ground_bench','indoor_perception_bench','easy2hard_bench'], 
                   default='main')

    args = parser.parse_args()
    result_dir = args.eval_dir
    if args.bench != 'main':
        bench_type_id = json.load(open(f'./meta_data/{args.bench}.json'))
        bench_id_type = {}
        for sub_type in bench_type_id:
            for id_ in bench_type_id[sub_type]:
                bench_id_type[id_] = sub_type
        ORDER_LIST = list(bench_type_id.keys())
    else:
        ORDER_LIST = ['(Cross-Video) Memoery Update', '(Cross-Video) Multi-View Integration', 'Planning',
                      'Prediction', '(Motion Understanding) Camera Motion', '(Motion Understanding) Instance Motion', 
                      '(Motion Understanding) Interactive Motion', '(Spatial Construction) Instance-Instance Spatial Relationship',
                      '(Spatial Construction) Instance-Scene Spatial Relationship', '(Spatial Construction) Scene-Scene Spatial Relationship', 
                      '(Spatial Construction) Instance/Scene Attribute', '(Spatial Construction) Camera-Instance Spatial Relationship',
                      '(Spatial Construction) Camera-Scene Spatial Relationship']
    score_dict = {'Overall':[]}
    error_list = []
    skip_it = 0
    for json_file in os.listdir(result_dir):
        json_data = json.load(open(os.path.join(result_dir,json_file)))
        
        q_id = json_data['id']
        response = json_data['response']  
        gt =  json_data["ground_truth"]
        
        if args.bench == 'main':
            question_type =  json_data["type"]
        else:
            if q_id not in bench_id_type:
                continue
            else:
                question_type = bench_id_type[q_id] 

        try:
            pred = extract_answer(response)
            assert pred in ['A','B','C','D','E','F']
            if question_type not in score_dict:
                score_dict[question_type] = []
            score_dict[question_type].append(float(pred==gt))
            score_dict['Overall'].append(float(pred==gt))
        except:
            if question_type not in score_dict:
                score_dict[question_type] = []
            score_dict[question_type].append(0.0)
            score_dict['Overall'].append(0.0)
            skip_it+=1
            print('Fail to extract the answer from the response in',os.path.join(result_dir,json_file))
            error_list.append(os.path.join(result_dir,json_file))
    
    for key in ['Overall']+ORDER_LIST:
        print(key,': ',sum(score_dict[key])/len(score_dict[key]),len(score_dict[key]))
    print(f'failure count/ total count: {len(error_list)} / {len(os.listdir(result_dir))}')