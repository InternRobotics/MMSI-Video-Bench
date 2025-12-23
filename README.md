<div align="center">

# MMSI-Video-Bench: A Holistic Benchmark for Video-Based Spatial Intelligence

</div>

<p align="center">
  <!-- <b>Authors</b><br> -->
  <a href="https://github.com/rbler1234" target="_blank">Jingli Lin<sup>1,2*</sup></a>, </span>
  <a href="https://runsenxu.com/" target="_blank">Runsen Xu<sup>1,3*</sup><sup>†</sup></a>, </span>
  <a href="https://openreview.net/profile?id=%7EShaohao_Zhu2" target="_blank">Shaohao Zhu<sup>1,4</sup></a>, 
  <a href="https://sihany077.github.io/" target="_blank">Sihan Yang<sup>1</sup></a>, 
  <a href="https://openreview.net/profile?id=~Peizhou_Cao1" target="_blank">Peizhou Cao<sup>1,5</sup></a>, 
  <a href="https://kingteeloki-ran.github.io/" target="_blank">Yunlong Ran<sup>1,4</sup></a>,
  <a href="https://openreview.net/profile?id=~Miao_Hu4" target="_blank">Miao Hu<sup>6</sup></a>,
  <a href="https://github.com/ZCMax" target="_blank">Chenming Zhu<sup>1,7</sup></a>,
    <a href="https://kuaikuaixym.github.io/" target="_blank">Yiman Xie<sup>1,4</sup></a>,
    <a href="https://openreview.net/profile?id=%7EYilin_Long2" target="_blank">Yilin Long<sup>1,8</sup></a>,
    <a href="https://gordonhu608.github.io/" target="_blank">Wenbo Hu<sup>1,9</sup></a>,
    <br>
    <a href="http://dahua.site/" target="_blank">Dahua Lin<sup>1,3</sup></a>,
    <a href="https://tai-wang.github.io/" target="_blank">Tai Wang<sup>1✉</sup></a>,
    <a href="https://oceanpang.github.io/" target="_blank">Jiangmiao Pang<sup>1✉</sup></a>
</p>

<p align="center">
 <span>1. Shanghai AI Laboratory &nbsp;&nbsp; 2. Shanghai Jiaotong University &nbsp;&nbsp; 3. The Chinese University of Hong Kong &nbsp;&nbsp; 4. Zhejiang University &nbsp;&nbsp; 5. Beihang University</span>
        <br>
        <span>6. Xi'an Jiaotong University &nbsp;&nbsp; 7. University of Hong Kong &nbsp;&nbsp; 8. Fudan University &nbsp;&nbsp; 9. University of California, Los Angeles</span>
  
</p>

<p align="center">
  <sup>*</sup>Equal Contribution &nbsp;&nbsp;
  <sup>†</sup>Project Lead &nbsp;&nbsp;
</p>

<div align="center">

[![Homepage](https://img.shields.io/badge/Project_Page-4285F4?logo=googlechrome&logoColor=white)](https://rbler1234.github.io/MMSI-VIdeo-Bench.github.io/)
[![Huggingface](https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/rbler/MMSI-Video-Bench)
[![Arxiv](https://img.shields.io/badge/arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.10863)
[![GitHub star chart](https://img.shields.io/github/stars/InternRobotics/MMSI-Video-Bench?style=square)](#)
[![GitHub Issues](https://img.shields.io/github/issues/InternRobotics/MMSI-Video-Bench)](#)


</div>


## 🎥 Demo
[![demo](assets/first_frame.png "demo")](https://youtu.be/HTp_k_qq_Wc)

## 🔔 News

🔥[2025-12]: Our MMSI-Video-Bench has been integrated into [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

🔥[2025-12]: We released our paper, benchmark, and evaluation codes.


## Features of MMSI-Video-Bench

MMSI-Video-Bench is a fully human-annotated benchmark for evaluating video-based spatial intelligence in MLLMs. It implements a four-level framework—Perception, Planning, Prediction, and Cross-Video Reasoning, through 1,106 questions grounded in 1,278 video clips sourced from 25 public datasets as well as in-house recordings.

- **High quality.** All data are manually annotated by 11 domain experts in 3D vision, following a rigorous review and acceptance process to ensure annotation accuracy and reliability.

- **Challenging.** We evaluate 25 strong open-source and proprietary MLLMs, revealing a striking human–AI gap, even the best reasoning model trails human performance by nearly 60%.

- **Diverse Video Sources.** MMSI-Video-Bench includes videos from 25 public datasets and 1 in-house collection, spanning tabletop recordings, indoor and multi-floor environments, outdoor scenes, sports, and movie footage.

- **Comprehensive and Hostile Task Coverage.** The benchmark covers spatial layout reasoning, motion understanding, decision-making, and cross-video reasoning, providing a more holistic evaluation of video-based spatial intelligence.


<div style="text-align: center;">
    <img src="assets/teasor.png" alt="Dialogue_Teaser" width=100% >
</div>





## Example

The questions in MMSI-VIdeo-Bench span 5 major categories: **(1).Spatial Construction**：This category focuses on spatial attributes of instances and scenes, as well as spatial relationships among instances, scenes, and cameras (six subtypes in total); **(2).Motion Understanding**:
This includes understanding camera motion, instance motion, and interactive motion between instances; **(3).Planning** based on spatiotemporal video information;  **(4).Prediction**: Assessing a model’s ability to predict, anticipate, or imagine future states based on the observed video; **(5).Cross-Video Reasoning**: This involves memory update across temporally separated video segments and multi-view integration across videos captured from different viewpoints.


<div style="text-align: center;">
    <img src="assets/benchmark_samples.png" alt="Dialogue_Teaser" width=100% >
</div>

## 📊 Data Details

All of our data is available on [Hugging Face](https://huggingface.co/datasets/rbler/MMSI-Video-Bench) and includes the following components:

🎥 **Video Data** (`videos.zip`): Contains the video clip file (.mp4) corresponding to each sample. This file is generally not required for most models.

🎥 **Frame Data** (`frames.zip`): Contains the frames (.jpg) extracted from each sample's video at the **base sampling rate**. This rate ensures no key information loss during sampling.  Each frame file is named using the format `{timestamp}_frame_{base_interval}_{image_id}` (e.g., 00:06.00_frame_1.50_4), where the timestamp, also shown on the **top-left corner** of the frame, indicates its **capture time in the original recording**.

🖼️ **Reference Image Data** (`ref_images.zip`): Contains the auxiliary images referenced in the questions for each sample.

📝 **Text Annotation** (`mmsivideo.json`)：This file contains the annotation information for MMSI-Video-Bench. All time references in the questions correspond to the capture time in the original recording and **align with** the timestamp flag on each frame. Key fields include:

```
{
  "ref_images": [Paths to auxiliary images referenced in the question,...],
  "video_list": [
    {
      "path": Video clip file path,
      "start": Timestamp (in seconds) of the first frame of the video clip in the original recording,
      "end": Timestamp (in seconds) of the last frame of the video clip in the original recording,
      "base_fps": Base sampling rate
    },
    ...
  ],
  "frames_list": [[Paths to frames sampled at the base sampling rate,...],...],
  "system_prompt": "...",
  "task_prompt": Task-specific prompt,
  "user_prompt": Question text, with <video> as a placeholder for video and <image> for auxiliary images,
  "format_prompt": Output format requirements,
  "ground_truth": Correct answer
}
```


Unless otherwise specified, the model input generally consists of:
`system_prompt + task_prompt + user_prompt + format_prompt`.


## 🚀 Quick Start

As outlined in our paper, we support two evaluation settings: **Sufficient-Coverage** ensures lossless sampling by preserving all essential information, while **Uniform‑50** uniformly samples 50 frames and cannot guarantee full information retention. Although most models—due to API constraints or GPU memory limits—only support the Uniform‑50 setting, we still strongly recommend evaluating under the **Sufficient‑Coverage** setting whenever possible.

### Native Codebase Evaluation

1. Data Preparation: After downloading the data, unzip the files and organize the directory as follows:
 
    ```
    data/
    ├── mmsivideo.json
    ├── frames/
    ├── ref_images/
    ├── videos/
    ```
2. Install Required Packages. If you plan to evaluate open-source models, please set up their corresponding environments.
    ```
    pip install -r requirements.txt
    ```
3. Run Inference & Evaluation: For proprietary models: Update the `base_url` and `api_key` in `utils/openai_api.py` with your own credentials. For open-source models: Modify the `load_model` function in `inference.py` to point to the corresponding model path. Run the following commands to perform inference for a specific model under a given setting, and evaluate the results on a specific benchmark (Main, Robot Bench, Indoor Scene Perception Bench or Grounding Bench):
    ```python
    python inference.py --model_name {model_name} --setting Uniform-50/Sufficient-Coverage # inference
    python evaluation.py --eval_dir {path/to/results} --bench main/robot_bench/ground_bench/indoor_perception_bench/easy2hard_bench # evaluation
    ```

### VLMevalKit Evaluation
Our MMSI-Video-Bench has been integrated into [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). As VLMEvalKit currently does not support mixed video and image inputs, we have adapted a version that accepts only image inputs.
Follow the [QuickStart](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) guide of VLMEvalKit and place the required data under the following structure:

  ```
  LMUDATA/
├── MMSIVideo_SC.tsv
├── MMSIVideo_U50.tsv
├── images/
│   └── MMSIVideo/
│       ├── frames/
│       ├── ref_images/
  ```

When evaluating proprietary models, set `img_detail = low`; when evaluating QwenVL-series models, set `max_pixels = 360*420`. Run the evaluation with the following command:

```python 
python run.py --model Qwen2.5-VL-32B-Instruct --data MMSIVideo_SC/MMSIVideo_U50
```

*Note*: The results reported in our paper were generated through the Native Codebase Evaluation. While actual evaluation results may differ slightly from the paper, variations remain within an acceptable range. Potential contributing factors include:

(1) Variations in option ordering, runtime environments, or random seeds, as well as differences in inference configurations between VLMEvalKit and the Native Codebase Evaluation, may lead to fluctuations of 1–2% in the overall score.

(2) For certain models—such as the QwenVL series and the LLaVA-Video series—the input configuration in VLMEvalKit (image-only) differs from that in the Native Codebase (image + video), which may result in a variance of 2–3% in the overall score.

(3) The overall scores of thinking models (e.g., GPT-4o, Gemini 3 Pro, Gemini 2.5 Flash) may also fluctuate within a range of 2–3% across runs.

## 🏆 Leaderboard

<details> <summary>📦 Uniform-50 Setting</summary>

| Model                      | Avg.(%) | Type        |
|----------------------------|---------|-------------|
| Human                   | 96.40   | Baseline    |
|🥇Gemini 3 pro            | 37.97   | Proprietary |
|🥈 O3                      | 36.98   | Proprietary |
|🥉GPT-5                      | 36.80   | Proprietary |
| Gemini 2.5 Flash           | 35.44   | Proprietary |
| Gemini 2.5 Flash (Thinking) | 35.17   | Proprietary |
| Seed-1.6-vision            | 34.87   | Proprietary |
| Claude-haiku-4.5           | 34.27   | Proprietary |
| O4-mini                    | 34.18   | Proprietary |
| QwenVL2.5-72B              | 32.73   | Open-Source |
| InternVL3-78B              | 32.55   | Open-Source |
| Doubao-1.5-thinking        | 31.65   | Proprietary |
| GPT-4o                     | 31.56   | Proprietary |
| InternVL2.5-78B            | 31.37   | Open-Source |
| InternVL2.5-38B            | 31.01   | Open-Source |
| QwenVL3-30B (Thinking)      | 30.83   | Open-Source |
| LLaVA-Video-72B            | 30.38   | Open-Source |
| InternVL3-8B               | 30.38   | Open-Source |
| QwenVL2.5-VL-7B-Instruct   | 29.66   | Open-Source |
| InternVL2.5-8B             | 29.11   | Open-Source |
| InternVL3-38B              | 28.84   | Open-Source |
| QwenVL3-30B                | 28.75   | Open-Source |
| QwenVL2.5-32B              | 28.57   | Open-Source |
| LLaVA-Video-7B             | 28.48   | Open-Source |
| QwenVL3-8B                 | 27.58   | Open-Source |
| InternVideo2.5-8B          | 27.40   | Open-Source |
| Random Guessing            | 24.10   | Baseline    |

</details>

<details> <summary>📦 Sufficient-Coverage Setting</summary>

| Model                      | Avg.(%) | Type        |
|----------------------------|---------|-------------|
| Human                      | 96.4    | Baseline    |
| 🥇O3                         | 37.34   | Proprietary |
| 🥈Gemini 2.5 Flash (Thinking) | 36.71   | Proprietary |
| 🥉Gemini 2.5 Flash           | 36.62   | Proprietary |
| O4-mini                    | 35.08   | Proprietary |
| QwenVL2.5-32B              | 32.37   | Open-Source |
| QwenVL2.5-72B              | 31.83   | Open-Source |
| InternVL3-8B               | 29.57   | Open-Source |
| QwenVL3-30B                | 29.11   | Open-Source |
| QwenVL3-8B                 | 29.09   | Open-Source |
| QwenVL2.5-7B               | 28.84   | Open-Source |
| InternVL2.5-8B             | 28.66   | Open-Source |
| GPT-4o                     | 28.12   | Proprietary |
| QwenVL3-30B (Thinking)      | 28.03   | Open-Source |
| InternVideo2.5-8B          | 26.85   | Open-Source |
| Random Guessing            | 24.10   | Baseline    |

</details>

<details> <summary>🤖 Robot Sub-bench</summary>

| Model                      | Avg.(%) | Type        |
|----------------------------|---------|-------------|
| 🥇Gemini 3 Pro               | 40.20   | Proprietary |
| 🥈Gemini 2.5 Flash (Thinking) | 39.71   | Proprietary |
| 🥉Seed-1.6-vision            | 39.34   | Proprietary |
| O3                         | 39.22   | Proprietary |
| QwenVL2.5-72B              | 37.75   | Open-Source |
| InternVL3-8B               | 37.75   | Open-Source |
| GPT-5                      | 37.75   | Proprietary |
| InternVL2.5-38B            | 36.27   | Open-Source |
| Doubao-1.5-thinking        | 36.07   | Proprietary |
| Gemini 2.5 Flash           | 35.78   | Proprietary |
| O4-mini                    | 35.29   | Proprietary |
| QwenVL2.5-7B               | 34.8    | Open-Source |
| InternVL2.5-78B            | 34.8    | Open-Source |
| Claude-haiku-4.5           | 34.8    | Proprietary |
| InternVL3-78B              | 34.31   | Open-Source |
| LLaVA-Video-72B            | 34.31   | Open-Source |
| QwenVL3-30B                | 32.84   | Open-Source |
| QwenVL2.5-32B              | 32.84   | Open-Source |
| QwenVL3-8B                 | 32.12   | Open-Source |
| InternVideo2.5-8B          | 29.90   | Open-Source |
| GPT-4o                     | 29.90   | Proprietary |
| InternVL2.5-8B             | 28.43   | Open-Source |
| InternVL3-38B              | 27.94   | Open-Source |
| QwenVL3-30B (Thinking)      | 27.94   | Open-Source |
| LLaVA-Video-7B             | 24.51   | Open-Source |


</details>

<details> <summary>🏠 Indoor Scene Perception Sub-bench</summary>

| Model                      | Avg.(%) | Type        |
|----------------------------|---------|-------------|
| 🥇GPT-5                      | 41.68   | Proprietary |
| 🥈O3                         | 40.73   | Proprietary |
| 🥉Gemini 2.5 Flash           | 39.39   | Proprietary |
| Gemini 3 Pro               | 39.39   | Proprietary |
| Gemini 2.5 Flash (Thinking) | 37.86   | Proprietary |
| O4-mini                    | 37.48   | Proprietary |
| Seed-1.6-vision            | 34.2    | Proprietary |
| Claude-haiku-4.5           | 33.46   | Proprietary |
| Doubao-1.5-thinking        | 33.04   | Proprietary |
| InternVL3-78B              | 32.5    | Open-Source |
| QwenVL3-30B (Thinking)      | 32.31   | Open-Source |
| GPT-4o                     | 31.74   | Proprietary |
| QwenVL2.5-72B              | 30.78   | Open-Source |
| InternVL2.5-78B            | 30.4    | Open-Source |
| QwenVL3-30B                | 30.02   | Open-Source |
| QwenVL2.5-32B              | 29.64   | Open-Source |
| InternVL2.5-8B             | 29.45   | Open-Source |
| InternVL3-38B              | 29.06   | Open-Source |
| QwenVL3-8B                 | 28.68   | Open-Source |
| InternVL2.5-38B            | 28.3    | Open-Source |
| LLaVA-Video-72B            | 28.11   | Open-Source |
| InternVL3-8B               | 27.72   | Open-Source |
| LLaVA-Video-7B             | 27.53   | Open-Source |
| QwenVL2.5-7B               | 27.15   | Open-Source |
| InternVideo2.5-8B          | 26.77   | Open-Source |


</details>

<details> <summary>📍 Grounding Sub-bench</summary>

| Model                      | Avg.(%) | Type        |
|----------------------------|---------|-------------|
| 🥇Gemini 2.5 Flash           | 38.81   | Proprietary |
| 🥈Gemini 2.5 Flash (Thinking) | 38.21   | Proprietary |
| 🥉O3                         | 37.61   | Proprietary |
| Doubao-1.5-thinking        | 37.05   | Proprietary |
| InternVL3-78B              | 35.52   | Open-Source |
| GPT-5                      | 35.22   | Proprietary |
| Gemini 3 Pro               | 35.22   | Proprietary |
| O4-mini                    | 34.33   | Proprietary |
| QwenVL2.5-72B              | 34.33   | Open-Source |
| Seed-1.6-vision            | 33.04   | Proprietary |
| Claude-haiku-4.5           | 32.84   | Proprietary |
| InternVL2.5-38B            | 31.94   | Open-Source |
| InternVL3-8B               | 31.94   | Open-Source |
| GPT-4o                     | 31.94   | Proprietary |
| QwenVL3-30B (Thinking)      | 31.64   | Open-Source |
| QwenVL2.5-32B              | 31.04   | Open-Source |
| LLaVA-Video-72B            | 31.04   | Open-Source |
| InternVL3-38B              | 30.45   | Open-Source |
| InternVL2.5-8B             | 30.15   | Open-Source |
| InternVL2.5-78B            | 29.85   | Open-Source |
| QwenVL3-30B                | 29.25   | Open-Source |
| QwenVL2.5-7B               | 28.66   | Open-Source |
| QwenVL3-8B                 | 28.66   | Open-Source |
| InternVideo2.5-8B          | 27.76   | Open-Source |
| LLaVA-Video-7B             | 27.16   | Open-Source |

</details>

*Note: For the three sub-benchmarks, we take the higher score of each model across the two settings for easier presentation.*

## 🔗 Citation

```bibtex
@misc{lin2025mmsivideobenchholisticbenchmarkvideobased,
      title={MMSI-Video-Bench: A Holistic Benchmark for Video-Based Spatial Intelligence}, 
      author={Jingli Lin and Runsen Xu and Shaohao Zhu and Sihan Yang and Peizhou Cao and Yunlong Ran and Miao Hu and Chenming Zhu and Yiman Xie and Yilin Long and Wenbo Hu and Dahua Lin and Tai Wang and Jiangmiao Pang},
      year={2025},
      eprint={2512.10863},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.10863}, 
}
```

## 📄 License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## Acknowledgment
MMSI-Video-Bench utilizes data from the following open-source datasets: Roomtour3d, ScanNet, ScanNet++, 3RScan, ARKitScenes, RealEstate10k, DL3DV, Waymo, NuScenes, OVIS, TrackingNet, LaSOT, UAV123, Ego4D, EPIC-KITCHENS, EgoExoLearn, MultiSports, charades, LEMMA, TF2023, CVMHT, AVA, DROID, RH20T, DTU. We sincerely thank the respective teams for their valuable contributions to the research community.



## Contact
- Jingli Lin: linjingli166@gmail.com
- Runsen Xu: runsxu@gmail.com