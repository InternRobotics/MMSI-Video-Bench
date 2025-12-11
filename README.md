<div align="center">

# MMSI-Video-Bench: A Holistic Benchmark for Video-Based Spatial Intelligence
[**🌐 Homepage**](https://rbler1234.github.io/MMSI-VIdeo-Bench.github.io/)  | [**📑 Paper**]() | [**🤗 dataset**](https://huggingface.co/datasets/rbler/MMSI-Video-Bench) | [**📖 arXiv**]()
</div>


<!-- contents with emoji -->

## 🔔 News


🔥[2025-12]: We released our paper, benchmark, and evaluation codes.

## Introduction

<!-- ![Teaser](assets/teaser.jpg) -->

We introduce **MMSI-Video-Bench**, a fully human-annotated benchmark for video-based spatial intelligence in MLLMs. It operationalizes a four-level framework, Perception, Planning, Prediction, and Cross-Video Reasoning, through 1,106 questions grounded in 1,278 clips from 25 datasets and in-house videos. Each item is carefully designed and reviewed by 3DV experts with explanatory rationales to ensure precise, unambiguous grounding. We evaluate 25 strong open-source and proprietary MLLMs, revealing a striking human--AI gap: many models perform near chance, and the best reasoning model lags humans by nearly 60\%. Fine-grained error analysis exposes systematic failures in geometric reasoning, motion grounding, long-horizon prediction, and cross-video correspondence. We also show that typical frame-sampling strategies transfer poorly to our benchmark due to its reasoning-intensive nature. We expect our benchmark to establish a solid testbed for advancing video-based spatial intelligence.

<div style="text-align: center;">
    <img src="assets/teasor.png" alt="Dialogue_Teaser" width=100% >
</div>


## Features of MMSI-Video-Bench

- **High quality.** All data are manually annotated by 11 domain experts in 3D vision, following a rigorous review and acceptance process to ensure annotation accuracy and reliability.

- **Challenging.** We evaluate 25 strong open-source and proprietary MLLMs, revealing a striking human–AI gap, even the best reasoning model trails human performance by nearly 60%.

- **Diverse Video Sources.** MMSI-Video-Bench includes videos from 25 public datasets and 1 in-house collection, spanning tabletop recordings, indoor and multi-floor environments, outdoor scenes, sports, and movie footage.

- **Comprehensive and Hostile Task Coverage.** The benchmark covers spatial layout reasoning, motion understanding, decision-making, and cross-video reasoning, providing a more holistic evaluation of video-based spatial intelligence.



## Example

The questions in MMSI-VIdeo-Bench span 5 major categories: **(1).Spatial Construction**：This category focuses on spatial attributes of instances and scenes, as well as spatial relationships among instances, scenes, and cameras (six subtypes in total); **(2).Motion Understanding**:
This includes understanding camera motion, instance motion, and interactive motion between instances; **(3).Planning** based on spatiotemporal video information;  **(4).Prediction**: Assessing a model’s ability to predict, anticipate, or imagine future states based on the observed video; **(5).Cross-Video Reasoning**: This involves memory update across temporally separated video segments and multi-view integration across videos captured from different viewpoints.


<div style="text-align: center;">
    <img src="assets/benchmark_samples.png" alt="Dialogue_Teaser" width=100% >
</div>

## 🚀 Getting Started

### Installation

1. Clone Github repo.

2. Install requirements.

   ```shell
   conda activate your_env_name
   pip install -r requirements.txt
   ```

   *Note:* If you want to evaluate open-source models, you need to set up their corresponding environments.

### Data Preparation

  Download the MMSI-Video-Bench data from [Hugging Face](https://huggingface.co/datasets/rbler/MMSI-Video-Bench). The dataset includes:

(1) Annotations: `mmsivideo.json`.

(2) Reference images for questions: `ref_images.zip`.

(3) Video frames: `frames.zip`.

(Optional) Original video files: `videos.zip`.

After downloading, unzip the files and organize them as follows:

  ```
  |-data/
  |-- mmsivideo.json
  |-- frames/
  |-- ref_images/
  |-- videos/
  ```

  For more detail about the json-format data, refer to [documention](https://huggingface.co/datasets/rbler/MMSI-Video-Bench).

## 👓 Evaluation

Please note that while **Sufficient Coverage** ensures that all video information is fully preserved, we **Recommend** using this setting for evaluation.
Evaluation under the **Uniform** setting may lead to missing critical information. The Uniform-50 setting is only provided due to current input-length limitations in some models.

1. Run infernece

    For open-source models, change the openai `base_url` and `api_key` to your own in `utils/openai_api.py`. For proprietary models, modify the `load_model` function in `inference.py` to use the corresponding model path. Run the following command to perform inference for a specific model under a particular setting:
    ```python
    python inference.py --model_name {model_name} --setting Uniform-50/Sufficient-Coverage
    ```

2. Run evaluation
   
   Run the following command to obtain scores for a specific benchmark. The default is the main benchmark.

   ```python
    python evaluation.py --eval_dir {path/to/results} --bench main/robot_bench/ground_bench/indoor_perception_bench/easy2hard_bench
    ```

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
| 🥇gemini-2.5-flash           | 38.81   | Proprietary |
| 🥈gemini-2.5-flash(thinking) | 38.21   | Proprietary |
| 🥉o3                         | 37.61   | Proprietary |
| doubao-1-5-thinking        | 37.05   | Proprietary |
| InternVL3-78B              | 35.52   | Open-Source |
| gpt-5                      | 35.22   | Proprietary |
| gemini-3-pro               | 35.22   | Proprietary |
| o4-mini                    | 34.33   | Proprietary |
| QwenVL2.5-72B              | 34.33   | Open-Source |
| seed-1-6-vision            | 33.04   | Proprietary |
| claude-haiku-4-5           | 32.84   | Proprietary |
| InternVL2.5-38B            | 31.94   | Open-Source |
| InternVL3-8B               | 31.94   | Open-Source |
| gpt-4o                     | 31.94   | Proprietary |
| QwenVL3-30B(Thinking)      | 31.64   | Open-Source |
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