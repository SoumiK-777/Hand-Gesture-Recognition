
# Personalized Hand Gestures Recognition for Controlling Vehicle Infotainment

Traditional interfaces for vehicle infotainment systems, such as physical buttons or touchscreens, often cause distractions and may not always be the most user-friendly or convenient, especially while driving. These distractions can compromise driver safety and reduce the overall driving experience.

To address these issues, we propose the development of a personalized hand gesture recognition system utilizing computer vision and artificial intelligence (AI) techniques. Our solution aims to provide a seamless and intuitive user experience by enabling drivers to control infotainment functions through natural hand movements.

## Proposed Solution

To address the challenge of traditional vehicle infotainment interfaces, we propose a personalized hand gesture recognition system based on few-shot learning. Few-shot learning enables a model to learn and make accurate predictions from a limited number of training examples. In this context, drivers will provide only a few samples of their personalized hand gestures, and our model will be able to recognize and predict these gestures with high accuracy.

## Timeline
The research project was done under the supervision of Prof. Dr. Soodeh and as a part of my 12-week MITACS Globalink Research Internship at Western University, London, Ontario.

| Task                     | Completed |
|--------------------------|-----------|
| Week 3: Complete data collection and pre-processing  | ✅ |  
| Week 6: Achieve baseline performance on hand gesture
recognition      | ✅ | 
| Week 10: Functional prototype of the personalized
infotainment control system | ✅ |  


## Dataset Overview
#### HaGRID - HAnd Gesture Recognition Image Dataset

HaGRID lightweight version size is 26.4GB and dataset contains 554,800 FullHD RGB images divided into 18 classes of gestures.

 Gesture                           | No. of samples    | Gesture                                   | No. of samples    |
|-----------------------------------|---------|-------------------------------------------|---------|
| [`call`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/call.zip)    | 37.2 GB | [`peace`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/peace.zip)           | 41.4 GB |
| [`dislike`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/dislike.zip) | 40.9 GB | [`peace_inverted`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/peace_inverted.zip)  | 40.5 GB |
| [`fist`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/fist.zip)    | 42.3 GB | [`rock`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/rock.zip)            | 41.7 GB |
| [`four`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/four.zip)    | 43.1 GB | [`stop`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/stop.zip)            | 41.8 GB |
| [`like`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/like.zip)    | 42.2 GB | [`stop_inverted`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/stop_inverted.zip)   | 41.4 GB |
| [`mute`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/mute.zip)    | 43.2 GB | [`three`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/three.zip)           | 42.2 GB |
| [`ok`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/ok.zip)      | 42.5 GB | [`three2`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/three2.zip)          | 40.2 GB |
| [`one`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/one.zip)     | 42.7 GB | [`two_up`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/two_up.zip)          | 41.8 GB |
| [`palm`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/palm.zip)    | 43.0 GB | [`two_up_inverted`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/two_up_inverted.zip) | 40.9 GB |

The dataset was divided into three parts:
- Training
- Validation
- Testing

[Access the dataset](https://github.com/hukenovs/hagrid)\
[Access the official arxiv paper](https://arxiv.org/abs/2206.08219)


## Methodologies

- Siamese Network
- Prototypical Network
- Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions

## Installation

## Usage

## Model Training

## Evaluation

## References

