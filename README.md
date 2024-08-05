
# Personalized Hand Gestures Recognition for Controlling Vehicle Infotainment

Traditional interfaces for vehicle infotainment systems, such as physical buttons or touchscreens, often cause distractions and may not always be the most user-friendly or convenient, especially while driving. These distractions can compromise driver safety and reduce the overall driving experience.

To address these issues, we propose the development of a personalized hand gesture recognition system utilizing computer vision and artificial intelligence (AI) techniques. Our solution aims to provide a seamless and intuitive user experience by enabling drivers to control infotainment functions through natural hand movements.

## Proposed Solution

To address the challenge of traditional vehicle infotainment interfaces, we propose a personalized hand gesture recognition system based on few-shot learning. Few-shot learning enables a model to learn and make accurate predictions from a limited number of training examples. In this context, drivers will provide only a few samples of their personalized hand gestures, and our model will be able to recognize and predict these gestures with high accuracy.

## Timeline
The research project was done under the supervision of Prof. Dr. Soodeh Nikan and as a part of my 12-week MITACS Globalink Research Internship at Western University, London, Ontario, Canada.

| Task                          | Completed |
|-------------------------------|-----------|
| Week 3: Complete data collection and pre-processing  | ✅ |  
| Week 6: Achieve baseline performance on hand gesture
recognition      | ✅ | 
| Week 10: Functional prototype of the personalized
infotainment control system | ✅ |  


## Dataset Overview
#### HaGRID - HAnd Gesture Recognition Image Dataset

HaGRID lightweight version size is 26.4GB and dataset contains 88,991 FullHD RGB images divided into 18 classes of gestures. The dataset is used to develop Hand Gesture Recognition Systems which can be used in video conferencing services (Zoom, Skype, Discord, Jazz etc.), home automation systems, the automotive sector, etc. There are at least 37,583 distinct individuals and scenes in the collection. The subjects range in age from eighteen to sixty-five. The majority of the data collection took place inside, where artificial and natural light were present in varying degrees. Additionally, the dataset contains photos that were taken under challenging circumstances, such backing up against a window. Additionally, the subjects had to make motions between 0.5 and 4 meters away from the camera.

![Dataset](./imgs/Screenshot%20from%202024-08-04%2023-33-50.png)

Dataset folder structure:
```
data 
│
└─── gesture1
│   │   img000.jpg
│   │   img001.jpg
|   |   ....
│   
└─── gesture2
│   │   img000.jpg
│   │   img001.jpg
|   |   ....
|
└─── gesture3
│   │   img000.jpg
│   │   img001.jpg
|   |   ....
│   │
```
| Gesture          | Samples | Gesture         | Samples |
|------------------|---------|-----------------|---------|
| call             | 5089    | like            | 4753    |
| dislike          | 4850    | mute            | 5013    |
| fist             | 4932    | ok              | 4875    |
| four             | 4886    | one             | 4801    |
| palm             | 4948    | peace           | 4901    |
| peace_inverted   | 5112    | rock            | 4846    |
| stop             | 4928    | stop_inverted   | 4983    |
| three            | 4867    | three2          | 5124    |
| two_up           | 5057    | two_up_inverted | 5026    |

#### Data preprocessing

The dataset was divided into three parts:
- Training (80%)
- Validation (10%)
- Testing (10%)

[Access the dataset](https://github.com/hukenovs/hagrid)\
[Access the official arxiv paper](https://arxiv.org/abs/2206.08219)


## Methodologies

- ### Siamese Network
    These type of neural network architecture designed for tasks that involve comparing two inputs to determine their similarity. They are frequently used in image classification, especially in cases involving few-shot learning or tasks where the goal is to determine if two images are of the same class.
    - #### Architecture of Feature Extraction Layer
        The network consists of two identical subnetworks which share the same architecture and weights. Each subnetwork extracts features from its input image, using convolutional layers. These features are then represented as high-dimensional vectors.
    - #### Loss Function
        We used the contrastive loss for training our siamese network.  It encourages the network to output a small distance for similar pairs and a larger distance for dissimilar pairs.

      | Training Loss | Validation Accuracy | Test Accuracy |
    |---------------|---------------------|---------------|
    | 0.00136          | 100%                 | 99.87%    |

- ### Prototypical Network
    Given only a few examples of each new class, prototype networks generalize to new classes not observed in the training set. Prototypical networks pick up a metric space where distances to prototype representations of each class can be calculated for the purpose of classification. They obtain outstanding results and reflect a simpler inductive bias that is useful in this limited-data domain, compared to other techniques for few-shot learning.

    A custom resnet12 backbone extracts the input features with SGD optimizer in the provided approach.

  ![Proto Loss and Curves](./imgs/Screenshot%20from%202024-08-03%2021-34-09.png)

    | Training Loss | Validation Accuracy | Test Accuracy |
    |---------------|---------------------|---------------|
    | 0.00358          | 100%                 | 99.34%           |
- ### Few-shot Embedding Adaptation with Transformer (FEAT)
    A novel model-based approach is proposed to adapt instance embeddings to the target classification task using a set-to-set function, yielding task-specific and discriminative embeddings. Various instantiations of such set-to-set functions were empirically investigated, with the Transformer being the most effective due to its alignment with key desired properties. This method is denoted as a Few-shot Embedding Adaptation with Transformer (FEAT).

    A custom resnet12 backbone with pre-trained weights from the MiniImageNet Dataset extracts the input features with SGD optimizer in the provided approach.

  ![FEAT Loss and Curves](./imgs/Screenshot%20from%202024-08-03%2021-56-41.png)

    | Training Loss | Validation Accuracy | Test Accuracy |
    |---------------|---------------------|---------------|
    | 0.00136          | 100%                 | 99.87%    |


<!-- ## Installation

## Usage

## Model Training

## Evaluation

## References -->

