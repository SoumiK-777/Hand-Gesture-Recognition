
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

HaGRID lightweight version size is 26.4GB and dataset contains 88,991 FullHD RGB images divided into 18 classes of gestures.

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
- Training
- Validation
- Testing

[Access the dataset](https://github.com/hukenovs/hagrid)\
[Access the official arxiv paper](https://arxiv.org/abs/2206.08219)


## Methodologies

- ### Siamese Network
    These type of neural network architecture designed for tasks that involve comparing two inputs to determine their similarity. They are frequently used in image classification, especially in cases involving few-shot learning or tasks where the goal is to determine if two images are of the same class.
    - #### Architecture of Feature Extraction Layer
        The network consists of two identical subnetworks which share the same architecture and weights. Each subnetwork extracts features from its input image, using convolutional layers. These features are then represented as high-dimensional vectors.
    - #### Loss Function
        We used the contrastive loss for training our siamese network.  It encourages the network to output a small distance for similar pairs and a larger distance for dissimilar pairs. The loss is computed as:

            $$$
            \text{Loss} = \frac{1}{2} \left[ y \cdot D^2 + (1 - y) \cdot \max(0, \text{margin} - D)^2 \right]
            $$$

            Where:
            - \( y \) is the label indicating whether the two input images are from the same class (1 for same, 0 for different).
            - \( D \) is the Euclidean distance between the feature vectors of the two images.
            - \( \text{margin} \) is a parameter that defines how far apart dissimilar pairs should be.

    - #### Training and Evaluation


- ### Prototypical Network
- ### Few-shot Embedding Adaptation with Transformer (FEAT)

## Installation

## Usage

## Model Training

## Evaluation

## References

