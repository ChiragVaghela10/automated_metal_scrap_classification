# Study of image preprocessing steps on scrap metal classification using object detection [Master's Thesis]

## Overview
The large-scale enterprise SMS group has expertise in the metal industry and industrial processes. It is a multinational 
company that values sustainability and social-corporate responsibilities. Consequently, it emphasizes recycling of the 
metal products turned into scraps after consumption.

Scrap-based Electric Arc Furnaces (EAFs) play a vital role in controlling carbon emissions during the steelmaking 
process. The sorted scraps based on their shape, size, and chemical composition are sent to the EAFs for further 
processing. Each EAF operates on different temperatures and other parameters based on the type of scrap used in the EAF.

At present, the steel scraps are sorted manually and sent to different EAFs with the help of
electromagnetic scrap lifter. This process requires trained persons for monitoring and
manual efforts to execute the task. This manual sorting can be automated with the help of computer-vision-based 
techniques.

- **Project Title:** Study of Image Preprocessing Steps on Scrap Metal Classification using YOLOv5 Algorithm
- **Author:** Chirag Vaghela
- **Degree:** Master of Science in Data Engineering
- **Institution:** Jacobs University Bremen
- **Supervisors:** Prof. Dr. Stephan Kettemann, Dr. Christoph Kirmse
- **Collaboration:** SMS digital GmbH, Mönchengladbach

## Problem Statement
This project automates the detection and classification of metal scrap in railway containers through computer vision. 
The implementation uses YOLOv5's small model architecture (YOLOv5s) to 
perform automated detection and classification of metal scrap materials from container top-view imagery. 
The solution streamlines the process of identifying and categorizing different types of steel scrap through 
advanced object detection techniques.

Images captured in scrapyard environments face various quality challenges, including dust, inconsistent lighting, and 
equipment-related issues. These factors can significantly impact model accuracy. Another key focus of this project is 
analyzing how different image preprocessing techniques, applied to the training dataset, affect the model's benchmark 
performance.

## Publication
Co-authored a research paper presented at AISTech 2023. This applied research combined my master’s thesis with an 
industrial-scale project at the SMS group. The solution leveraged YOLOv5 for automated scrap classification using 
real-world scrapyard images. <br/>
🔗 Paper: [Classifications of Scrap Types using Computer Vision algorithms](https://imis.aist.org/AISTPapers/Abstracts_Only_PDF/PR-387-076.pdf) <br/>
📍 Page: 665 | DOI: 10.33313/387/076 <br/>
🧠 Focus areas: image processing, YOLOv5 tuning, model ensemble, and real time inference <br/>
🏛️ Published by: Association for Iron & Steel Technology (AIST) <br/>

## Dataset
- Source: Top-view images of railway scrap containers from SMS group facilities.
- Classes:
    - 0: Turnings
    - 1: Rails
- Format: YOLO annotation format using CVAT tool

Turnings sample             |  Rails sample
:--------------------------:|:-------------------------:
![image](img/turnings.png)  | ![image](img/rails.png)
Table 1: comparison of Turnings and Rails scrap type

## Methodology

The application has mainly two phases: 
1. Training of YOLOv5 on a custom image dataset 
2. Inference of scrap types in images

### 1. Training

#### Preprocessing
The train set images are enhanced using image preprocessing operations before model training. There are mainly three 
types of operations chosen for this project: 
(a) augmentation using the Albumentation library 
(b) non-local denoising operation using OpenCV library
(c) Contrast-Limited Adaptive Histogram Equalization (CLAHE) using OpenCV library 

The operations are also performed in combination to generate more diversified copies of the original train set. The 
training of the YOLOv5 algorithm is performed on these augmented and enhanced datasets.

Each training run is executed separately for original dataset and preprocessed datasets as described in the below table.
The algorithm runs mentioned in the below table are studied into three groups to facilitate visualizations of graphs 
and performance parameters.

| Training Run | Group |                Operations performed on training dataset                 |
|:------------:|:-----:|:-----------------------------------------------------------------------:|
|      1       |   -   |                            Original Dataset                             |
|      2       |   1   |          Dataset with only pixel level augmentations performed          |
|      3       |   1   |            Dataset with only spatial augmentations performed            |
|      4       |   1   | Dataset with both pixel level and spatial level augmentations performed |
|      5       |   2   |        Dataset preprocessed with only CLAHE operation performed         |
|      6       |   2   |      Dataset preprocessed with only denoising operation performed       |
|      7       |   2   | Dataset preprocessed with both denoising and CLAHE operations performed |
|      8       |   3   |       Dataset preprocessed with CLAHE operation and augmentations       |
|      9       |   3   |     Dataset preprocessed with denoising operation and augmentations     |
|      10      |   3   |      Dataset preprocessed with denoising, CLAHE, and augmentations      |
Table 2: Operations performed on train set

The example image with CLAHE operation performed is shown below:

|  Original train set image  | Train set image preprocessed using CLAHE operation |
|:--------------------------:|:--------------------------------------------------:|
| ![image](img/original.png) |              ![image](img/clahe.png)               |
Table 3: comparison of original and preprocessed (CLAHE operation) dataset image

#### Annotations
- Tool: Computer Vision Annotation Tool (CVAT) hosted on the on-site server of SMS Group
- Format: YOLO - normalized `(x_center, y_center, width, height)`
- Each image → `.txt` file with 1 line per bounding box:
```
class_id x_center y_center width height
```
- Albumentation library supports annotations for spatial transformation on images

The original train set images are annotated in YOLO format using as shown below:
<br/><br/>
![image](img/cvat.png)
Figure 1: CVAT Interface for image annotation

```
0 0.3 0.48 0.21 0.17
0 0.8 0.42 0.18 0.16
```

The preprocessed training sets are pushed to AWS S3 bucket using DVC. Then the actual training is performed with 
the project instance running on AWS EC2.

![image](img/training.png) <br/>
Figure 2: Architecture diagram for the training phase

### 2. Inference

The obtained model weights after each training run are stored in the S3 bucket. These trained model weights are used to
perform inference on the test set. The confidence threshold value 0.5 was chosen for detection. The results are also 
saved in text form containing the annotations in addition to predicting bounding box, confidence score, and scrap type 
in images as shown below:

![image](img/inference.png)
Figure 3: (a) Test set image with predicted bounding boxes, scrap type, confidence score (b) text file containing 
labels for the same image

The architecture of the inference phase is shown below: <br/><br/>
![image](img/inference_architecture.png) <br/>
Figure 4: Architecture diagram for the inference phase

## Results
<cvat and other stuffs to be added>

<train/val loss diagrams to be added>

### Key Findings
<>

## Installation
```bash
# Clone and install dependencies
git clone https://github.com/ChiragVaghela10/automated_metal_scrap_classification
cd automated_metal_scrap_classification
pip install -r requirements.txt

# Train YOLOv5
python train.py --data config/data.yaml --weights models/yolov5s.pt --epochs 300

# Run inference
python detect.py --weights models/best.pt --source data/sym_data/scrap/

```

## About Me
For queries or collaboration opportunities, feel free to reach out: 
<br/> 📧 Email: chiragvaghela10@ymail.com
<br/> 💼 LinkedIn: https://www.linkedin.com/in/chiragvaghela/
