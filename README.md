# MASA: Machine Learning Assisted Swallowing Assessment

## Project Description
MASA (Machine Learning Assisted Swallowing Assessment) is an innovative project designed to leverage the power of machine learning (ML) to transform the way swallowing assessments are performed, with an aim to improve quality of care. The primary goal of this project is to harness ML capabilities in spectral analysis of human voice during swallowing assessments to classify the assessment state as per the screening standard (at our site it is TOR-BSST&copy;, however many other screening tests are used at other centers), a recognized tool for assessing swallowing disorders particularly after acute stroke.

**Please also note the [MASA supplementary material](https://github.com/UofTNeurology/masa-open-source/blob/main/MASA%20supplementary%20material.pdf) methods accompanying our paper.**

This project involves the application of advanced techniques such as Convolutional Neural Networks (CNNs) initially, with plans to expand to Visual Transformers (ViT) in the future. The use of these cutting-edge technologies is aimed at capturing and understanding the intricate nuances of human voice during swallowing assessments that are potentially missed in traditional assessments.

The algorithms developed under MASA aim to bring efficiency, accuracy and scalability in the assessment process, potentially enabling clinicians to make more informed decisions regarding patient treatment and management.

## Project Objectives
Develop a robust machine learning model capable of performing spectral analysis on human voice during swallowing assessments.
Validate the model's performance against screening test labeling.
Improve the quality and efficiency of swallowing assessments, particularly in the context of post-acute stroke care.
Explore the incorporation of advanced technologies such as Visual Transformers to enhance the model's capabilities.
Getting Started
Refer to the Getting Started Guide for instructions on how to install, run, and use this project.

## How to Get Started


This repository uses Docker containers to run the machine learning notebooks. Prior to starting, we recommend installing and updating Docker to the latest version. If you are using Linux, make sure you set up the [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to be able to use GPU acceleration. This repository was tested on Windows 11 and Ubuntu 22.04.


You can use the [Makefile](Makefile) to automate the process of building and running the notebooks. 

To build the tensorflow docker:

```
make build-tensorflow
```
To run the tensorflow docker:
```
make run-tensorflow
```
Please note the docker images will automatically start jupyter lab servers as this repository mainly relies on python notebooks for the ML experiments. You can similarly run the Pytorch docker images.


After running the Docker Image, open Jupyter Lab in your browser and go the [sample base notebook](Notebooks/Base%20(Single%20Network)-Densenet.ipynb) 


If you have prepared the dataset and placed it under the Audio Data folder (see the next section on preparing the dataset), you should be able to simply run the CNN notebook without modifications.


## Preparing the Dataset


This repository provides preprocessing code to convert raw audio signals to spectrogram images.
### Introduction
The Audio Processing Toolkit is designed to facilitate the loading, processing, and analysis of `.wav` audio files. This toolkit can handle multiple audio transformations like Mel spectrograms and Superlets. It also has utilities for batch processing of audio files, categorizing them into 'Pass' and 'Fail' based on certain conditions.

### Features
- Load `.wav` audio files from a specified directory
- Apply Superlet and Mel transformations
- Export processed data as images
- Generate histograms for 'Pass' and 'Fail' sound file lengths
- Supports custom epoch durations, sample rates, and more

### Installation
```bash
pip install -r requirements.txt
```
### Usage
To use the toolkit, you will need to edit the `main.py` file. Here are the primary areas you might want to customize:

### 1. Input and Output Directories
Edit the following lines to specify the directory containing the `.wav` files you wish to process and where the output should be saved:
```python
path = "Audio Data/For Processing"
output_path = "Audio Data/Outputs/"
```

### 2. Audio Processing Parameters
You can customize the epoch duration, overlap, and other parameters by editing these lines:
```python
epoch_duration = 0.5
overlap = 0.5
min_power = 0
target_sampling_rate = 22050
output_type = 'mel'
```

After configuring, run `main.py` to start the processing. This will create the dataset that will be used in the deep learning notebooks included in this repository.

### 3. Organize Image Outputs In The Following Structure:
- Dataset Parent Directory
  - Train
    - Pass
      - Participant_1
        - spectrogram1.png
        - spectrogram2.png
        - ...
      - Participant_2
        - spectrogram1.png
        - spectrogram2.png
    - Fail
      - Participant_3
        - spectrogram1.png
        - spectrogram2.png
  - Test
    - Pass
      - Participant_5
        - spectrogram1.png
        - spectrogram2.png
    - Fail
      - Participant_7
        - spectrogram1.png
        - spectrogram2.png



## Contact
Please feel free to raise an issue for any queries, suggestions, or discussions.

This project is a stepping stone towards improving the quality of patient care by leveraging the potential of AI and machine learning in healthcare. We welcome you to join us on this exciting journey!
