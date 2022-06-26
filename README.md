# Camouflaged insect segmentation using progressive refinement network

> **Authors:** 
> [Jing Wang*](https://abigale666.github.io/wangjing_homepage.github.io/), 
> [Minglin Hong*](https://hongminglin08.github.io/index-en.html), 
> [Xia Hu*](https://lxy.fafu.edu.cn/37/85/c6857a145285/page.htm),
> [Xiaolin Li](https://orcid.org/0000-0002-0997-2776),
> [Shiguo Huang](https://xxxy.fafu.edu.cn/2017/0604/c7843a183612/page.psp),
> [Rong Wang](https://lxy.fafu.edu.cn/38/8d/c6857a145549/page.psp), and 
> [Feiping Zhang](https://lxy.fafu.edu.cn/37/7c/c6857a145276/page.psp).

## 0. Preface

- This repository provides the source code and evaluation toolbox for "_**Camouflaged insect segmentation using progressive refinement network**_". 
([paper]( ))

- If you have any questions about our paper, feel free to contact [Jing Wang](wangjingsay@gmail.com) 
or [Mingling Hong](2663321079@qq.com) via E-mail. And if you are using PRNet for your research, please cite this paper. ([BibTeX](#3-citation)).

## 1. Overview

### 1.1. Introduction
Accurately segmenting the insect from its original ecological image is the core technology restricting the accuracy and efficiency of automatic recognition. However, the performance of existing segmentation methods is unsatisfactory in insect images shot in wild backgrounds with several challenges: various sizes, similar colours or textures to the surroundings, transparent body parts and vague outlines.
<p align="center">
    <img src="imgs/salient insects and camouflaged insects.png"/> <br />
    <em> 
    Figure 1: Salient insects and camouflaged insects.
    </em>
</p>
These challenges of image segmentation are accentuated when dealing with camouflaged insects. Here, we developed an insect image segmentation method based on deep learning termed progressive refinement network (PRNet), especially for camouflaged insects. Unlike existing insect segmentation methods. PRNet captures the possible scale and location of insects by extracting the contextual information of the image, and fuses comprehensive features to suppress distractors, thereby clearly segmenting insect outlines.

Experimental results based on 1900 camouflaged insect images demonstrated PRNet could effectively segment the camouflaged insects and achieved a superior detection performance, with the mean absolute error of 3.2\%, pixel matching degree of 89.7\%, structural similarity of 83.6\%, and precision and recall error of 72\%, which achieved the improvement of 8.1\%, 25.9\%, 19.5\%, and 35.8\%, respectively, when compared to the recent salient object detection methods.

### 1.2. Framework Overview
<p align="center">
    <img src="imgs/framework.png"/> <br />
    <em> 
    Figure 2: Overview of the proposed PRNet, which consists of Asymmetric Receptive Field, Self-Refinement Module, and Reverse Guidance Based Decoder. See § 2 in the paper for details.
    </em>
</p>

### 1.3. Results

<p align="center">
    <img src="imgs/qualitative_results.png"/> <br />
    <em> 
    Figure 3: Qualitative Results.
    </em>
</p>

<p align="center">
    <img src="imgs/quantitative_results.png"/> <br />
    <em> 
    Figure 4: Quantitative Results.
    </em>
</p>

## 2. How to use?


PRNet can be run on Windows, Linux, or MacOS. And a GPU should be in your machine, if not, use Google Colaboratory GPUs for free
(read more [here](https://github.com/DeepLabCut/DeepLabCut/tree/master/examples#demo-4-deeplabcut-training-and-analysis-on-google-colaboratory-with-googles-gpus) and there are a lot of helper videos on [our YouTube channel!](https://www.youtube.com/playlist?list=PLjpMSEOb9vRFwwgIkLLN1NmJxFprkO_zi)).

### Step 1: You need to have Python virtual environment installed

> Anaconda is perhaps the easiest way to install Python and additional packages across various operating systems. With Anaconda you create all the dependencies in an [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) on your machine.

- Simply download the appropriate files here: https://www.anaconda.com/distribution/

- Verify the current version of conda by `conda-v`
       
- Create a virtual environment in terminal: `conda create -n PRNet python=3.6`, where `PRNet` is the name of the virtual environment, `python=3.6` sets the version of python as 3.6. 

### Step 2: Install necessary packages: PyTorch

- To check your GPU is working, in the terminal, run: `nvcc -V` to check your installed version(s).

- Click [HERE](https://pytorch.org/get-started/previous-versions/) to check the pytorch version required for your CUDA, and download it.

- Now, in Terminal (or Anaconda Command Prompt for windows users), go to the folder where you downloaded the file.    
For example, if you downloaded it from The CLICK HERE above, it likely went into your downloads folder: `cd C:\Users\YourUserName\Downloads`

- Get the location and in the terminal run: `pip install torch-XX-XX.whl` and `pip install torchvision-XX-XX.whl` to install the PyTorch

      - To get the location right, a cool trick is to drag the folder and drop it into Terminal. 
        Alternatively, you can (on Windows) hold SHIFT and right-click > Copy as path, or (on Mac) right-click and while in the menu press the OPTION key to reveal Copy as Pathname.
      
      - If the virtual environment have installed, for reference, you can install the necessary packages directly by `pip install -r requirements.txt`.

### Step 3: To train/test PRNet

1. Downloading necessary data:

    + downloading the dataset from this [download link (Google Drive)]( ).
      Partitioning the testing dataset and training dataset into `./data/TestDataset/`,`./data/TrainDataset/`, respectively.
    
    + downloading pretrained weights and move it into `snapshots/PRNet/`, 
    which can be found in this [download link (Google Drive)]( ).
    
    + downloading Res2Net weights and move it into `snapshots/Backbone/`
     which can be found in this [download link (Google Drive)]( ).
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `MyTrain.py`.
    
    + Just enjoy it!

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map
    after replacing your trained model directory (`--pth_path`).
    
    + Just enjoy it!
    

### Step 4: Evaluation Toolbox
One-key evaluation toolbox is provided for benchmarking within a uniform standard. 
It is written in MATLAB code ([link](https://drive.google.com/file/d/1_h4_CjD5GKEf7B1MRuzye97H0MXf2GE9/view?usp=sharing)), 


- Prerequisites: MATLAB Software.

- run `cd ./eval/` and `matlab` open the Matlab software via terminal.

- Edit the parameters in the `main.m` to evaluate your custom methods. Please refer to the instructions in the `main.m`.

- Just run `main.m` to get the overall evaluation results in `./res/`.

> Python Version: Please refer to the work of ACMMM2021 https://github.com/plemeri/UACANet 

## Tips:

- Computer:

     - For reference, we use e.g. NVIDIA Tesla V100 GPU of 64 GB Memory with CentOS 7.8.2003.

    > Note that our model also supports low memory GPU, which means you can lower the batch size
    (~419 MB per image in `apex-mode=O1`, and ~305 MB per image in `apex-mode=O2`)

- Computer Hardware:
     - Ideally, you will use a strong GPU with *at least* 8GB memory such as the [NVIDIA GeForce 1080 Ti or 2080 Ti](https://www.nvidia.com/en-us/shop/geforce/?page=1&limit=9&locale=en-us).  
     
      The ONLY thing you need to do **first** if you have an NVIDIA GPU and the matching NVIDIA CUDA+driver installed.

      - CUDA: https://developer.nvidia.com/cuda-downloads (just follow the prompts here!)

      - DRIVERS: https://www.nvidia.com/Download/index.aspx
     
     - A GPU is not necessary, but on a CPU the (training and evaluation) code is considerably slower (10x) for model. You might also consider using cloud computing services like [Google cloud/amazon web services](https://github.com/DeepLabCut/DeepLabCut/issues/47) or Google Colaboratory.


- Software:
     - Operating System: Linux, MacOS or Windows 10. However, the authors strongly recommend Linux! *MacOS does not support NVIDIA GPUs (easily), so we only suggest this option for CPU use or a case where the user wants to label data, refine data, etc and then push the project to a cloud resource for GPU computing steps.
     
     - Anaconda/Python3: Anaconda: a free and open source distribution of the Python programming language (download from https://www.anaconda.com/). DeepLabCut is written in Python 3 (https://www.python.org/) and not compatible with Python 2.
     
 - Image Pre-process:
     - The PRNet is an end-to-end tool to segment images from any camera (cell phone cameras, grayscale, color; different manufacturers, etc.). 
    No additional processing of the imagse is required when you use the PRNet, please put the pending images into the `./data/TestDataset/` and further just run `MyTest.py` to generate the final prediction map in `save_path`.
    

## 4. Citation

Please cite our paper if you find the work useful: 

    @inproceedings{Wang2022Camouflage,
  	    title={Camouflaged insect segmentation using progressive refinement network},
  	    author={Wang, Jing and Hong, Minglin and Hu, Xia and Li, Xiaolin and Huang, Shiguo and Wang, Rong and Zhang. Feiping},
  	    booktitle={Methods in Ecology and Evolution},
  	    year={2022}
	}

## 5. FAQ

1. If the image cannot be loaded in the page (mostly in the domestic network situations).

    [Solution Link](https://blog.csdn.net/weixin_42128813/article/details/102915578)

## 6. License

The source code is free for research and education use only. Any commercial use should get formal permission first.

---

**[⬆ back to top](#0-preface)**
