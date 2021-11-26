# Progressive refinement network for camouflaged insect detection

> **Authors:** 


## 1. Preface

- This repository provides code for "_**Progressive refinement network for camouflaged insect detection**_". 
([paper]( ) )

- If you have any questions about our paper, feel free to contact me. And if you are using PRNet 
or for your research, please cite this paper ([BibTeX]( )).

## 2. Overview

### 2.1. Introduction


### 2.2. Framework Overview



### 2.3. Results



## 3. How to use

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
an NVIDIA Tesla V100 GPU of 32 GB Memory. 

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that PRNet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n PRNet python=3.6`.
    
    + Installing necessary packages: PyTorch 1.1

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

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).
    
    + Just enjoy it!

### 3.2 Evaluating your trained model:
One-key evaluation toolbox is provided for benchmarking within a uniform standard. 
It is written in MATLAB code ([link](https://drive.google.com/file/d/1_h4_CjD5GKEf7B1MRuzye97H0MXf2GE9/view?usp=sharing)), 
please follow the instructions in `./eval/main.m` and just run it to generate the evaluation results in `./res/`.




## 4. Citation

Please cite our paper if you find the work useful: 

    @article{ 

    }



## 5. License

The source code is free for research and education use only. Any commercial use should get formal permission first.

---


