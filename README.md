# Analysis of unimodal Music Emotion Recognition using audio features

This repository presents different approaches to the Music Emotion Recognition task from a regression perspective, using Convolutional Neural Networks trained on MFCC audio features.  
Considering a 2D representation of emotions defined by **valence** and **arousal** dimensions, two types of models are created:  
 - ***2D-output*** model : predicts values for both valence and arousal  
 - ***1D-output*** models : separately predict values for valence and arousal, respectively

### For the complete description of the methods and experiments, please refer to [unimodal MER](https://gloria-m.github.io/unimodal.html).  

<br/>  

## Dataset

The dataset used is [The MediaEval Database for Emotional Analysis of Music](https://www.researchgate.net/publication/314656874_Developing_a_benchmark_for_emotional_analysis_of_music) , consisting in 1,744 song excerpts of ~45sec duration, with two types of annotations for valence and arousal available: **dynamic** —measured per second— and **static** —measured per 45sec. In this project, the ***static annotations*** are used.

### Data path structure

The data directory should have the following structure:
```
.
├── Data
    ├── DEAM_dataset
    │   ├── Audio
    │   │   ├── *.mp3
    │   ├── static_annotations.csv
```  

## Usage

### 1. Prepare data

#### run `python main.py --mode=preprocess`  
Resume training by specifying a valid value for `--restore_epoch`.  
> The model saved as `Models/checkpoint_<restore_epoch>.pt` will be loaded

<!-- ### 2. Train

#### run `python main.py`  

Control the training by modifying the default values for the following parameters:
```
--device = cuda (train on cuda)  
--log_interval = 1 (print train & validation loss each epoch)
--checkpoint_interval = 100 (save trained model and optimizer parameters every 100 epochs)
--num_epochs = 500
```

### 3. Test

#### run `python main.py --mode=test --restore_epoch=* --test_ct_names=*`  
Test the model saved at training epoch `--restore_epoch` on CT images specified.
> `--test_ct_names` accepts a list of the CT images without the `.npy` extension.
> > for example, the CT image located at `Data/Test/ct_sample1.npy` will be passed as `ct_sample1`.  

> The model saved as `Models/checkpoint_<restore_epoch>.pt` will be loaded.

<br/>   -->

### Tools  
`PyTorch`, `librosa`
