# Analysis of unimodal Music Emotion Recognition using audio features

This repository presents different approaches to the Music Emotion Recognition task from a regression perspective, using Convolutional Neural Networks trained on MFCC audio features.  
Considering the 2D representation of emotions defined by **valence** and **arousal** dimensions [[Circumplex Model of Affect]](https://www.researchgate.net/publication/235361517_A_Circumplex_Model_of_Affect), two types of models are created:  
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

> Extract annotations and audio names from `static_annotations.csv`  
> Augment dataset
> Extract MFCC features from waveforms
> Make train and test sets

### 2. Train

#### run `python main.py`  

There are three options for training:  
 - 2D-output model: `--dimension=both` will create a model to predict both valence and arousal, with filter size defined in `--params_dict`  
 - valence model: `--dimension=valence` will create a model to predict valence with filter size defined in `--valence_params_dict`  
 - arousal model: `--dimension=arousal` will create a model to predict valence with filter size defined in `--arousal_params_dict`  
  
Control the training by modifying the default values for the following parameters:
```
--device = cuda (train on cuda)  
--log_interval = 1 (print train & validation loss each epoch)
--num_epochs = 2000
```  

### 3. Test

#### run `python main.py --mode=test --dimension=*`  
  
> The model saved as `Models/model_<dimension>.pt` will be loaded.  
> - for the 2D-output model: `--dimension=both`  
> - for the valence model: `--dimension=valence`  
> - for the arousal model: `--dimension=arousal` 

<br/>  

### Tools  
`PyTorch`, `librosa`
