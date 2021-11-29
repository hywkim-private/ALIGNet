#DEFINE global variables 
import torch
import os

def initialize_config():
        #define hyper parameters 
    global USE_CUDA
    global DEVICE
    global CPU
    global EPOCHS 
    global BATCH_SIZE 
    
    global TARGET_PROPORTION
    global TARGET_PROPORTION_VAL
    global TARGET_PROPORTION_TEST 
    global TRAIN_SIZE
    global VAL_SIZE

    #parameters for mask operation
    global MASK_SIZE
    global AUGMENT_TIMES_TR
    global AUGMENT_TIMES_VAL
    
    
    global MODEL_PATH
    #signifies which model to use 
    global MODEL_IDX
    #path to save/load the model 
    global MODEL_PATH_VASE
    global MODEL_PATH_PLANE
    #parameters for the warp-field/grid
    global GRID_SIZE
    global VOX_SIZE
    #number of features to train for the model
    global MAXFEAT
    global LAMBDA
    global LEARN_OFFSET

    
    #Path to which we will save the training data--MUST BE SPECIFIED BY THE USER 
    global DATA_PATH
    global DATA_PATH_PLANE


    
    #Specify the download url
    global URL_DATA

    
    #define parameters for the training loop
    global GRAPH_LOSS
    global RESULT_CHECK
    
    global PT_SAMPLE
    global NUM_SAMPLE
    
    #path to initial yaml config file
    global CONFIG_PATH

    #define hyper parameters 
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    CPU = torch.device('cpu')
    
    
    EPOCHS = 50
    BATCH_SIZE = 20
    
    TARGET_PROPORTION = 0.5
    TARGET_PROPORTION_VAL = 0.5
    TARGET_PROPORTION_TEST = 0.5
    TRAIN_SIZE = 300
    VAL_SIZE = 30

    #parameters for mask operation
    MASK_SIZE = 60
    AUGMENT_TIMES_TR = 2
    AUGMENT_TIMES_VAL = 1
    #parameters for the warp-field/grid
    GRID_SIZE = 9
    VOX_SIZE = 32
    #number of features to train for the model 
    MAXFEAT = 64
    
  
    #path to save/load the model 
    MODEL_PATH = './obj/'
    #signifies which model to use 
    MODEL_IDX = 2
    
    #Path to which we will save the training data--MUST BE SPECIFIED BY THE USER 
    DATA_PATH = './data/'
    DATA_PATH_PLANE = os.path.join(DATA_PATH,'train/02691156/')

    #Specify the download url
    URL_DATA = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/SHREC16/train.zip'


    #define parameters for the training loop
    GRAPH_LOSS = True
    RESULT_CHECK = True

    #visualization parameters
    #points to sample for the pointcloud
    PT_SAMPLE = 10000
    #number of samples to get from batch
    NUM_SAMPLE = 10
    
    #path to initial yaml config file
    CONFIG_PATH = 'model_config.yaml'