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
    global N_CLASS
    global IMAGE_SIZE
    
    #parameters for mask operation
    global MASK_SIZE
    global MASK_STRIDE 
    global AUGMENT_TIMES_TR
    global AUGMENT_TIMES_VAL
    #parameters for the warp-field/grid
    global GRID_SIZE
    global VOX_SIZE
    #index numbers for data types
    global VASE
    global PLANE
    
    global MODEL_PATH

    #path to save/load the model 
    global MODEL_PATH_VASE
    global MODEL_PATH_PLANE
    
    #Path to which we will save the training data--MUST BE SPECIFIED BY THE USER 
    global DATA_PATH
    global DATA_PATH_PLANE


    
    #Specify the download url
    global URL_DATA

    
    #define parameters for the training loop
    global TRAIN_MODE
    global GRAPH_LOSS
    global RESULT_CHECK
    
    global PT_SAMPLE
    global NUM_SAMPLE
    

    #define hyper parameters 
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    CPU = torch.device('cpu')
    EPOCHS = 50
    BATCH_SIZE = 50
    
    TARGET_PROPORTION = 0.5
    TARGET_PROPORTION_VAL = 0.5
    TARGET_PROPORTION_TEST = 0.5
    TRAIN_SIZE = 100
    VAL_SIZE = 20
    N_CLASS = 10
    TEST_SIZE = 10
    IMAGE_SIZE = 32
    
    #parameters for mask operation
    MASK_SIZE = 60
    MASK_STRIDE = 20
    AUGMENT_TIMES_TR = 2
    AUGMENT_TIMES_VAL = 2
    #parameters for the warp-field/grid
    GRID_SIZE = 7
    VOX_SIZE = 32
    
    #index numbers for data types
    VASE = 0
    PLANE = 1
    
    #path to save/load the model 
    MODEL_PATH = './obj/'
    
    #Path to which we will save the training data--MUST BE SPECIFIED BY THE USER 
    DATA_PATH = './data/'
    DATA_PATH_PLANE = os.path.join(DATA_PATH,'train/02691156/')

    #Specify the download url
    URL_DATA = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/SHREC16/train.zip'


    #define parameters for the training loop
    TRAIN_MODE = 0
    GRAPH_LOSS = True
    RESULT_CHECK = True

    #visualization parameters
    #points to sample for the pointcloud
    PT_SAMPLE = 10000
    #number of samples to get from batch
    NUM_SAMPLE = 10
    