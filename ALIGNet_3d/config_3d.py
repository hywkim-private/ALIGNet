#DEFINE global variables 
import torch
import os

def initialize_config():
    #general configurations
    #USE_CUDA, DEVICE, CPU  are the only parameters that will be initialized in config_3ds
    global USE_CUDA
    global DEVICE
    global CPU
    #parameters for train config 
    global EPOCHS 
    global STEP
    global ITER
    global MASK_SIZE
    global AUGMENT_TIMES_TR

    #needed for both train and valid
    global MODEL_PATH
    global BATCH_SIZE 
    global AUGMENT_TIMES_VAL

    #parameters for valid 
    global NUM_SAMPLE 
    global VISUALIZE_TYPE
    
    #parameters for data config 
    global TARGET_PROPORTION
    global TARGET_PROPORTION_VAL
    global TRAIN_SIZE
    global VAL_SIZE
    #Path to which we will save the training data--MUST BE SPECIFIED BY THE USER 
    global LOAD_DATA_PATH
    #Specify the download url
    global URL_DATA
    global PT_SAMPLE
    global DATA_TYPE
    
    #signifies which model to use 
    global MODEL_IDX
    #parameters for the warp-field/grid
    global GRID_SIZE
    global VOX_SIZE
    #number of features to train for the model
    global MAXFEAT
    global LAMBDA
    global LEARN_OFFSET
    global DATA_TYPE
    global DATA_PATH

    
   
    
    #Specify the download url
    global URL_DATA

    
    #define parameters for the training loop
    global GRAPH_LOSS
    global RESULT_CHECK
    

    #path to initial yaml config file
    global CONFIG_PATH

    #define hyper parameters 
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    CPU = torch.device('cpu')
    
   