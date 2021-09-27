#DEFINE global variables 
import torch

def initialize_config():
        #define hyper parameters 
    global USE_CUDA
    global DEVICE
    global EPOCHS 
    global BATCH_SIZE 
    
    global TARGET_PROPORTION
    global TARGET_PROPORTION_VAL
    global TARGET_PROPORTION_TEST 
    global VAL_SIZE
    global N_CLASS
    global TEST_SIZE
    global IMAGE_SIZE
    
    #parameters for mask operation
    global MASK_SIZE
    global MASK_STRIDE 
    global AUGMENT_TIMES_TR
    global AUGMENT_TIMES_VAL
    #parameters for the warp-field/grid
    global GRID_SIZE
    
    #index numbers for data types
    global VASE
    global PLANE
    
    global MODEL_PATH

    #path to save/load the model 
    global MODEL_PATH_VASE
    global MODEL_PATH_PLANE
    
    #Path to which we will save the training data--MUST BE SPECIFIED BY THE USER 
    global FILE_PATH
    global FILE_PATH_VASE
    global FILE_PATH_PLANE
    global FILE_PATH_VASE_DATA
    global FILE_PATH_PLANE_DATA
    
    #Specify the download url
    global URL_VASE
    global URL_PLANE
    
    #define parameters for the training loop
    global TRAIN_MODE
    global GRAPH_LOSS
    global OVERFIT_CHECK
    
    
    #define hyper parameters 
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print(DEVICE)
    EPOCHS = 50
    BATCH_SIZE = 50
    
    TARGET_PROPORTION = 0.5
    TARGET_PROPORTION_VAL = 0.5
    TARGET_PROPORTION_TEST = 0.5
    VAL_SIZE = 20
    N_CLASS = 10
    TEST_SIZE = 10
    IMAGE_SIZE =128
    
    #parameters for mask operation
    MASK_SIZE = 60
    MASK_STRIDE = 20
    AUGMENT_TIMES_TR = 2
    AUGMENT_TIMES_VAL = 2
    #parameters for the warp-field/grid
    GRID_SIZE = 8
    
    #index numbers for data types
    VASE = 0
    PLANE = 1
    
    #path to save/load the model 
    MODEL_PATH = './'
    
    #Path to which we will save the training data--MUST BE SPECIFIED BY THE USER 
    FILE_PATH = './data/'
    FILE_PATH_VASE = './data/vase/'
    FILE_PATH_PLANE = './data/plane/'
    FILE_PATH_VASE_DATA = FILE_PATH_VASE + 'h5files/'
    FILE_PATH_PLANE_DATA = FILE_PATH_PLANE + 'airplane_2d/'

    #Specify the download url
    URL_VASE = 'https://drive.google.com/uc?export=download&id=1Vv-Jz1VpI48MOVgK3Hq6ZYrs2NDP-FQ2'
    URL_PLANE = 'https://drive.google.com/uc?export=download&id=14Pnrp9ahtRbjEehkI-oM8cBMQKERg6GY'
    
    #define parameters for the training loop
    TRAIN_MODE = 0
    GRAPH_LOSS = True
    OVERFIT_CHECK = True