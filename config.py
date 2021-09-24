#DEFINE MACROS
#define hyper parameters 
global USE_CUDA = torch.cuda.is_available()
global DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
global EPOCHS = 50
global BATCH_SIZE = 50

global TARGET_PROPORTION = 0.5
global TARGET_PROPORTION_VAL = 0.5
global TARGET_PROPORTION_TEST = 0.5
global VAL_SIZE = 20
global N_CLASS = 10
global TEST_SIZE = 10
global IMAGE_SIZE =128

#parameters for mask operation
global MASK_SIZE = 60
global MASK_STRIDE = 20
global AUGMENT_TIMES_TR = 2
global AUGMENT_TIMES_VAL = 2
#parameters for the warp-field/grid
global GRID_SIZE = 8

#path to save/load the model 
global MODEL_PATH_VASE =
global MODEL_PATH_PLANE =

#Path to which we will save the training data--MUST BE SPECIFIED BY THE USER 
global FILE_PATH = 'gdrive/MyDrive/ALIGNet'
global FILE_PATH_VASE = 'gdrive/MyDrive/ALIGNet/h5files'
global FILE_PATH_PLANE =  'gdrive/MyDrive/ALIGNet/planes/airplane_2d'

#Specify the download url
global URL_VASE = 'https://drive.google.com/uc?export=download&id=1Vv-Jz1VpI48MOVgK3Hq6ZYrs2NDP-FQ2'
global URL_PLANE = 'https://drive.google.com/uc?export=download&id=14Pnrp9ahtRbjEehkI-oM8cBMQKERg6GY'