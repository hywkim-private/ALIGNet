#a module that parses yaml config file and stores model-specific configurations to each model
import yaml
import config as cfg


#load_config
#loads and returns the configuration file in the form of dictionary
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.full_load(file)
    return config

#render the parallel data loading capabilities    
def render_settings(config):
    #if num gpu is None, we will not specify 
    if config['setting']['num_gpu'] is None:
        return
    cfg.NUM_GPU = config['setting']['num_gpu']

#render the model-specific configurations in the config_3d.py file
def render_model_config(config):
    cfg.GRID_SIZE = config['model']['grid_size']
    cfg.MAXFEAT = config['model']['maxfeat']
    cfg.MODEL_IDX = config['model']['model_idx']
    cfg.LEARN_OFFSET = config['model']['learn_offset']
    cfg.LAMBDA = config['model']['lambda']
    cfg.DATA_PATH = config['model']['data_path']
    cfg.MODEL_PATH = config['model']['model_path']
    cfg.IMAGE_SIZE = config['model']['image_size']

def load_model_config(config):
    cfg.GRID_SIZE = config['grid_size']
    cfg.MAXFEAT = config['maxfeat']
    cfg.MODEL_IDX = config['model_idx']
    cfg.LEARN_OFFSET = config['learn_offset']
    cfg.DATA_IDX = config['data_idx']
    cfg.LAMBDA = config['lambda']
    cfg.DATA_TYPE = config['data_type']
    cfg.DATA_PATH = config['data_path']
    cfg.IMAGE_SIZE = config['image_size']

    
def write_model_config(config, model_path):
    try:
        with open(model_path+'model_cfg.yaml', 'w') as file:
            cfg_dict = {'model': config['model']}
            yaml.dump(cfg_dict, file, default_flow_style=False)
    except FileNotFoundError:
        print(f"file {model_path} not found")

def render_train_config(config):
    cfg.EPOCHS = config['train']['epochs']
    cfg.BATCH_SIZE = config['train']['batch_size']
    cfg.STEP = config['train']['step']
    cfg.ITER = config['train']['iter']
    cfg.MASK_SIZE = tuple(config['train']['mask_size'])
    cfg.AUGMENT_TIMES_TR = config['train']['augment_times_tr']
    cfg.AUGMENT_TIMES_VAL = config['train']['augment_times_val']
    cfg.MODEL_PATH = config['train']['model_path']
    cfg.RESULT_CHECK = config['train']['result_check']
    cfg.GRAPH_LOSS = config['train']['graph_loss']
    cfg.TRANSFORM_NO = config['train']['transform_no']

    
def render_data_config(config):
    cfg.TARGET_PROPORTION = config['data']['target_proportion']
    cfg.TARGET_PROPORTION_VAL = config['data']['target_proportion_val']
    cfg.TRAIN_SIZE = config['data']['train_datasize']
    cfg.VAL_SIZE = config['data']['val_datasize']
    cfg.LOAD_DATA_PATH = config['data']['load_data_path']
    cfg.DATA_PATH = config['data']['data_path']
    cfg.URL_DATA = config['data']['url_data']
    cfg.DATA_TYPE = config['data']['data_type']
    cfg.GRID_SIZE = config['data']['grid_size']
    
def write_data_config(config, data_path):
    try: 
        with open(data_path + 'data_cfg.yaml', 'w') as file:
            cfg_dict = {'data': config['data']}
            yaml.dump(cfg_dict, file, default_flow_style=False)
    except FileNotFoundError:
        print(f"file {data_path} not found")
    
def render_valid_config(config):
    cfg.MODEL_PATH = config['valid']['model_path']
    cfg.AUGMENT_TIMES_VAL = config['valid']['augment_times_val']
    cfg.VAL_SAMPLE = config['valid']['num_sample']
    cfg.BATCH_SIZE = config['valid']['batch_size']
    cfg.MASK_SIZE = tuple(config['valid']['mask_size'])
    cfg.TRANSFORM_NO = config['valid']['transform_no']
    cfg.CHECKERBOARD = config['valid']['checkerboard']