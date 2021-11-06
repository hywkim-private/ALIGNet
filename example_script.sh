export PYTHONPATH=./utils
python main.py new -ty vase -n model_vase -i 50
python main.py valid -ty vase -n model_vase -v true

#check for outputs in directory ./model_vase/outputs/images and ./model_vase/outputs/loss_graphs