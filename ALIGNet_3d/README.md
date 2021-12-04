# ALIGNet
A pytorch implementation of ALIGNet, developed by Prof. Rana Hanocka.

1. Install dependencies
run bash install.sh

2. Download data
-Define the url to get data in init_config => data => url_data (or use the predefined url in the default init_config)
-run python main.py data -d

3. Make a dataset
-Define all the necessary parameters in init_config => data (read the description for each parameter)
-run python main.py data

4. Make a new model
-Define all the necessary parameters in init_config => model (read the description for each parameter)
-run python main.py new

5. Train the model 
-Define all the necessary parameters in init_config => train (read the description for each parameter)
-run python main.py train

6. Validate
-Define all the necessary parameters in init_config => valid (read the description for each parameter)
-run python main.py valid
-check [model_path]/outputs/images/ for results
