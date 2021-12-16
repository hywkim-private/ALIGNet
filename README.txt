# ALIGNet
A pytorch implementation of ALIGNet, developed by Prof. Rana Hanocka.

0. Navigate to ALIGNet or ALIGNet_3d 

1. Install dependencies
Using virtual environments (pipenv)
-install dependencies by running pip -r requirements.txt

2. Download data
-Define the url to get data in init_config => data => url_data (or use the predefined url in the default init_config)
-run python main.py data -d to download data from url source

3. Make a dataset
-Define all the necessary parameters in init_config => data (read the description for each parameter)
-run python main.py data to augment data, split into validation/train sets, and store it in the defined filepath 

4. Make a new model
-Define all the necessary parameters in init_config => model (read the description for each parameter)
-run python main.py new to make new model, train the model as defined in init_config => train, and save it in the defined filepath

5. Train the model 
-Define all the necessary parameters in init_config => train (read the description for each parameter)
-run python main.py train to train model according to configurations

6. Validate
-Define all the necessary parameters in init_config => valid (read the description for each parameter)
-run python main.py valid to run validation as specified in init_config.yaml
-check [model_path]/outputs/images/ for results (creates .png files with the latest integer numbering as its file name)
