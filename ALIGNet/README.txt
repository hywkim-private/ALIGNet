# ALIGNet
A pytorch implementation of ALIGNet, developed by Prof. Rana Hanocka.

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

7. Using distributed data parallel
-If you are running on multiple gpus, then you can train the model in parallel
-set the num_gpu parameter in init_config to however many gpus you need to use,
and run the model.

*CAUTION
-The distributed data parallel model, if run on a light model with small batch data size,
will be slower than running on a single gpu, due to the significant overhead caused by
copying each batch of tensor to multiple devices.
-Only use the parallel mode if you are either using a significantly heavy network, a very 
large datasize, very large batch inputs, or if your processor runs out of memory during execution
-Generally, the data parallel mode is not really necessary for the 2d model since it does well
on a single gpu