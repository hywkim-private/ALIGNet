#install all the dependencies
#IMPORTANT: This is a skeleton code for install.sh, which merely enlists all the required libraries with the pip-install command. 
#Python Requirements: 3.6 or higher

#install numpy
echo "installing numpy"
pip install numpy
#install torch
echo "installing pytoch"
pip install pytorch
#install torch vision
echo "installing torchvision"
pip install torchvision
#install hfpy
echo "installing hfpy"
pip install hfpy
#install matplotlib
echo "installing matplotlib"
pip install matplotlib
pip install skimage
pip install imgaug 
pip install cv2
pip install wget
pip install pathlibs
#export python path
export PYTHONPATH=./utils
