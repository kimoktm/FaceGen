# !/bin/bash
echo "#########################"
echo "Feedback dataset 20 k    "
echo "#########################"


# # echo 'Generating Delete set:'
# python test_render.py --output '/home/karim/Documents/Development/FacialCapture/face3dMM/examples/results/faces/del' \
# 						          --samples 100


echo 'Generating training set:'
python test_render.py --output '/home/karim/Documents/Development/FacialCapture/face3dMM/examples/results/faces/funcspace_4k/train' \
						          --samples 4000


echo 'Generating validation set:'
python test_render.py --output '/home/karim/Documents/Development/FacialCapture/face3dMM/examples/results/faces/funcspace_4k/validation' \
						          --samples 1984 --background '/home/karim/Documents/Data/Random_small'



# echo 'Generating training set:'
# python test_render.py --output '/home/karim/Documents/Development/FacialCapture/face3dMM/examples/results/faces/funcspace_20k/train' \
# 						          --samples 20000


# echo 'Generating validation set:'
# python test_render.py --output '/home/karim/Documents/Development/FacialCapture/face3dMM/examples/results/faces/funcspace_20k/validation' \
# 						          --samples 1984 --background '/home/karim/Documents/Data/Random_small'



# echo 'Generating training set:'
# python test_render.py --output '/home/karim/Documents/Development/FacialCapture/face3dMM/examples/results/faces/funcspace_320/train' \
# 						          --samples 320


# echo 'Generating validation set:'
# python test_render.py --output '/home/karim/Documents/Development/FacialCapture/face3dMM/examples/results/faces/funcspace_320/validation' \
# 						          --samples 32 --background '/home/karim/Documents/Data/Random_small'



