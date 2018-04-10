
### darknet
cd darknet
change Makefile for GPU, CUDNN, OPENCV
make
wget https://pjreddie.com/media/files/yolov3.weights
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
# how to get video captions as json?



### pix2pix



git clone https://github.com/pytorch/vision
cd vision
python setup.py install
cd ..
rm -rf vision


pip install visdom
pip install dominate


cd pytorch-CycleGAN-and-pix2pix
bash ./datasets/download_pix2pix_dataset.sh facades
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0



### pix2pix-tensorflow
python tools/download-dataset.py facades
python pix2pix.py  --mode train  --output_dir facades_train  --max_epochs 10 --input_dir facades/train --which_direction BtoA
python pix2pix.py  --mode test --output_dir facades_test --input_dir facades/val --checkpoint facades_train