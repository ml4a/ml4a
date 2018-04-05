
BEGAN

python main.py --dataset=portrait_64_64 --data_dir=../../datasets/ --load_path=logs/portrait_64_64_0404_100535 --use_gpu=True --is_train=False


DCGAN

python main.py --dataset landscapes_128_128 --input_height=128 --output_height=128 --epoch 100 --train




art-DCGAN

net=checkpoints/experiment4_800_net_G.t7 name=gen8 imsize=1 display=0 th generate.lua 


