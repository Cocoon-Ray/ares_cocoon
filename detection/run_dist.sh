CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --master_port 21245 --nproc_per_node 5 run.py \
--cfg configs/patch_demo.py