#!/bin/bash

cd ..

input_dir=./data/Stereo-seq/raw
out_dir=./result/benchmark/Stereo-seq
py=benchmark.py

if [ ! -d $out_dir ]; then
  mkdir -p $out_dir
fi

python $py --output_dir $out_dir --adata_file $input_dir/adata.h5ad \
--lr 0.001 --epochs 150 --margin 0 --margin_weight 2 --hvg --norm_target 1e4 --k 15 --temperature 0.3 --exp_out 512 --hidden_dim 256  --rep_dim 128 --proj_dim 128 --gamma 2

