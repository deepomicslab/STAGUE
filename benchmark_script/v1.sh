#!/bin/bash

cd ..

input_dir=./data/V1/raw
out_dir=./result/benchmark/V1
py=benchmark.py

if [ ! -d $out_dir ]; then
  mkdir -p $out_dir
fi

python $py --output_dir $out_dir --adata_file $input_dir/adata.h5ad \
--lr 0.0005 --epochs 200 --margin 0.5 --margin_weight 2 --k 15 --temperature 0.3 --exp_out 512 --hidden_dim 256 --rep_dim 64 --proj_dim 64 --gamma 2

