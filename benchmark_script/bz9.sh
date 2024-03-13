#!/bin/bash

cd ..

input_dir=./data/BZ9/raw
out_dir=./result/benchmark/BZ9
py=benchmark.py

if [ ! -d $out_dir ]; then
  mkdir -p $out_dir
fi

python $py --output_dir $out_dir --adata_file $input_dir/adata.h5ad \
--lr 0.0005 --epochs 200 --margin 1.5 --margin_weight 5 --k 15 --temperature 0.3 --exp_out 512 --hidden_dim 512 --rep_dim 64 --proj_dim 64 --gamma 2

