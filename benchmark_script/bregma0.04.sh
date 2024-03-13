#!/bin/bash

cd ..

input_dir=./data/Bregma0.04/raw
out_dir=./result/benchmark/Bregma0.04
py=benchmark.py

if [ ! -d $out_dir ]; then
  mkdir -p $out_dir
fi

python $py --output_dir $out_dir --adata_file $input_dir/adata.h5ad \
--lr 0.001 --epochs 300 --margin -0.2 --margin_weight 5 --a_k 10 --k 50 --temperature 0.3 --exp_out 256 --hidden_dim 256 --rep_dim 64 --proj_dim 64 --gamma 1 --maskfeat_rate_learner 0.3 --maskfeat_rate_anchor 0.3

