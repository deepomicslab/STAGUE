#!/bin/bash

cd ..

input_dir=./data/DLPFC/raw
py=benchmark.py

num=151507
out_dir=./result/benchmark/DLPFC${num}

if [ ! -d $out_dir ]; then
  mkdir -p $out_dir
fi

python $py --output_dir $out_dir --adata_file $input_dir/adata_${num}.h5ad \
--lr 0.001 --refine 50 --hvg --margin 0 --margin_weight 0.5 --filter_cell 50 --epochs 150 --a_k 6 --k 20 --temperature 0.2 --exp_out 512 --hidden_dim 256  --rep_dim 64 --proj_dim 64 --gamma 3
