cd ..

input_dir=./data/osmFISH/raw
out_dir=./result/benchmark/osmFISH
py=benchmark.py

if [ ! -d $out_dir ]; then
  mkdir -p $out_dir
fi

python $py --output_dir $out_dir --adata_file $input_dir/adata.h5ad \
--lr 0.0005 --epochs 200 --margin 0 --margin_weight 1 --k 30 --temperature 0.3 --exp_out 64 --hidden_dim 256  --rep_dim 64 --proj_dim 64 --gamma 3 --maskfeat_rate_learner 0.0 --maskfeat_rate_anchor 0.0

