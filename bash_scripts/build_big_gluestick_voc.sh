#!/bin/bash

L=5
k=10

# dir=/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick/descriptors/
# tmp_index_name=$dir"tmp_index.txt"
# ls $dir*[0-9].npy > $tmp_index_name
tmp_index_name=/home/colin/Research/ir/DBoW2/vocs/day_night_simple_exp/day_night_simple_exp_paths.txt

out_name=gluestick_day_night_voc_kri_carter_garrage_L$L"_k"$k
out_file=/home/colin/Research/ir/DBoW2/vocs/day_night_simple_exp/$out_name
# out_file=$(dirname $dir)/$out_name
echo $out_file
#build voc
/home/colin/Research/ir/DBoW2/build/build_ir_voc $tmp_index_name $out_file $k $L

L=7
k=10

# dir=/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick/descriptors/
# tmp_index_name=$dir"tmp_index.txt"
# ls $dir*[0-9].npy > $tmp_index_name
tmp_index_name=/home/colin/Research/ir/DBoW2/vocs/day_night_simple_exp/day_night_simple_exp_paths.txt

out_name=gluestick_day_night_voc_kri_carter_garrage_L$L"_k"$k
out_file=/home/colin/Research/ir/DBoW2/vocs/day_night_simple_exp/$out_name
# out_file=$(dirname $dir)/$out_name
echo $out_file
#build voc
/home/colin/Research/ir/DBoW2/build/build_ir_voc $tmp_index_name $out_file $k $L
