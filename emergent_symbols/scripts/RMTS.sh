#!/bin/bash

model_name=$1
device=$2
heads=$3
layers=$4
rules=$5
extras=$6

echo $extras

for r in {1..10}
do
	# m = 0
	python3 ./train_and_eval.py --model_name $model_name --norm_type tasksegmented_contextnorm --lr 5e-4 --task RMTS --m_holdout 0 --epochs 50 --run $r --device $device --train_gen_method subsample --test_gen_method subsample --heads $heads --layers $layers --rules $rules $extras
	# m = 50
	python3 ./train_and_eval.py --model_name $model_name --norm_type tasksegmented_contextnorm --lr 5e-4 --task RMTS --m_holdout 50 --epochs 50 --run $r --device $device --train_gen_method subsample --test_gen_method subsample --heads $heads --layers $layers --rules $rules $extras
	# m = 85
	python3 ./train_and_eval.py --model_name $model_name --norm_type tasksegmented_contextnorm --lr 5e-4 --task RMTS --m_holdout 85 --epochs 50 --run $r --device $device --train_gen_method full_space --test_gen_method subsample --heads $heads --layers $layers --rules $rules $extras
	# m = 95 
	python3 ./train_and_eval.py --model_name $model_name --norm_type tasksegmented_contextnorm --lr 5e-4 --task RMTS --m_holdout 95 --epochs 200 --run $r --device $device --train_gen_method full_space --test_gen_method subsample --heads $heads --layers $layers --rules $rules $extras
done