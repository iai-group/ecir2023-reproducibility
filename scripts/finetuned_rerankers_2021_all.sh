METHOD_LIST=( raw manual automatic )
PREV_LIST=( 0 1 2 3 4 5 all )
for method in "${METHOD_LIST[@]}"; do	
    for n in "${PREV_LIST[@]}"; do	
        CUDA_VISIBLE_DEVICES=0 python -m treccast.main -y 2021 --config-file config/fine_tuning/2021_${method}_prev_${n}_msmarco_treccast_wow.yaml
    done
done