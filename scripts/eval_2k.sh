METHOD_LIST=( raw manual automatic )
#METHOD_LIST=( automatic )
YEARS=( 2020 2021 )
PREV_LIST=( 0 )
# PREV_LIST=( 0 1 2 3 4 5 all )
    for year in "${YEARS[@]}"; do	
for method in "${METHOD_LIST[@]}"; do	
    for n in "${PREV_LIST[@]}"; do	
echo ${year}_${method}_prev_${n}
tools/trec_eval/trec_eval -m all_trec data/qrels/${year}.txt \
data/runs/${year}/cast_${year}_${method}_2k.trec > ${year}_${method}_2k_results.txt 
grep -E 'recip_rank|recall_1000|ndcg_cut_5|map' ${year}_${method}_2k_results.txt 
done
done
done
# echo 2021_manual_prev_all
# trec_eval/trec_eval -m all_trec data/qrels/2021.txt \
# data/runs/2021/cast_2021_manual_prev_all.trec > 2021_manual_prev_all_results.txt 
#  grep -E 'recip_rank|recall_1000' 2021_raw_prev_all_results.txt 

# echo 2021_automatic_prev_all
# trec_eval/trec_eval -m all_trec data/qrels/2021.txt \
# data/runs/2021/cast_2021_automatic_prev_all.trec > 2021_automatic_prev_all_results.txt 
# grep -E 'recip_rank|recall_1000' 2021_automatic_prev_all_results

