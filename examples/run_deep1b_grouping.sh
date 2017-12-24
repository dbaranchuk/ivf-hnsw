#!/bin/bash

efConstruction="500"
M="16"
numMillions="1000"
n="${numMillions}000000"

homeBabenko="/home/arbabenko/"
homeBaranchuk="/home/dbaranchuk/"

k="100"
nq="10000"

d="96"
gt_d="1"

M_PQ="16"
nprobes="32" # 8M: 128 260 450 600
max_codes="10000"

nsubcentroids="64"

nt="10000000"
nsubt="1000000"

nc="1"
ncentroids="999973"
efSearch="80"

subdir="final_models/DEEP1B/"

#General
path_data="${homeBabenko}Bigann/deep1B_base.fvecs" 
path_learn="${homeBabenko}Bigann/deep1B_learn.fvecs"

path_edges="${homeBaranchuk}${subdir}centroids${ncentroids}_ef${efConstruction}_nsubc${nsubcentroids}.ivecs" 
path_info="${homeBaranchuk}${subdir}centroids${ncentroids}_ef${efConstruction}_nsubc${nsubcentroids}.bin" 

path_gt="${homeBabenko}Bigann/deep1B_groundtruth.ivecs" 
path_q="${homeBabenko}Bigann/deep1B_queries.fvecs"

path_pq="${homeBaranchuk}${subdir}pq${M_PQ}_nsubc${nsubcentroids}.pq"
path_norm_pq="${homeBaranchuk}${subdir}norm_pq${M_PQ}_nsubc${nsubcentroids}.pq"
path_precomputed_idxs="${homeBaranchuk}staff-ivf-hnsw/deep1B_precomputed_idxs_${ncentroids}.ivecs"
path_index="${homeBaranchuk}${subdir}hybrid${nc}M_PQ${M_PQ}_nsubc${nsubcentroids}.index"
path_centroids="${homeBaranchuk}data/centroids${nc}M.fvecs"

path_groups="${homeBaranchuk}data/groups/groups${ncentroids}.dat"
path_idxs="${homeBaranchuk}data/groups/idxs${ncentroids}.ivecs"

#./main --help
/home/dbaranchuk/ivf-hnsw/bin/demo_ivfhnsw_grouping_deep1b -path_centroids ${path_centroids} -path_learn ${path_learn} -path_index ${path_index} -path_precomputed_idx ${path_precomputed_idxs} -path_pq ${path_pq} -path_norm_pq ${path_norm_pq} -path_data ${path_data} -path_edges ${path_edges} -path_info ${path_info} -path_gt ${path_gt} -path_q ${path_q} -path_groups ${path_groups} -path_idxs ${path_idxs} -k ${k} -n ${n} -nq ${nq} -nc ${ncentroids} -M ${M} -M_PQ ${M_PQ} -efConstruction ${efConstruction} -efSearch ${efSearch} -d ${d} -gt_d ${gt_d} -nprobes ${nprobes} -max_codes ${max_codes} -nsubc ${nsubcentroids} -nt ${nt} -nsubt ${nsubt}