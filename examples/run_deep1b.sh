#!/bin/bash

efConstruction="500"
M="16"
numMillions="1000"
n="${numMillions}000000"

homeBabenko="/home/arbabenko/"
homeBaranchuk="/home/dbaranchuk/"

k="100"
nq="10000"

d="16"
gt_d="1"

M_PQ="96"
nprobes="32" # 8M: 128 260 450 600
max_codes="10000"

nt="1000000"
nsubt="131072"

nc="1"
ncentroids="999973"
efSearch="500"

subdir="new_models/DEEP1B/"

#Paths
path_data="${homeBabenko}Bigann/deep1B_base.fvecs" 
path_learn="${homeBabenko}Bigann/deep1B_learn.fvecs"

path_edges="${homeBaranchuk}${subdir}centroids${ncentroids}_ef${efConstruction}.ivecs" 
path_info="${homeBaranchuk}${subdir}centroids${ncentroids}_ef${efConstruction}.bin" 

path_gt="${homeBabenko}Bigann/deep1B_groundtruth.ivecs" 
path_q="${homeBabenko}Bigann/deep1B_queries.fvecs"

path_pq="${homeBaranchuk}${subdir}pq${M_PQ}.pq"
path_norm_pq="${homeBaranchuk}${subdir}norm_pq${M_PQ}.pq"
path_precomputed_idxs="${homeBaranchuk}staff-ivf-hnsw/deep1B_precomputed_idxs_${ncentroids}.ivecs"
path_index="${homeBaranchuk}${subdir}hybrid${nc}M_PQ${M_PQ}.index"
path_centroids="${homeBaranchuk}data/centroids${nc}M.fvecs"

#./main --help
/home/dbaranchuk/ivf-hnsw/bin/demo_ivfhnsw_deep1b -path_centroids ${path_centroids} -path_learn ${path_learn} -path_index ${path_index} -path_precomputed_idx ${path_precomputed_idxs} -path_pq ${path_pq} -path_norm_pq ${path_norm_pq} -path_data ${path_data} -path_edges ${path_edges} -path_info ${path_info} -path_gt ${path_gt} -path_q ${path_q} -k ${k} -n ${n} -nq ${nq} -nc ${ncentroids} -M ${M} -M_PQ ${M_PQ} -efConstruction ${efConstruction} -efSearch ${efSearch} -d ${d} -gt_d ${gt_d} -nprobes ${nprobes} -max_codes ${max_codes} -nt ${nt} -nsubt ${nsubt}

