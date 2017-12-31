#!/bin/bash

################################
# HNSW construction parameters #
################################

M="16"                # minimum number of edges per point
efConstruction="500"  # maximum number of observed vertices at once during construction

###################
# Data parameters #
###################

n="1000000000"        # Number of base vectors

nt="10000000"         # Number of learn vectors
nsubt="65536"         # Number of learn vectors to train (random subset of the learn set)

nc="993127"           # Number of centroids for HNSW
nsubc="64"            # Number of subcentroids per group

nq="10000"            # Number of queries
ngt="1000"            # Number of groundtruth neighbours per query

d="128"               # Vector dimension


baseDir="/home/dbaranchuk/"



M_PQ="16"

k="100"

nprobes="32"
max_codes="10000"
efSearch="80"

subdir="new_models/SIFT1B/"

# Paths
path_data="${baseDir}data/bigann/bigann_base.bvecs" 
path_edges="${baseDir}${subdir}centroids${nc}_ef${efConstruction}.ivecs"
path_info="${baseDir}${subdir}centroids${nc}_ef${efConstruction}.bin" 

path_gt="${baseDir}data/bigann/gnd/idx_1000M.ivecs" 
path_q="${baseDir}data/bigann/bigann_query.bvecs"

path_pq="${baseDir}${subdir}pq${M_PQ}_nsubc${nsubc}.pq"
path_norm_pq="${baseDir}${subdir}norm_pq${M_PQ}_nsubc${nsubc}.pq"
path_precomputed_idxs="${baseDir}staff-ivf-hnsw/sift1B_precomputed_idxs_${nc}.ivecs"
path_index="${baseDir}${subdir}hybrid${nc}M_PQ${M_PQ}_nsubc${nsubc}.index"
path_learn="${baseDir}data/bigann/bigann_learn.bvecs"
path_centroids="${baseDir}data/sift1B_centroids${nc}M.fvecs"
 
path_groups="${baseDir}data/groups/sift1B_groups.dat";
path_idxs="${baseDir}data/groups/sift1B_idxs.ivecs"

#M_PQ="16"
#path_edges="${baseDir}${subdir}centroids${nc}_ef${efConstruction}_nsubc${nsubc}_PQ${M_PQ}.ivecs"
#path_info="${baseDir}${subdir}centroids${nc}_ef${efConstruction}_nsubc${nsubc}_PQ${M_PQ}.bin"
#./main --help
/home/dbaranchuk/ivf-hnsw/bin/demo_ivfhnsw_grouping_sift1b -path_centroids ${path_centroids} -path_learn ${path_learn} -path_index ${path_index} -path_precomputed_idx ${path_precomputed_idxs} -path_pq ${path_pq} -path_norm_pq ${path_norm_pq} -path_data ${path_data} -path_edges ${path_edges} -path_info ${path_info} -path_gt ${path_gt} -path_q ${path_q} -path_groups ${path_groups} -path_idxs ${path_idxs} -k ${k} -n ${n} -nq ${nq} -nc ${nc} -M ${M} -M_PQ ${M_PQ} -efConstruction ${efConstruction} -efSearch ${efSearch} -d ${d} -gt_d ${gt_d} -nprobes ${nprobes} -max_codes ${max_codes} -nsubc ${nsubc} -nt ${nt} -nsubt ${nsubt}
