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
code_size="16"        # Code size per vector in bytes

#####################
# Search parameters #
#####################

k="100"
nprobe="32"           # Number of probes at query time
max_codes="10000"     # Max number of codes to visit to do a query
efSearch="80"

#########
# Paths #
#########

path_data="$PWD/data/SIFT1B"
path_model="$PWD/models/SIFT1B"

path_base="${path_data}/bigann_base.bvecs"
path_learn="${path_data}/bigann_learn.bvecs"
path_gt="${path_data}/gnd/idx_1000M.ivecs"
path_q="${path_data}/bigann_query.bvecs"
path_centroids="${path_data}/centroids.fvecs"

path_precomputed_idxs="${path_data}/precomputed_idxs_${nc}.ivecs"

path_groups="${path_data}/groups/groups.dat";
path_idxs="${path_data}/groups/idxs.ivecs"

path_edges="${path_model}/hnsw_M${M}_ef${efConstruction}.ivecs"
path_info="${path_model}/hnsw_M${M}_ef${efConstruction}.bin"

path_pq="${path_model}/pq${code_size}_nsubc${nsubc}.pq"
path_norm_pq="${path_model}/norm_pq${code_size}_nsubc${nsubc}.pq"
path_index="${path_model}/ivfhnsw_PQ${code_size}_nsubc${nsubc}.index"

#######
# Run #
#######
/home/dbaranchuk/ivf-hnsw/bin/demo_ivfhnsw_grouping_sift1b -path_centroids ${path_centroids} -path_learn ${path_learn} -path_index ${path_index} -path_precomputed_idx ${path_precomputed_idxs} -path_pq ${path_pq} -path_norm_pq ${path_norm_pq} -path_data ${path_base} -path_edges ${path_edges} -path_info ${path_info} -path_gt ${path_gt} -path_q ${path_q} -path_groups ${path_groups} -path_idxs ${path_idxs} -k ${k} -n ${n} -nq ${nq} -nc ${nc} -M ${M} -M_PQ ${code_size} -efConstruction ${efConstruction} -efSearch ${efSearch} -d ${d} -gt_d ${ngt} -nprobes ${nprobe} -max_codes ${max_codes} -nsubc ${nsubc} -nt ${nt} -nsubt ${nsubt}
