#!/bin/bash

################################
# HNSW construction parameters #
################################

M="16"                # Min number of edges per point
efConstruction="500"  # Max number of candidate vertices in priority queue to observe during construction

###################
# Data parameters #
###################

nb="1000000000"       # Number of base vectors

nt="10000000"         # Number of learn vectors
nsubt="65536"         # Number of learn vectors to train (random subset of the learn set)

nc="1999473"          # Number of centroids for HNSW quantizer

nq="10000"            # Number of queries
ngt="1"               # Number of groundtruth neighbours per query

d="96"                # Vector dimension
code_size="16"        # Code size per vector in bytes

#####################
# Search parameters #
#####################

#######################################
#        Paper configurations         #
# (<nprobe>, <max_codes>, <efSearch>) #
# (   32,       10000,        80    ) #
# (   64,       30000,       100    ) #
# (  128,       50000,       140    ) #
#######################################

k="100"               # Number of the closest vertices to search
nprobe="32"           # Number of probes at query time
max_codes="10000"     # Max number of codes to visit to do a query
efSearch="80"         # Max number of candidate vertices in priority queue to observe during searching

#########
# Paths #
#########

path_data="${PWD}/data/DEEP1B"
path_model="/home/dbaranchuk/data"

path_base="/home/arbabenko/Bigann/deep1B_base.fvecs"
path_learn="/home/arbabenko/Bigann/deep1B_learn.fvecs"
path_gt="$path_data/deep1B_groundtruth.ivecs"
path_q="$path_data/deep1B_queries.fvecs"
path_centroids="/home/dbaranchuk/data/centroids2M.fvecs"

path_precomputed_idxs="/home/dbaranchuk/data/deep1B_precomputed_idxs_${nc}.ivecs"

path_edges="/home/dbaranchuk/data/deep2m_ef${efConstruction}_${nc}_new.ivecs"
path_info="/home/dbaranchuk/data/deep2m_ef${efConstruction}_${nc}_new.bin"

path_pq="${path_model}/pq${code_size}.pq"
path_norm_pq="${path_model}/norm_pq${code_size}.pq"
path_index="${path_model}/ivfhnsw_PQ${code_size}_2M_new.index"

#######
# Run #
#######
/home/dbaranchuk/ivf-hnsw/bin/demo_ivfhnsw_deep1b -M ${M} \
                                                  -efConstruction ${efConstruction} \
                                                  -nb ${nb} \
                                                  -nt ${nt} \
                                                  -nsubt ${nsubt} \
                                                  -nc ${nc} \
                                                  -nq ${nq} \
                                                  -ngt ${ngt} \
                                                  -d ${d} \
                                                  -code_size ${code_size} \
                                                  -k ${k} \
                                                  -nprobe ${nprobe} \
                                                  -max_codes ${max_codes} \
                                                  -efSearch ${efSearch} \
                                                  -path_base ${path_base} \
                                                  -path_learn ${path_learn} \
                                                  -path_gt ${path_gt} \
                                                  -path_q ${path_q} \
                                                  -path_centroids ${path_centroids} \
                                                  -path_precomputed_idx ${path_precomputed_idxs} \
                                                  -path_edges ${path_edges} \
                                                  -path_info ${path_info} \
                                                  -path_pq ${path_pq} \
                                                  -path_norm_pq ${path_norm_pq} \
                                                  -path_index ${path_index}