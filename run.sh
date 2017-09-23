#!/bin/bash

efConstruction="500"
M="5"
numMillions="1000"
n="${numMillions}000000"
baseDir="/sata2/dbaranchuk/"
k="1"
nq="10000"
d="128"
l2space="int"

#General
path_data="${baseDir}bigann/base1B_M16/bigann_base_pq.bvecs" 
path_edges="${baseDir}bigann/base1B_M16/sift${numMillions}m_ef${efConstruction}_M5_maxM_32_24_16_9_8_7_6_1.ivecs" 
path_info="${baseDir}bigann/base1B_M16/sift${numMillions}m_ef${efConstruction}_M5_maxM_32_24_16_9_8_7_6_1.bin" 
path_gt="${baseDir}bigann/gnd/idx_${numMillions}M.ivecs" 
path_q="${baseDir}bigann/bigann_query.bvecs"

#PQ
path_codebooks="${baseDir}bigann/base1B_M16/codebooks.fvecs"
path_tables="${baseDir}bigann/base1B_M16/distance_tables.dat"

#./main --help
nohup /home/dbaranchuk/hnsw/main  -path_data ${path_data} -path_edges ${path_edges} -path_info ${path_info} -path_gt ${path_gt} -path_q ${path_q} -path_codebooks ${path_codebooks} -path_tables ${path_tables} -k ${k} -n ${n} -nq ${nq} -M ${M} -efConstruction ${efConstruction} -d ${d} -l2space ${l2space} -1layer 



