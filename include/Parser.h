#ifndef IVF_HNSW_LIB_PARSER_H
#define IVF_HNSW_LIB_PARSER_H

//==============
// Parser Class
//==============
struct Parser
{
    const char *cmd;        ///< main command - argv[0]

    //=================
    // HNSW parameters
    //=================
    size_t M;               ///< Min number of edges per point
    size_t efConstruction;  ///< Max number of candidate vertices in priority queue to observe during construction

    //=================
    // Data parameters
    //=================
    size_t nb;             ///< Number of base vectors
    size_t nt;             ///< Number of learn vectors
    size_t nsubt;          ///< Number of learn vectors to train (random subset of the learn set)
    size_t nc;             ///< Number of centroids for HNSW quantizer
    size_t nsubc;          ///< Number of subcentroids per group
    size_t nq;             ///< Number of queries
    size_t ngt;            ///< Number of groundtruth neighbours per query
    size_t d;              ///< Vector dimension

    //=================
    // PQ parameters
    //=================
    size_t code_size;      ///< Code size per vector in bytes
    bool do_opq;           ///< Turn on/off OPQ fine encoding

    //===================
    // Search parameters
    //===================
    size_t k;              ///< Number of the closest vertices to search
    size_t nprobe;         ///< Number of probes at query time
    size_t max_codes;      ///< Max number of codes to visit to do a query
    size_t efSearch;       ///< Max number of candidate vertices in priority queue to observe during searching
    bool do_pruning;       ///< Turn on/off pruning in the grouping scheme

    //=======
    // Paths
    //=======
    const char *path_base;             ///< Path to a base set
    const char *path_learn;            ///< Path to a learn set
    const char *path_q;                ///< Path to queries
    const char *path_gt;               ///< Path to groundtruth
    const char *path_centroids;        ///< Path to coarse centroids

    const char *path_precomputed_idxs; ///< Path to coarse centroid indices for base points

    const char *path_info;             ///< Path to parameters of HNSW graph
    const char *path_edges;            ///< Path to edges of HNSW graph

    const char *path_pq;               ///< Path to the product quantizer for residuals
    const char *path_opq_matrix;       ///< Path to OPQ rotation matrix for OPQ fine encoding
    const char *path_norm_pq;          ///< Path to the product quantizer for norms of reconstructed base points
    const char *path_index;            ///< Path to the constructed index

    Parser(int argc, char **argv);
    void usage();
};

#endif //IVF_HNSW_LIB_PARSER_H
