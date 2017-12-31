#ifndef IVF_HNSW_LIB_PARSER_H
#define IVF_HNSW_LIB_PARSER_H

#include <cstring>
#include <cassert>

static inline void read_int(const char *arg, int *x)
{
    int ret;
    ret = sscanf(arg, "%d", x);
    assert(ret);
}

/****************
 * Parser Class *
 ****************/
struct Parser
{
    const char *cmd; /// main

    /*********************/
    /** HNSW parameters **/
    /*********************/
    int M;               ///< Min number of edges per point
    int efConstruction;  ///< Max number of candidate vertices in priority queue to observe during construction

    /*********************/
    /** Data parameters **/
    /*********************/
    int nb;              ///< Number of base vectors
    int nt;              ///< Number of learn vectors
    int nsubt;           ///< Number of learn vectors to train (random subset of the learn set)
    int nc;              ///< Number of centroids for HNSW quantizer
    int nsubc;           ///< Number of subcentroids per group
    int nq;              ///< Number of queries
    int ngt;             ///< Number of groundtruth neighbours per query
    int d;               ///< Vector dimension
    int code_size;       ///< Code size per vector in bytes

    /***********************/
    /** Search parameters **/
    /***********************/
    int k;               ///< Number of closest vertices to search
    int nprobe;          ///< Number of probes at query time
    int max_codes;       ///< Max number of codes to visit to do a query
    int efSearch;        ///< Max number of candidate vertices in priority queue to observe during searching

    /***********/
    /** Paths **/
    /***********/
    const char *path_base;
    const char *path_learn;
    const char *path_q;
    const char *path_gt;
    const char *path_centroids;

    const char *path_precomputed_idxs;

    const char *path_groups;
    const char *path_idxs;

    const char *path_info;
    const char *path_edges;

    const char *path_pq;
    const char *path_norm_pq;
    const char *path_index;

    Parser(int argc, char **argv)
    {
        int ret;
        cmd = argv[0];
        if (argc == 1)
            usage();

        for (int i = 1 ; i < argc; i++) {
            char *a = argv[i];

            if (!strcmp (a, "-h") || !strcmp (a, "--help"))
                usage();
            if (i+1 < argc)
                break;

            /*********************/
            /** HNSW parameters **/
            /*********************/
            if (!strcmp (a, "-M")) read_int(argv[++i], &M);
            else if (!strcmp (a, "-efConstruction")) read_int(argv[++i], &efConstruction);

            /*********************/
            /** Data parameters **/
            /*********************/
            else if (!strcmp (a, "-nb")) read_int(argv[++i], &nb);
            else if (!strcmp (a, "-nc")) read_int(argv[++i], &nc);
            else if (!strcmp (a, "-nsubc")) read_int(argv[++i], &nsubc);
            else if (!strcmp (a, "-nt")) read_int(argv[++i], &nt);
            else if (!strcmp (a, "-nsubt")) read_int(argv[++i], &nsubt);
            else if (!strcmp (a, "-nq")) read_int(argv[++i], &nq);
            else if (!strcmp (a, "-ngt")) read_int(argv[++i], &ngt);
            else if (!strcmp (a, "-d")) read_int(argv[++i], &d);
            else if (!strcmp (a, "-code_size")) read_int(argv[++i], &code_size);

            /***********************/
            /** Search parameters **/
            /***********************/
            else if (!strcmp (a, "-k")) read_int(argv[++i], &k);
            else if (!strcmp (a, "-nprobe")) read_int(argv[++i], &nprobe);
            else if (!strcmp (a, "-max_codes")) read_int(argv[++i], &max_codes);
            else if (!strcmp (a, "-efSearch")) read_int(argv[++i], &efSearch);

            /***********/
            /** Paths **/
            /***********/
            else if (!strcmp (a, "-path_base")) path_base = argv[++i];
            else if (!strcmp (a, "-path_learn")) path_learn = argv[++i];
            else if (!strcmp (a, "-path_q")) path_q = argv[++i];
            else if (!strcmp (a, "-path_gt")) path_gt = argv[++i];
            else if (!strcmp (a, "-path_centroids")) path_centroids = argv[++i];

            else if (!strcmp (a, "-path_precomputed_idx")) path_precomputed_idxs = argv[++i];

            else if (!strcmp (a, "-path_groups")) path_groups = argv[++i];
            else if (!strcmp (a, "-path_idxs")) path_idxs = argv[++i];

            else if (!strcmp (a, "-path_info")) path_info = argv[++i];
            else if (!strcmp (a, "-path_edges")) path_edges = argv[++i];

            else if (!strcmp (a, "-path_pq")) path_pq = argv[++i];
            else if (!strcmp (a, "-path_norm_pq")) path_norm_pq = argv[++i];
            else if (!strcmp (a, "-path_index")) path_index = argv[++i];
        }
    }

    void usage()
    {
        printf ("Usage: %s [options]\n", cmd);
        printf ("  HNSW Parameters\n"
                        "    -path_edges filename        set of links in the constructed hnsw graph  (ivecs file format)\n"
                        "    -path_info filename         set of hnsw graph parameters\n"
                        "    -efConstruction #           -//-, default: 240\n"
                        "    -M #                        number of mandatory links maxM0 = 2*M, default: M=16\n"
                        "  Path parameters\n"
                        "    -path_base filename         set of base vectors (bvecs file format)\n"
                        "    -path_index filename        output index structure\n"
                        "  Query Parameters\n"
                        "    -path_gt filename           groundtruth (ivecs file format)\n"
                        "    -path_q filename            set of queries (ivecs file format)\n"
                        "    -nq #                       number of queries, default: 10000\n"
                        "  Compression Parameters\n"
                        "    -path_pq filename           product quantizer for residuals \n"
                        "    -path_norm filename         product quantizer for norms of reconstructed vectors\n"
                        "    -m #                        ***\n"
                        "  General parameters\n"
                        "    -n #                        use n points from the file, default: 1B\n"
                        "    -d #                        dimension of the vector, default: 128\n"
                        "    -k #                        number of NN to search, default: 1\n"
                        "    -dataset SIFT1B / DEEP1B    Choose int for PQ compressed data or integer datasets like SIFT\n"
                        "                                Choose float for real datasets like DEEP\n"
        );
    }
};

#endif //IVF_HNSW_LIB_PARSER_H
