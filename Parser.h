#ifndef IVF_HNSW_LIB_PARSER_H
#define IVF_HNSW_LIB_PARSER_H

#include <iostream>
#include <cstring>
#include <cassert>

struct Parser
{
    const char * cmd;

    const char *path_centroids;
    const char *path_index;
    const char *path_precomputed_idxs;
    const char *path_pq;
    const char *path_norm_pq;
    const char *path_learn;
    const char *path_data;
    const char *path_q;
    const char *path_gt;
    const char *path_info;
    const char *path_edges;
    const char *path_groups;
    const char *path_idxs;

    int k;
    int vecsize;
    int qsize;
    int vecdim;
    int gtdim;
    int efConstruction;
    int M;
    int M_PQ;
    int efSearch;
    int nprobes;
    int max_codes;
    int ncentroids;
    int nsubcentroids;

    Parser(int argc, char **argv)
    {
        int ret;

        cmd = argv[0];
        if (argc == 1)
            usage();

        for (int i = 1 ; i < argc ; i++) {
            char *a = argv[i];

            if (!strcmp (a, "-h") || !strcmp (a, "--help"))
                usage();

            /** Paths **/
            if (!strcmp (a, "-path_data") && i+1 < argc) {
                path_data = argv[++i];
            }
            else if (!strcmp (a, "-path_info") && i+1 < argc) {
                path_info = argv[++i];
            }
            else if (!strcmp (a, "-path_edges") && i+1 < argc) {
                path_edges = argv[++i];
            }
            else if (!strcmp (a, "-path_q") && i+1 < argc) {
                path_q = argv[++i];
            }
            else if (!strcmp (a, "-path_gt") && i+1 < argc) {
                path_gt = argv[++i];
            }
            else if (!strcmp (a, "-path_pq") && i+1 < argc) {
                path_pq = argv[++i];
            }
            else if (!strcmp (a, "-path_norm_pq") && i+1 < argc) {
                path_norm_pq = argv[++i];
            }
            else if (!strcmp (a, "-path_precomputed_idx") && i+1 < argc) {
                path_precomputed_idxs = argv[++i];
            }
            else if (!strcmp (a, "-path_index") && i+1 < argc) {
                path_index = argv[++i];
            }
            else if (!strcmp (a, "-path_learn") && i+1 < argc) {
                path_learn = argv[++i];
            }
            else if (!strcmp (a, "-path_centroids") && i+1 < argc) {
                path_centroids = argv[++i];
            }
            else if (!strcmp (a, "-path_groups") && i+1 < argc) {
                path_groups = argv[++i];
            }
            else if (!strcmp (a, "-path_idxs") && i+1 < argc) {
                path_idxs = argv[++i];
            }
                /** Int Parameters **/
            else if (!strcmp (a, "-k") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &k);
                assert (ret);
            }
            else if (!strcmp (a, "-efConstruction") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &efConstruction);
                assert (ret);
            }
            else if (!strcmp (a, "-d") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &vecdim);
                assert (ret);
            }
            else if (!strcmp (a, "-gt_d") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &gtdim);
                assert (ret);
            }
            else if (!strcmp (a, "-n") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &vecsize);
                assert (ret);
            }
            else if (!strcmp (a, "-nq") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &qsize);
                assert (ret);
            }
            else if (!strcmp (a, "-M") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &M);
                assert (ret);
            }
            else if (!strcmp (a, "-nc") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &ncentroids);
                assert (ret);
            }
            else if (!strcmp (a, "-M_PQ") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &M_PQ);
                assert (ret);
            }
            else if (!strcmp (a, "-efSearch") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &efSearch);
                assert (ret);
            }
            else if (!strcmp (a, "-nprobes") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &nprobes);
                assert (ret);
            }
            else if (!strcmp (a, "-max_codes") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &max_codes);
                assert (ret);
            }
            else if (!strcmp (a, "-nsubcentroids") && i+1 < argc) {
                ret = sscanf (argv[++i], "%d", &nsubcentroids);
                assert (ret);
            }
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
                        "    -path_data filename         set of base vectors (bvecs file format)\n"
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
