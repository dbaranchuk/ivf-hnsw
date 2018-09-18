#include <cstring>
#include <iostream>
#include "Parser.h"

    Parser::Parser(int argc, char **argv)
    {
        cmd = argv[0];
        if (argc == 1)
            usage();

        for (size_t i = 1 ; i < argc; i++) {
            char *a = argv[i];

            if (!strcmp (a, "-h") || !strcmp (a, "--help"))
                usage();

            if (i == argc-1)
                break;

            //=================
            // HNSW parameters
            //=================
            if (!strcmp (a, "-M")) sscanf(argv[++i], "%zu", &M);
            else if (!strcmp (a, "-efConstruction")) sscanf(argv[++i], "%zu", &efConstruction);

            //=================
            // Data parameters
            //=================
            else if (!strcmp (a, "-nb")) sscanf(argv[++i], "%zu", &nb);
            else if (!strcmp (a, "-nc")) sscanf(argv[++i], "%zu", &nc);
            else if (!strcmp (a, "-nsubc")) sscanf(argv[++i], "%zu", &nsubc);
            else if (!strcmp (a, "-nt")) sscanf(argv[++i], "%zu", &nt);
            else if (!strcmp (a, "-nsubt")) sscanf(argv[++i], "%zu", &nsubt);
            else if (!strcmp (a, "-nq")) sscanf(argv[++i], "%zu", &nq);
            else if (!strcmp (a, "-ngt")) sscanf(argv[++i], "%zu", &ngt);
            else if (!strcmp (a, "-d")) sscanf(argv[++i], "%zu", &d);

            //===============
            // PQ parameters
            //===============
            else if (!strcmp (a, "-code_size"))sscanf(argv[++i], "%zu", &code_size);
            else if (!strcmp (a, "-opq")) do_opq = !strcmp(argv[++i], "on");

            //===================
            // Search parameters
            //===================
            else if (!strcmp (a, "-k")) sscanf(argv[++i], "%zu", &k);
            else if (!strcmp (a, "-nprobe")) sscanf(argv[++i], "%zu", &nprobe);
            else if (!strcmp (a, "-max_codes")) sscanf(argv[++i], "%zu", &max_codes);
            else if (!strcmp (a, "-efSearch")) sscanf(argv[++i], "%zu", &efSearch);
            else if (!strcmp (a, "-pruning")) do_pruning = !strcmp(argv[++i], "on");

            //=======
            // Paths
            //=======
            else if (!strcmp (a, "-path_base")) path_base = argv[++i];
            else if (!strcmp (a, "-path_learn")) path_learn = argv[++i];
            else if (!strcmp (a, "-path_q")) path_q = argv[++i];
            else if (!strcmp (a, "-path_gt")) path_gt = argv[++i];
            else if (!strcmp (a, "-path_centroids")) path_centroids = argv[++i];

            else if (!strcmp (a, "-path_precomputed_idx")) path_precomputed_idxs = argv[++i];

            else if (!strcmp (a, "-path_info")) path_info = argv[++i];
            else if (!strcmp (a, "-path_edges")) path_edges = argv[++i];

            else if (!strcmp (a, "-path_pq")) path_pq = argv[++i];
            else if (!strcmp (a, "-path_opq_matrix")) path_opq_matrix = argv[++i];
            else if (!strcmp (a, "-path_norm_pq")) path_norm_pq = argv[++i];
            else if (!strcmp (a, "-path_index")) path_index = argv[++i];
        }
    }

    void Parser::usage()
    {
        printf ("Usage: %s [options]\n", cmd);
        printf ("###################\n"
                "# HNSW Parameters #\n"
                "###################\n"
                "    -M #                  Min number of edges per point\n"
                "    -efConstruction #     Max number of candidate vertices in priority queue to observe during construction\n"
                "###################\n"
                "# Data Parameters #\n"
                "###################\n"
                "    -nb #                 Number of base vectors\n"
                "    -nt #                 Number of learn vectors\n"
                "    -nsubt #              Number of learn vectors to train (random subset of the learn set)\n"
                "    -nc #                 Number of centroids for HNSW quantizer\n"
                "    -nsubc #              Number of subcentroids per group\n"
                "    -nq #                 Number of queries\n"
                "    -ngt #                Number of groundtruth neighbours per query\n"
                "    -d #                  Vector dimension\n"
                "#################\n"
                "# PQ Parameters #\n"
                "#################\n"
                "    -code_size #          Code size per vector in bytes\n"
                "    -opq on/off           Turn on/off OPQ compression\n"
                "####################\n"
                "# Search Parameters #\n"
                "#####################\n"
                "    -k #                  Number of the closest vertices to search\n"
                "    -nprobe #             Number of probes at query time\n"
                "    -max_codes #          Max number of codes to visit to do a query\n"
                "    -efSearch #           Max number of candidate vertices in priority queue to observe during searching\n"
                "    -pruning on/off       Turn on/off pruning in the grouping scheme\n"
                "#########\n"
                "# Paths #\n"
                "#########\n"
                "    -path_base filename               Path to a base set\n"
                "    -path_learn filename              Path to a learn set\n"
                "    -path_q filename                  Path to queries\n"
                "    -path_gt filename                 Path to groundtruth\n"
                "    -path_centroids filename          Path to coarse centroids\n"
                "                            \n"
                "    -path_precomputed_idxs filename   Path to coarse centroid indices for base points\n"
                "                       \n"
                "    -path_info filename               Path to parameters of HNSW graph\n"
                "    -path_edges filename              Path to edges of HNSW graph\n"
                "                        \n"
                "    -path_pq filename                 Path to the product quantizer for residuals\n"
                "    -path_opq_matrix filename         Path to the rotation matrix for OPQ compression\n"
                "    -path_norm_pq filename            Path to the product quantizer for norms of reconstructed base points\n"
                "    "
                "    -path_index filename              Path to the constructed index\n"
        );
        exit(0);
    }
