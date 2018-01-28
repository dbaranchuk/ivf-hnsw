#!/bin/bash

P="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

wget -P $P ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
gzip -d $P/bigann_base.bvecs.gz

wget -P $P ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz
gzip -d $P/bigann_learn.bvecs.gz

wget -P $P ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
gzip -d $P/bigann_query.bvecs.gz

wget -P $P ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
tar xvzf $P/bigann_gnd.tar.gz