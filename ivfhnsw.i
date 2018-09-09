%module ivfhnsw
%{
#define SWIG_FILE_WITH_INIT
#include "IndexIVF_HNSW.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int DIM1, float* IN_ARRAY1) {(size_t n, const float* x)};
%include "IndexIVF_HNSW.h"
