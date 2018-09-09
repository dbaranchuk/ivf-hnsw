%module ivfhnsw
%{
#define SWIG_FILE_WITH_INIT
#include "IndexIVF_HNSW.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%include "IndexIVF_HNSW.h"
