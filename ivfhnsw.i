%module ivfhnsw
%{
#define SWIG_FILE_WITH_INIT
#include "IndexIVF_HNSW.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float *x, size_t n, size_t d)};
%apply (unsigned int* ARGOUT_ARRAY1, int DIM1) {(ivfhnsw::IndexIVF_HNSW::idx_t *labels, size_t k)};

%rename (assign) assign_numpy;
%exception my_dot {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%extend ivfhnsw::IndexIVF_HNSW {
void assign_numpy(const float *x, size_t n, size_t d, idx_t *labels, size_t k) {
    if (d != $self->d) {
        PyErr_Format(PyExc_ValueError,
                     "Vector length must be equal d=%d, got %d",
                     $self->d, d);
        return;
    }
    return $self->assign(n, x, labels, k);
}
}

%ignore assign;
%include "IndexIVF_HNSW.h"

