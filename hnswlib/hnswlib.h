#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)

#endif

typedef unsigned int labeltype;

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

using namespace std;


inline bool exists_test(const std::string& name) {
	std::ifstream f(name.c_str());
	return f.good();
}

namespace hnswlib {
    template<typename dist_t>
    class SpaceInterface {
    public:
        virtual size_t get_data_size() = 0;
        virtual size_t get_data_dim() = 0;
        virtual dist_t fstdistfunc(const void *, const void *) = 0;
        virtual dist_t fstdistfuncST(const void *) = 0;
    };
}
#include "L2space.h"
#include "hnswalg.h"