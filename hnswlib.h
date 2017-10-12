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


namespace hnswlib {
	//typedef float(*DISTFUNC) (const void *, const void *, const void *);
	template<typename MTYPE>
	using DISTFUNC = MTYPE(*) (const void *, const void *, const void *);


	typedef unsigned int idx_t;
	typedef unsigned char uint8_t;

	inline bool exists_test(const std::string& name) {
		std::ifstream f(name.c_str());
		return f.good();
	}

	template <typename format>
	void readXvec(std::ifstream &input, format *mass, const int d, const int n = 1)
	{
		int in = 0;
		for (int i = 0; i < n; i++) {
			input.read((char *) &in, sizeof(int));
			if (in != d) {
				std::cout << "file error\n";
				exit(1);
			}
			input.read((char *)(mass+i*d), in * sizeof(format));
		}
	}
	
//	template<typename MTYPE>
//	class SpaceInterface {
//	public:
//		//virtual void search(void *);
//		virtual size_t get_data_size() = 0;
//		virtual DISTFUNC<MTYPE> get_dist_func() = 0;
//		virtual void *get_dist_func_param() = 0;
//
//	};
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
#include "brutoforce.h"
#include "hnswalg.h"
