#pragma once
#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib{

	typedef uint16_t vl_type;

class VisitedList {
public:
	vl_type curV;
	vl_type *mass;
	size_t numelements;

	VisitedList(size_t numelements1)
	{
		curV = -1;
		numelements = numelements1;
		mass = new vl_type[numelements];
	}

	void reset()
	{
		curV++;
		if (curV == 0) {
			memset(mass, 0, sizeof(vl_type) * numelements);
			curV++;
		}
	};

	~VisitedList() { delete mass; }
};

///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
	std::deque<VisitedList *> pool;
	std::mutex poolguard;
	size_t maxpools;
	size_t numelements;

public:
	VisitedListPool(size_t initmaxpools, size_t numelements1)
	{
		numelements = numelements1;
		for (size_t i = 0; i < initmaxpools; i++)
			pool.push_front(new VisitedList(numelements));
	}

	VisitedList *getFreeVisitedList()
	{
		VisitedList *rez;
		{
			std::unique_lock<std::mutex> lock(poolguard);
			if (pool.size() > 0) {
				rez = pool.front();
				pool.pop_front();
			}
			else {
				rez = new VisitedList(numelements);
			}
		}
		rez->reset();
		return rez;
	};

	void releaseVisitedList(VisitedList *vl)
	{
		std::unique_lock<std::mutex> lock(poolguard);
		pool.push_front(vl);
	};

	~VisitedListPool()
	{
		while (pool.size()) {
			VisitedList *rez = pool.front();
			pool.pop_front();
			delete rez;
		}
	};
};
}

