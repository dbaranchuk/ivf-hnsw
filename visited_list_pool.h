#pragma once
#include <mutex>
#include <string.h>

//#include <unordered_set>
#include <sparsehash/dense_hash_set>

using google::dense_hash_set;

namespace hnswlib{
typedef unsigned short vl_type;
typedef dense_hash_set<unsigned int> VisitedSet;

class VisitedList {
public:
	vl_type curV;
	vl_type *mass;
	unsigned int numelements;

	VisitedList(unsigned int numelements1)
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

class VisitedSetPool {
	deque<VisitedSet *> pool;
	mutex poolguard;

public:
	VisitedSetPool(int initmaxpools)
	{
		for (int i = 0; i < initmaxpools; i++) {
			pool.push_front(new VisitedSet());
			pool.front()->set_empty_key(NULL);
		}
	}
	VisitedSet *getFreeVisitedSet()
	{
		VisitedSet *rez;
		{
			unique_lock<mutex> lock(poolguard);
			if (pool.size() > 0) {
				rez = pool.front();
				pool.pop_front();
			}
			else {
				rez = new VisitedSet();
				rez->set_empty_key(NULL);
			}
		}
		rez->clear();
		return rez;
	};
	void releaseVisitedSet(VisitedSet *vs)
	{
		unique_lock<mutex> lock(poolguard);
		pool.push_front(vs);
	};
	~VisitedSetPool()
	{
		while (pool.size()) {
			VisitedSet *rez = pool.front();
			pool.pop_front();
			delete rez;
		}
	};
};

class VisitedListPool {
	deque<VisitedList *> pool;
	mutex poolguard;
	int maxpools;
	int numelements;

public:
	VisitedListPool(int initmaxpools, int numelements1)
	{
		numelements = numelements1;
		for (int i = 0; i < initmaxpools; i++)
			pool.push_front(new VisitedList(numelements));
	}
	VisitedList *getFreeVisitedList()
	{
		VisitedList *rez;
		{
			std::cout << "HUI\n";
			unique_lock<mutex> lock(poolguard);
			if (pool.size() > 0) {
				rez = pool.front();
				pool.pop_front();
			}
			else {
				std::cout << numelements << std::endl;
				rez = new VisitedList(numelements);
			}
		}
		std::cout << "HUI\n";
		rez->reset();
		std::cout << "HUI\n";
		return rez;
	};
	void releaseVisitedList(VisitedList *vl)
	{
		unique_lock<mutex> lock(poolguard);
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

