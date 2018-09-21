# Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors

This is the code for the current state-of-the-art billion-scale nearest neighbor search system presented in the paper:

[Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors](http://openaccess.thecvf.com/content_ECCV_2018/html/Dmitry_Baranchuk_Revisiting_the_Inverted_ECCV_2018_paper.html),
<br>
Dmitry Baranchuk, Artem Babenko, Yury Malkov


The code is developed upon the [FAISS](https://github.com/facebookresearch/faiss) library.

### Build

Today we provide the C++ implementation supporting only the CPU version, 
which requires a BLAS library. 

The code requires a C++ compiler that understands: 

- the Intel intrinsics for SSE instructions
- the GCC intrinsic for the popcount instruction
- basic OpenMP

#### Installation instructions
1) Clone repository

```git clone https://github.com/dbaranchuk/ivf-hnsw --recursive```

2) Configure FAISS

There are a few models for makefile.inc in the faiss/example_makefiles/
subdirectory. Copy the relevant one for your system to faiss/ and adjust to your
needs. In particular, for ivf-hnsw project, you need to set a proper BLAS library paths.
There are also indications for specific configurations in the
troubleshooting section of the [FAISS wiki](https://github.com/facebookresearch/faiss/wiki/Troubleshooting)

3) Replace FAISS CMakeList.txt

Replace faiss/CMakeList.txt with CMakeList.txt.faiss in order to 
deactivate building of unnecessary tests and the GPU version.

```mv CMakeLists.txt.faiss faiss/CMakeLists.txt```

4) Build project

```cmake . ; make```

### Data
The proposed methods are tested on two 1 billion datasets: SIFT1B and DEEP1B. 
For using provided examples, all data files have to be in data/SIFT1B and data/DEEP1B.

#### Data files:
* SIFT1B:
   - dataset, [Datasets for approximate nearest neighbor search](http://corpus-texmex.irisa.fr/)
   
   ```cd data/SIFT1B ; bash load_sift1b.sh```
   - learned 993127 centroids, [GoogleDrive](https://drive.google.com/file/d/1p9Aq5lTiXzmuP1ftJAIqKYEEN5EVBZsS/view?usp=sharing)
   - precomputed indices of assigned base points, [GoogleDrive](https://drive.google.com/file/d/1iFgzY2niWsCwKCPpbsjZh1urudrswEyL/view?usp=sharing)
* DEEP1B:
   - dataset, [YandexDrive](https://yadi.sk/d/11eDCm7Dsn9GA)
   
   ```cd data/DEEP1B ; python load_deep1b.py```
   - learned 999973 centroids, [GoogleDrive](https://drive.google.com/file/d/1loJ0rEIBORM34vsVSZrNeJrq1OtrcmKu/view?usp=sharing)
   - precomputed indices of assigned base points, [GoogleDrive](https://drive.google.com/file/d/10DMFnLUs5Fdr_BCht9nsa2vSyG1LKJeV/view?usp=sharing) 
    
Note: precomputed indices are optional, as it just lets avoid assigning step, which takes about 2-3 days for 2^20 centroids.

### Run
tests/ provides two tests for each dataset: 
- IVFADC
- IVFADC + Grouping (+ Pruning)

Each test requires many options, so we provide bash scripts in examples/, 
exploiting these tests. Scripts are commented and 
the Parser class provides short descriptions for each option.  
  
Make sure that:
- models/SIFT1B/ and models/DEEP1B/ exist

```mkdir models ; mkdir models/SIFT1B ; mkdir models/DEEP1B```
- the data is placed to data/SIFT1B/ and data/DEEP1B/ respectively 
(or just make symbolic links)
- run, for example:

```bash examples/run_deep1b_grouping.sh```

### Documentation
The [doxygen documentation](https://cdn.rawgit.com/dbaranchuk/ivf-hnsw/fe2e4a85/docs/html/annotated.html) 
gives per-class information
 

