# DRAFT

## Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors


### Build
Today we provide the C++ implementation supporting only the CPU version, 
which requires a BLAS library. 

The code requires a C++ compiler that understands:
- the Intel intrinsics for SSE instructions
- the GCC intrinsic for the popcount instruction
- basic OpenMP

1) Configure FAISS

There are a few models for makefile.inc in the faiss/example_makefiles/
subdirectory. Copy the relevant one for your system to faiss/ and adjust to your
needs. In particular, for ivf-hnsw project, you need to set a proper BLAS library paths.
There are also indications for specific configurations in the
troubleshooting section of the FAISS wiki.

https://github.com/facebookresearch/faiss/wiki/Troubleshooting

2) Replace FAISS CMakeList.txt

Replace faiss/CMakeList.txt with CMakeList.txt.faiss in order to 
deactivate building of unnecessary tests and the GPU version.   

3) Build project
```
cmake .
make
```

### Data
The proposed methods are tested on two 1 billion datasets: SIFT1B and DEEP1B. 
For using provided examples, all data files have to be in data/SIFT1B and data/DEEP1B.

Data files:
* SIFT1B:
   - dataset [Datasets for approximate nearest neighbor search](http://corpus-texmex.irisa.fr/)
   - learned 2^20 centroids [GoogleDrive](https://drive.google.com/open?id=1p9Aq5lTiXzmuP1ftJAIqKYEEN5EVBZsS)
   - centroid labels for assigned base points [GoogleDrive]()
* DEEP1B:
   - dataset [YandexDrive](https://yadi.sk/d/11eDCm7Dsn9GA)
   - learned 2^20 centroids [GoogleDrive](https://drive.google.com/drive/folders/1lW6zTpW8doSItU2HYUzoKWLTG_rjtYEV)
   - centroid labels for assigned base points [GoogleDrive]() 
    
Note: centroid labels are optional, as it just lets avoid assigning step, which takes about 2-3 days for 2^20 centroids.

We provide scripts loading all necessary files, including centroid labels.

##### SIFT1B:
```
cd data/SIFT1B
bash load_sift1b.sh
```
##### DEEP1B:
```
cd data/DEEP1B
python load_deep1b.py
```


 

