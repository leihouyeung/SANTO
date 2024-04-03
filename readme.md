
# SANTO: a coarse-to-fine alignment and stitching method for spatial omics

![Pipeline of SANTO](/pipeline.png)

SANTO is a coarse-to-fine method targeting alignment and stitching tasks for spatial omics data. Before using SANTO, several parameters should be specified:
- `mode`: Users should choose their task is `align` or `stitch`.
- `dimension`: The dimensionality of spatial omics data (2 or 3).
- `diff_omics`: `True` or `False`. If users want to align two different spatial omics data, e.g. ATAC-seq and RNA-seq.
- `alpha`: The weight of feature-level loss.
- `k`: The number of neighbors during graph construction.
- `lr`: Learning rate.
- `epochs`: Epochs. 
- `device`: The device you use. (recommend 'cuda:0')


## Installation
Users need to create an environment and install SANTO by following procedures:
```
conda create -n santo_env python=3.10
conda activate santo_env
pip install -r requirements.txt
```

## Usage
The input data of SANTO includes:
- Two `h5ad` files of spatial omics data with spatial coordinates (`.obsm['spatial']`). One is source slice and the other is target slice. SANTO aims to align/stitch source slice to target slice.
- The dict of arguments mentioned above. 

The function users should use is:

`aligned_source_coor, transform_dict = santo(source_slice, target_slice, args)`

Returned `align_source_coor` is the transformed spatial coordinates of source slice. 
`transform_dict` includes the coarse and fine rotation and translation. 

## Demo
Please see the Jupyter notebook called `SANTO.ipynb`. It includes the demos for aligning Starmap PLUS, DLPFC and MERFISH datasets by SANTO.

## Reproducibility

We supplied the reproducibility of our method, including the benchmarking with PASTE, PASTE2, SLAT and STAligner under Starmap PLUS, DLPFC and MERFISH datasets.
