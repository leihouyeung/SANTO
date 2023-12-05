
# SANTO: a coarse-to-fine stitching and alignment method for spatial omics


SANTO is a coarse-to-fine method targeting alignment and stitching tasks for spatial omics data. Before using SANTO, several parameters should be specified:
- `mode`: Users should choose their task is `align` or `stitch`.
- `dimension`: The dimensionality of spatial omics data (2 or 3).
- `diff_omics`: `True` or `False`. If users want to align two different spatial omics data, e.g. ATAC-seq and RNA-seq.
- `alpha`: The weight of feature-level loss.
- `k`: The number of neighbors during graph construction.
- `lr`: Learning rate.
- `epochs`: Epochs. 

## Installation
It's easy to install SANTO by only one command.
`pip install SANTO`

## Usage
The input data of SANTO including:
- Two `h5ad` files of spatial omics data with spatial coordinates (`.obsm['spatial']`). One is source slice and the other is target slice. SANTO aims to align/stitch source slice to target slice.
- The dict of arguments mentioned above. 

The function users should use is:

`aligned_source_coor, transform_dict = santo(source_slice, target_slice, args)`

Returned `align_source_coor` is transformed spatial coordinates of source slice. `transform_dict` includes the coarse and fine rotation and translation. 

The tutorial is shown in this notebook. Please have a check. 


```
