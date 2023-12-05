
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

Here is the detailed tutorial:
```
import santo as san
import scanpy as sc
import easydict

# For alignment task
args = easydict.EasyDict({})
args.epochs = 30
args.lr = 0.01
args.k = 10             # number of neighbors during graph construction
args.diff_omics = False # whether to use different omics data
args.alpha = 0.9        # weight of transcriptional loss (0 to 1)
args.mode = 'align'     # Choose the mode among 'align', 'stitch' and None
args.dimension = 2      # choose the dimension of coordinates (2 or 3)

path1 = './examples/starmap_align_source.h5ad'
path2 = './examples/starmap_align_target.h5ad'
source = sc.read_h5ad(path1)
target = sc.read_h5ad(path2)

align_source_cor, trans_dict = san.santo(source, target, args)


# For stitching task

args = easydict.EasyDict({})
args.epochs = 100
args.lr = 0.001
args.k = 10             # number of neighbors during graph construction
args.diff_omics = False # whether to use different omics data
args.alpha = 0.9        # weight of transcriptional loss (0 to 1)
args.mode = 'stitch'    # Choose the mode among 'align', 'stitch' and None
args.dimension = 2      # choose the dimension of coordinates (2 or 3)

path1 = './examples/visium_stitch_source.h5ad'
path2 = './examples/visium_stitch_target.h5ad'
source = sc.read_h5ad(path1)
target = sc.read_h5ad(path2)

align_source_cor, trans_dict = san.santo(source, target, args)

```

For example dataset, please download from [here](https://drive.google.com/drive/folders/18VAlAIkixUksd_I8oiZMVys3zeMrk9pA?usp=sharing). Create a new folder `/examples` and put the data into `/examples` folder.
