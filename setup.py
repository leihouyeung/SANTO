import setuptools

setuptools.setup(
    name="SANTO",
    version="0.0.2",
    author="Haoyang Li",
    author_email="lihy1995@gmail.com",
    description="SANTO: a coarse-to-fine stitching and alignment method for spatial omics",
    url="https://github.com/leihouyeung/SANTO",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # 依赖模块
    install_requires=[
        'torch==1.10.2',
        'numpy==1.21.6',
        'networkx==2.6.3',
        'scipy==1.5.0',
        'tqdm==4.65.0',
        'scanpy==1.9.3',
        'anndata==0.8.0',
        'glob2==0.7',
        'pandas==1.3.5',
        'ruptures==1.1.8',
        'harmonypy==0.0.6'
    ],
    python_requires='>=3.7',
)