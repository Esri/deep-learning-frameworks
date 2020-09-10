# Deep Learning Libraries Installers for ArcGIS

<div align="center">
  <img src="images/included-frameworks.png" />
</div>

ArcGIS Pro, Server and the ArcGIS API for Python all include tools to use AI and Deep Learning to solve geospatial problems, such as feature extraction, pixel classification, and feature categorization.  This installer includes a broad collection of components, such as PyTorch, TensorFlow, Fast.ai and scikit-learn, for performing deep learning and machine learning tasks, a total collection of 95 packages. These packages can be used with the new Deep Learning Training tools, by using the arcgis.learn module within the ArcGIS API for Python, and directly imported into your own scripts and tools. Most of the tools in this collection will work on any machine, but common deep learning workflows require a recent NVIDIA graphics processing unit (GPU), and problem sizes are bound by available GPU memory, see [the requirements section](#requirements).

This installer adds all the included packages to the default `arcgispro-py3` environment that Pro and Server both ship with, and no additional environments are necessary to create in order to get started using the tools. If you do create custom environments, these packages will also be included so you can use the same tools in your own custom environments as well. If you cannot use the installer, you can [install the libraries manually using these instructions](install-deep-learning-frameworks-manually.pdf).

For an example of the kinds of workflows this installer and ArcGIS enables, see the [AI & Deep Learning in the UC 2020 Plenary video](https://www.youtube.com/watch?v=eI5Sv_FsPgk&feature=youtu.be&list=PLaPDDLTCmy4YwK56yHaEdtRgNUoPBiZTz)

Download
--------

![GitHub All Releases](https://img.shields.io/github/downloads/esri/deep-learning-frameworks/total?style=for-the-badge)

  - **[Deep Learning Libraries Installer for ArcGIS Pro 2.6](https://github.com/Esri/deep-learning-frameworks/releases/download/pro-2.6/ArcGIS_Pro_26_Deep_Learning_Libraries.zip)**
  - **[Deep Learning Libraries Installer for ArcGIS Server 10.8.1](https://github.com/Esri/deep-learning-frameworks/releases/download/server-10.8.1/ArcGIS_Server_1081_Deep_Learning_Libraries.zip)**
  - **[Deep Learning Libraries Installer for ArcGIS Server Linux 10.8.1](https://github.com/Esri/deep-learning-frameworks/releases/download/linux-server-10.8.1/ArcGIS_Linux_Server_1081_Deep_Learning_Libraries.tar.gz)**
 
Next Steps
----------

Once you've installed the deep learning libraries, you can use the [Deep Learning Tools](https://pro.arcgis.com/en/pro-app/help/analysis/image-analyst/deep-learning-in-arcgis-pro.htm) to train geospatial deep learning models. You can also find out more about the capabilities of the [arcgis.learn module](https://developers.arcgis.com/python/guide/geospatial-deep-learning/) which provides specialized access to many geospatial models beyond those directly available as Geoprocessing tools. Finally, you can add any of the above libraries to your own workflows, by importing the packages listed below.

 - Learn ArcGIS lesson on using [Deep Learning to Access Palm Tree Health](https://learn.arcgis.com/en/projects/use-deep-learning-to-assess-palm-tree-health/)
 - Join us in October 2020 for the [Spatial Data Science MOOC](https://www.esri.com/training/catalog/5d76dcf7e9ccda09bef61294/spatial-data-science:-the-new-frontier-in-analytics/#!)

A collection of recent User Conference 2020 Technical Workshops on Deep Learning:

 - [Deep Dive into Deep Learning](https://uc2020.esri.com/live-stream/15346124/Deep-Dive-into-Deep-Learning)
 - [Deep Learning for Geographers](https://uc2020.esri.com/sessions/15346174/Deep-Learning-for-Geographers)
 - [Using Deep Learning with Imagery in ArcGIS](https://uc2020.esri.com/live-stream/15345718/Using-Deep-Learning-with-Imagery-in-ArcGIS)

Requirements
------------

Most of the packages included in the Deep Learning Libraries installer will work out of the box on any machine configuration. For example, PyTorch optionally can take advantage of a GPU, but will fall back to running its calculations on the CPU if a GPU is not available. However, GPU computation is significantly faster, and some packages such as TensorFlow in this distribution only will work with a supported GPU.  CUDA, or Compute Unified Device Architecture, is a general purpose computing platform for GPUs, a requirement for current GPU backed deep learning tools.

 GPU requirement | Supported
 -----|---------------------
 GPU Type | NVIDIA with CUDA Compute Capability<sup>&ast;</sup> 3.5 or greater
 Dedicated graphics memory <sup>&dagger;</sup> | minimum: 2GB <br />recommended: 4GB or more, depending on the size of models trained

&ast; NVIDIA provides a list of [CUDA enabled products](https://developer.nvidia.com/cuda-gpus#compute) and their compute capability.

&dagger; GPU memory, unlike system memory, cannot be accessed 'virtually'. If a model training consumes more GPU memory than you have available, it will fail. GPU memory is also shared across all uses of the machine, so open Pro projects with maps and other applications can limit the available memory for use with these tools.


Manifest of included packages
-----------------------------

Library Name | Version | Description
-------------|---------|------------
absl-py | 0.9.0 | Abseil Python Common Libraries
appdirs | 1.4.3 | A small Python module for determining appropriate platform-specific directories
ase | 3.19.1 | Set of tools for atomistic simulations
astor | 0.8.0 | Read, rewrite, and write Python ASTs nicely
beautifulsoup4 | 4.9.0 | Python library designed for screen-scraping
cachetools | 3.1.1 | Extensible memoizing collections and decorators
catalogue | 1.0.0 | Super lightweight function registries for your library
click | 7.1.2 | Python composable command line interface toolkit
cloudpickle | 1.4.1 | Extended pickling support for Python objects
cudatoolkit | 10.1.243 | NVIDIA CUDA toolkit
cudnn | 7.6.5 | NVIDIA's cuDNN deep neural network acceleration library
cymem | 2.0.2 | Manage calls to calloc/free through Cython
cython-blis | 0.4.1 | Fast matrix-multiplication as a self-contained Python library
cytoolz | 0.10.1 | Cython implementation of Toolz. High performance functional utilities
dask-core | 2.17.2 | Parallel Python with task scheduling
dataclasses | 0.6 | A backport of the dataclasses module for Python 3.6
fastai | 1.0.60 | fastai makes deep learning with PyTorch faster, more accurate, and easier
fastprogress | 0.2.3 | A fast and simple progress bar for Jupyter Notebook and console
gast | 0.2.2 | Python AST that abstracts the underlying Python version
google-auth | 1.14.1 | Google authentication library for Python
google-auth-oauthlib | 0.4.1 | Google Authentication Library, oauthlib integration with google-auth
google-pasta | 0.2.0 | pasta is an AST-based Python refactoring library
googledrivedownloader | 0.4 | Minimal class to download shared files from Google Drive
graphviz | 2.38 | Open Source graph visualization software
grpcio | 1.27.2 | HTTP/2-based RPC framework
imageio | 2.8.0 | A Python library for reading and writing image data
isodate | 0.6.0 | An ISO 8601 date/time/duration parser and formatter
joblib | 0.15.1 | Lightweight pipelining: using Python functions as pipeline jobs
jpeg | 9b | Read and write jpeg COM, EXIF, and IPTC medata
keepalive | 0.5 | An HTTP handler for urllib that supports HTTP 1.1 and keepalive
keras-applications | 1.0.8 | Applications module of the Keras deep learning library
keras-base | 2.3.1 | Deep Learning Library for Theano and TensorFlow, base package
keras-gpu | 2.3.1 | Deep Learning Library for Theano and TensorFlow
keras-preprocessing | 1.1.0 | Data preprocessing and data augmentation module of the Keras deep learning library
laspy | 1.7.0 | A Python library for reading, modifying and creating LAS files
libprotobuf | 3.11.4 | Protocol Buffers - Google's data interchange format. C++ Libraries
libtiff | 4.0.10 | Support for the Tag Image File Format (TIFF)
llvmlite | 0.32.1 | A lightweight LLVM python binding for writing JIT compilers
markdown | 3.1.1 | Python implementation of Markdown
murmurhash | 1.0.2 | Cython bindings for MurmurHash2
ninja | 1.9.0 | A small build system with a focus on speed
numba | 0.49.1 | NumPy aware dynamic Python compiler using LLVM
nvidia-ml-py3 | 7.352.0 | Python bindings to the NVIDIA Management Library
olefile | 0.46 | Parse, read and write Microsoft OLE2 files
onnx | 1.7.0 | Open Neural Network Exchange library
onnx-tf | 1.5.0 | Experimental Tensorflow Backend for ONNX
opencv | 4.0.1.24 | Computer vision and machine learning software library
opt_einsum | 3.1.0 | Optimizing einsum functions in NumPy, Tensorflow, Dask, and more with contraction order optimization
pillow-simd | 7.1.2 | Python image manipulation, SIMD optimized for SSE4 compatible CPUs
plac | 1.1.0 | The smartest command line arguments parser in the world
plotly | 4.5.4 | An interactive JavaScript-based visualization library for Python
pooch | 1.0.0 | A friend to fetch your Python library's sample data files
preshed | 3.0.2 | Cython Hash Table for Pre-Hashed Keys
protobuf | 3.11.4 | Protocol Buffers - Google's data interchange format
pyasn1 | 0.4.8 | ASN.1 types and codecs
pyasn1-modules | 0.2.7 | A collection of ASN.1-based protocols modules
pytorch | 1.4.0 | PyTorch is an optimized tensor library for deep learning using GPUs and CPUs
pywavelets | 1.1.1 | Discrete Wavelet Transforms in Python
pyyaml | 5.3.1 | YAML parser and emitter for Python
rdflib | 5.0.0 | Library for working with RDF, a simple yet powerful language for representing information
retrying | 1.3.3 | Simplify the task of adding retry behavior to just about anything
rsa | 4.0 | Pure-Python RSA implementation
scikit-image | 0.17.2 | Image processing routines for SciPy
scikit-learn | 0.22.1 | A set of python modules for machine learning and data mining
soupsieve | 2.0.1 | A modern CSS selector implementation for BeautifulSoup
spacy | 2.2.4 | Industrial-strength Natural Language Processing
sparqlwrapper | 1.8.5 | SPARQL Endpoint interface to Python for use with rdflib
srsly | 1.0.2 | Modern high-performance serialization utilities for Python
tensorboard | 2.2.1 | TensorBoard lets you watch Tensors Flow
tensorboard-plugin-wit | 1.6.0 | What-If Tool TensorBoard plugin
tensorboardx | 1.7 | Tensorboard for PyTorch
tensorflow | 2.1.0 | TensorFlow is a machine learning library
tensorflow-addons | 0.9.1 | Useful extra functionality for TensorFlow 2
tensorflow-base | 2.1.0 | TensorFlow is a machine learning library, base package
tensorflow-estimator | 2.1.0 | TensorFlow Estimator
tensorflow-gpu | 2.1.0 | Metapackage for selecting a TensorFlow variant
termcolor | 1.1.0 | ANSI Color formatting for output in terminal
thinc | 7.4.0 | Learn super-sparse multi-class models
tifffile | 0.15.1 | Read and write image data from and to TIFF files
toolz | 0.10.0 | A functional standard library for Python
torch-cluster | 1.5.4 | Extension library of highly optimized graph cluster algorithms for use in PyTorch
torch-geometric | 1.5.0 | Geometric deep learning extension library for PyTorch
torch-scatter | 2.0.4 | Extension library of highly optimized sparse update (scatter and segment) operations
torch-sparse | 0.6.1 | Extension library of optimized sparse matrix operations with autograd support
torch-spline-conv | 1.2.0 | PyTorch implementation of the spline-based convolution operator of SplineCNN
torchvision | 0.5.0 | Image and video datasets and models for torch deep learning
tqdm | 4.46.0 | A Fast, Extensible Progress Meter
transforms3d | 0.3.1 | Functions for 3D coordinate transformations
typeguard | 2.7.0 | Runtime type checker for Python
wasabi | 0.6.0 | A lightweight console printing and formatting toolkit
werkzeug | 0.16.1 | The comprehensive WSGI web application library
wrapt | 1.12.1 | Module for decorators, wrappers and monkey patching
xz | 5.2.5 | Data compression software with high compression ratio
yaml | 0.1.7 | A C library for parsing and emitting YAML
zstd | 1.3.7 | Zstandard - Fast real-time compression algorithm
_tflow_select | 2.1.0 | Metapackage for selecting GPU or CPU version of TensorFlow

Additional Notes
----------------

 - Though this package distributes the GPU based versions of packages, CPU versions can still be installed and used on any machine Pro supports. To install TensorFlow for the CPU, from the Python backstage you can install the `tensorflow-mkl` package to get a CPU only version.
 - This installer adds packages to the default `arcgispro-py3` environment. Any subsequent clones of that environment will also include this full collection of packages. This collection of packages is validated and tested against the version of Pro is installed alongside, and upgrades of Pro will also require reinstallation of the deep learning libraries. Note that when you upgrade the software to a new release, you'll need to uninstall the Deep Learning Libraries installation as well as Pro or Server, and reinstall the new version of this package for that release.
 - This installer is only available for ArcGIS Pro 2.6, and ArcGIS Server 10.8.1 -- for earlier releases, you'll need to follow the documentation on installing the packages through the Python backstage or Python command prompt.
 - If you want these packages for a specific environment only, you can install the `deep-learning-essentials` package which has the same list of dependencies as a standalone conda metapackage.
