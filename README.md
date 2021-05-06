# Deep Learning Libraries Installers for ArcGIS

<div align="center">
  <img src="images/included-frameworks.png" />
</div>

ArcGIS Pro, Server and the ArcGIS API for Python all include tools to use AI and Deep Learning to solve geospatial problems, such as feature extraction, pixel classification, and feature categorization.  This installer includes a broad collection of components, such as PyTorch, TensorFlow, Fast.ai and scikit-learn, for performing deep learning and machine learning tasks, a total collection of 99 packages. These packages can be used with the [Deep Learning Training tools](https://pro.arcgis.com/en/pro-app/latest/help/analysis/image-analyst/deep-learning-in-arcgis-pro.htm), [interactive object detection](https://pro.arcgis.com/en/pro-app/latest/help/mapping/exploratory-analysis/interactive-object-detection-basics.htm), by using the [`arcgis.learn`](https://developers.arcgis.com/python/guide/geospatial-deep-learning/) module within the ArcGIS API for Python, and directly imported into your own scripts and tools. Most of the tools in this collection will work on any machine, but common deep learning workflows require a recent NVIDIA graphics processing unit (GPU), and problem sizes are bound by available GPU memory, see [the requirements section](#requirements).

This installer adds all the included packages to the default [`arcgispro-py3` environment](https://pro.arcgis.com/en/pro-app/latest/arcpy/get-started/installing-python-for-arcgis-pro.htm) that Pro and Server both ship with, and no additional environments are necessary in order to get started using the tools. If you do create custom environments, these packages will also be included so you can use the same tools in your own custom environments as well. 

For an example of the kinds of workflows this installer and ArcGIS enables, see the [AI & Deep Learning in the UC 2020 Plenary video](https://www.youtube.com/watch?v=eI5Sv_FsPgk&feature=youtu.be&list=PLaPDDLTCmy4YwK56yHaEdtRgNUoPBiZTz)

Download
--------

![GitHub All Releases](https://img.shields.io/github/downloads/esri/deep-learning-frameworks/total?style=for-the-badge)

  - **[Deep Learning Libraries Installer for ArcGIS Pro 2.7](https://github.com/Esri/deep-learning-frameworks/releases/download/pro-2.7/ArcGIS_Pro_27_Deep_Learning_Libraries.zip)**
  - **[Deep Learning Libraries Installer for ArcGIS Pro 2.6](https://github.com/Esri/deep-learning-frameworks/releases/download/pro-2.6/ArcGIS_Pro_26_Deep_Learning_Libraries.zip)**
  - **[Deep Learning Libraries Installer for ArcGIS Server 10.8.1](https://github.com/Esri/deep-learning-frameworks/releases/download/server-10.8.1/ArcGIS_Server_1081_Deep_Learning_Libraries.zip)**
  - **[Deep Learning Libraries Installer for ArcGIS Server Linux 10.8.1](https://github.com/Esri/deep-learning-frameworks/releases/download/linux-server-10.8.1/ArcGIS_Linux_Server_1081_Deep_Learning_Libraries.tar.gz)**
  
Once you've downloaded the archive for your product, extract the Zip file to a new location, and run the Windows Installer (MSI, e.g. `ProDeepLearning.msi`) on Windows. Note that this will install the deep learning frameworks into your currently active Pro Python environment, so be sure to switch to the environment you wish to install into before running the MSI using either ArcGIS Pro's Python manager UI or on the command line with `proswap`. You'll need to extract the file (not just open the .MSI from within the Zip file) or the installer won't be able to find its contents. On Linux, extrac the .tar.gz archive, e.g. with `tar xvf <file>.tar.gz`, then run the `DeepLearning-Setup.sh` script. After installation, the archive and installer files can be deleted.


Manual Installation
--------

If you cannot use the Pro installer, you can install the libraries manually using these instructions:
  - **[Pro 2.7 Manual Installation Instructions](install-deep-learning-frameworks-manually-2-7.pdf)**
  - **[Pro 2.6 Manual Installation Instructions](install-deep-learning-frameworks-manually-2-6.pdf)**  

Additional Installation for Disconnected Environment
--------

If you will be working in a disconnected environment, download the [arcgis_dl_backbones package](https://geosaurus.maps.arcgis.com/home/item.html?id=d404fd50d05d475f8d92eedb78e1c961) and follow the instructions under the **Steps to Install** listed on the package page. The package places backbones for deep learning models in the specified install location, eliminating the need for internet access when training deep learning models in ArcGIS.


Next Steps
----------

Once you've installed the deep learning libraries, you can use the [Deep Learning Tools](https://pro.arcgis.com/en/pro-app/help/analysis/image-analyst/deep-learning-in-arcgis-pro.htm) to train geospatial deep learning models. You can also find out more about the capabilities of the [arcgis.learn module](https://developers.arcgis.com/python/guide/geospatial-deep-learning/) which provides specialized access to many geospatial models beyond those directly available as Geoprocessing tools. Finally, you can add any of the above libraries to your own workflows, by importing the packages listed below.

 - Learn ArcGIS lesson on using [Deep Learning to Access Palm Tree Health](https://learn.arcgis.com/en/projects/use-deep-learning-to-assess-palm-tree-health/)
 - Join us later in 2021 for the [Spatial Data Science MOOC](https://www.esri.com/training/catalog/5d76dcf7e9ccda09bef61294/spatial-data-science:-the-new-frontier-in-analytics/#!)

A collection of recent User Conference 2020 Technical Workshops on Deep Learning:

 - [Deep Dive into Deep Learning](https://uc2020.esri.com/live-stream/15346124/Deep-Dive-into-Deep-Learning)
 - [Deep Learning for Geographers](https://uc2020.esri.com/sessions/15346174/Deep-Learning-for-Geographers)
 - [Using Deep Learning with Imagery in ArcGIS](https://uc2020.esri.com/live-stream/15345718/Using-Deep-Learning-with-Imagery-in-ArcGIS)

Requirements
------------

Most of the packages included in the Deep Learning Libraries installer will work out of the box on any machine configuration. For example, PyTorch optionally can take advantage of a GPU, but will fall back to running its calculations on the CPU if a GPU is not available. However, GPU computation is significantly faster, and some packages such as TensorFlow in this distribution only will work with a supported GPU.  CUDA, or Compute Unified Device Architecture, is a general purpose computing platform for GPUs, a requirement for current GPU backed deep learning tools.

 GPU requirement | Supported
 -----|---------------------
 GPU Type | NVIDIA with CUDA Compute Capability<sup>&ast;</sup> 3.5 minimum, 6.1 or higher recommended
 Dedicated graphics memory <sup>&dagger;</sup> | minimum: 2GB <br />recommended: 8GB, depending on the deep learning model architecture and the batch size being used

&ast; NVIDIA provides a list of [CUDA enabled products](https://developer.nvidia.com/cuda-gpus#compute) and their compute capability.

&dagger; GPU memory, unlike system memory, cannot be accessed 'virtually'. If a model training consumes more GPU memory than you have available, it will fail. GPU memory is also shared across all uses of the machine, so open Pro projects with maps and other applications can limit the available memory for use with these tools.


Manifest of included packages
-----------------------------

Library Name | Version | Description
-------------|---------|------------
absl-py | 0.11.0 | Abseil Python Common Libraries, see https://github.com/abseil/abseil-py.
aiohttp | 3.6.3 | Async http client/server framework (asyncio)
ase | 3.19.1 | Set of tools for atomistic simulations
astor | 0.8.1 | Read, rewrite, and write Python ASTs nicely
async-timeout | 3.0.1 | Timeout context manager for asyncio programs
beautifulsoup4 | 4.9.3 | Python library designed for screen-scraping
cachetools | 4.1.1 | Extensible memoizing collections and decorators
catalogue | 1.0.0 | Super lightweight function registries for your library
cloudpickle | 1.6.0 | Extended pickling support for Python objects
cudatoolkit | 10.1.243 | NVIDIA CUDA toolkit
cudnn | 7.6.5 | NVIDIA's cuDNN deep neural network acceleration library
cymem | 2.0.4 | Manage calls to calloc/free through Cython
cython | 0.29.21 | The Cython compiler for writing C extensions for the Python language
cython-blis | 0.4.1 | Fast matrix-multiplication as a self-contained Python library – no system dependencies!
cytoolz | 0.11.0 | Cython implementation of Toolz. High performance functional utilities
dask-core | 2.30.0 | Parallel Python with task scheduling
deep-learning-essentials | 2.7 | A collection of the essential packages to work with deep learning packages and ArcGIS Pro.
fastai | 1.0.60 | fastai makes deep learning with PyTorch faster, more accurate, and easier
fastprogress | 0.2.3 | A fast and simple progress bar for Jupyter Notebook and console.
fasttext | 0.9.2 | fastText - Library for efficient text classification and representation learning
filelock | 3.0.12 | A platform independent file lock.
gast | 0.2.2 | Python AST that abstracts the underlying Python version
google-auth | 1.23.0 | Google authentication library for Python
google-auth-oauthlib | 0.4.2 | Google Authentication Library, oauthlib integration with google-auth
google-pasta | 0.2.0 | pasta is an AST-based Python refactoring library
googledrivedownloader | 0.4 | Minimal class to download shared files from Google Drive.
graphviz | 2.38 | Open Source graph visualization software.
grpcio | 1.31.0 | HTTP/2-based RPC framework
imageio | 2.8.0 | A Python library for reading and writing image data
isodate | 0.6.0 | An ISO 8601 date/time/duration parser and formatter.
joblib | 0.17.0 | Lightweight pipelining: using Python functions as pipeline jobs.
keepalive | 0.5 | An HTTP handler for urllib that supports HTTP 1.1 and keepalive
keras-applications | 1.0.8 | Applications module of the Keras deep learning library.
keras-base | 2.3.1 |
keras-gpu | 2.3.1 | Deep Learning Library for Theano and TensorFlow
keras-preprocessing | 1.1.0 | Data preprocessing and data augmentation module of the Keras deep learning library
laspy | 1.7.0 | A Python library for reading, modifying and creating LAS files
libopencv | 4.5.0 | Computer vision and machine learning software library.
libprotobuf | 3.13.0.1 | Protocol Buffers - Google's data interchange format. C++ Libraries and protoc, the protobuf compiler.
libwebp | 1.1.0 | WebP image library
llvmlite | 0.34.0 | A lightweight LLVM python binding for writing JIT compilers.
markdown | 3.3.3 | Python implementation of Markdown.
multidict | 4.7.6 | multidict implementation
murmurhash | 1.0.2 | Cython bindings for MurmurHash2
ninja | 1.10.1 | A small build system with a focus on speed
numba | 0.51.2 | NumPy aware dynamic Python compiler using LLVM
nvidia-ml-py3 | 7.352.0 | Python bindings to the NVIDIA Management Library
onnx | 1.7.0 | Open Neural Network Exchange library
onnx-tf | 1.5.0 | Experimental Tensorflow Backend for ONNX
opencv | 4.5.0 | Computer vision and machine learning software library.
opt_einsum | 3.1.0 | Optimizing einsum functions in NumPy, Tensorflow, Dask, and more with contraction order optimization.
plac | 1.1.0 | The smartest command line arguments parser in the world
plotly | 4.5.4 | An interactive JavaScript-based visualization library for Python
pooch | 1.0.0 | A friend to fetch your Python library's sample data files
preshed | 3.0.2 | Cython Hash Table for Pre-Hashed Keys
protobuf | 3.13.0.1 | Protocol Buffers - Google's data interchange format.
py-opencv | 4.5.0 | Computer vision and machine learning software library.
pyasn1 | 0.4.8 | ASN.1 types and codecs
pyasn1-modules | 0.2.8 | A collection of ASN.1-based protocols modules.
pytorch | 1.4.0 | PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.
pywavelets | 1.1.1 | Discrete Wavelet Transforms in Python
rdflib | 5.0.0 | Library for working with RDF, a simple yet powerful language for representing information.
retrying | 1.3.3 | Simplify the task of adding retry behavior to just about anything.
rsa | 4.6 | Pure-Python RSA implementation
sacremoses | 0.0.43 | SacreMoses
scikit-image | 0.17.2 | Image processing routines for SciPy
scikit-learn | 0.23.2 | A set of python modules for machine learning and data mining
sentencepiece | 0.1.91 | SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training.
soupsieve | 2.0.1 | A modern CSS selector implementation for BeautifulSoup
spacy | 2.2.4 | Industrial-strength Natural Language Processing
sparqlwrapper | 1.8.5 | SPARQL Endpoint interface to Python for use with rdflib
srsly | 1.0.2 | Modern high-performance serialization utilities for Python
tensorboard | 2.3.0 | TensorBoard lets you watch Tensors Flow
tensorboard-plugin-wit | 1.6.0 | What-If Tool TensorBoard plugin
tensorboardx | 2.1 | Tensorboard for PyTorch.
tensorflow | 2.1.0 | TensorFlow is a machine learning library.
tensorflow-addons | 0.9.1 | Useful extra functionality for TensorFlow 2.x
tensorflow-base | 2.1.0 | TensorFlow is a machine learning library, base GPU package, tensorflow only.
tensorflow-estimator | 2.1.0 | TensorFlow Estimator
tensorflow-gpu | 2.1.0 | Metapackage for selecting a TensorFlow variant.
termcolor | 1.1.0 | ANSII Color formatting for output in terminal.
thinc | 7.4.0 | Learn super-sparse multi-class models
threadpoolctl | 2.1.0 | Python helpers to control the threadpools of native libraries
tifffile | 2020.10.1 | Read and write image data from and to TIFF files.
tokenizers | 0.8.1 | Fast State-of-the-Art Tokenizers optimized for Research and Production
toolz | 0.11.1 | A functional standard library for Python
torch-cluster | 1.5.4 | Extension library of highly optimized graph cluster algorithms for use in PyTorch
torch-geometric | 1.5.0 | Geometric deep learning extension library for PyTorch
torch-scatter | 2.0.4 | Extension library of highly optimized sparse update (scatter and segment) operations
torch-sparse | 0.6.1 | Extension library of optimized sparse matrix operations with autograd support
torch-spline-conv | 1.2.0 | PyTorch implementation of the spline-based convolution operator of SplineCNN
torchvision | 0.5.0 | image and video datasets and models for torch deep learning
tqdm | 4.51.0 | A Fast, Extensible Progress Meter
transformers | 3.3.0 | State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch
typeguard | 2.7.0 | Runtime type checker for Python
wasabi | 0.6.0 | A lightweight console printing and formatting toolkit
werkzeug | 0.16.1 | The comprehensive WSGI web application library.
yarl | 1.6.2 | Yet another URL library

Additional Notes
----------------

 - Though this package distributes the GPU based versions of packages, CPU versions can still be installed and used on any machine Pro supports. To install TensorFlow for the CPU, from the Python backstage you can install the `tensorflow-mkl` package to get a CPU only version.
 - This installer adds packages to the default `arcgispro-py3` environment. Any subsequent clones of that environment will also include this full collection of packages. This collection of packages is validated and tested against the version of Pro is installed alongside, and upgrades of Pro will also require reinstallation of the deep learning libraries. Note that when you upgrade the software to a new release, you'll need to uninstall the Deep Learning Libraries installation as well as Pro or Server, and reinstall the new version of this package for that release.
 - This installer is only available for ArcGIS Pro 2.6+, and ArcGIS Server 10.8.1+ -- for earlier releases, you'll need to follow the documentation for that release on installing the packages through the Python backstage or Python command prompt.
 - If you want these packages for a specific environment only, you can install the `deep-learning-essentials` package which has the same list of dependencies as a standalone conda metapackage.
