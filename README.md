# Carpentry Shape Detection
## Overview
![](teaser3_sm.jpg)
This tool reverse engineers carpentered objects from multi-view imagery, along with a point cloud created from these images using appropriate software for reconstruction.

## Building and installation instructions
This project depends on OpenCV, LibIGL, CGAL, Pagmo, and [MAXFLOW](http://pub.ist.ac.at/~vnk/software.html). OpenMP is also preferred. To build, run
For MAXFLOW, place the "maxflow" folder into the project root directory.
```sh
mkdir build
cd build
cmake .. -DLIBIGL_INCLUDE_DIR=<path to /libigl/include> -DCMAKE_BUILD_TYPE=Release
make CarpentryOptimization
```

## Running
Run the `CarpentryOptimization` executable with a path to a configuration file as an argumment (see `examples/customstool.txt` as an example). The data currently needs to be downloaded separately.
The configuration file contains many parameters that affect the algorithm. Additionally, you may choose whether to obtain interactive 3D visualization of results at various stages of the pipeline by setting `visualization=1`.

The important settings for running with your own data are:
- points_filename: path to the point cloud data
- reconstruction_path: path to the associated reconstruction data containing the camera pose information for each view (this is assumed to be in the [bundler](https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html) format)
- result_path: where to save results