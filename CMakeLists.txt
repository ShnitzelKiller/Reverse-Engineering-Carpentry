CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
SET(CMAKE_CXX_STANDARD 17)
PROJECT(StockDetection)

SET( BUILD_CLOUDTOOLS false CACHE BOOL "build extra point cloud utilities" )

# SET(PCL_DIR "/projects/grail/jamesn8/Dependencies/yummed/pcl/usr/lib64/cmake/pcl/PCLConfig.cmake")

set(Boost_USE_MULTITHREADED TRUE)
FIND_PACKAGE( Boost REQUIRED COMPONENTS thread graph system)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

IF (BUILD_CLOUDTOOLS)
  FIND_PACKAGE( PCL 1.9 REQUIRED COMPONENTS IO COMMON KDTREE SEGMENTATION FILTERS )
  INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS} )
  LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
  ADD_DEFINITIONS(${PCL_DEFINITIONS})
ENDIF()

# CGAL and its components
FIND_PACKAGE( CGAL REQUIRED )

INCLUDE_DIRECTORIES( src )

FIND_PACKAGE( EIGEN3 3.3 REQUIRED NO_MODULE )

INCLUDE_DIRECTORIES( ${CGAL_INCLUDE_DIRS} )

INCLUDE( /usr/local/lib/cmake/pagmo/FindEigen3.cmake )
FIND_PACKAGE( Pagmo REQUIRED )
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIRS} )

# IGL for visualization
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option( LIBIGL_WITH_OPENGL            "Use OpenGL"         ON )
option( LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON )
option( LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          ON )
option( LIBIGL_WITH_TRIANGLE           "Use Triangle"       ON )
option(LIBIGL_WITH_PNG               "Use PNG"                      OFF)

include(libigl)

#find_package( Ceres REQUIRED )
find_package( OpenCV REQUIRED )
#find_package( gflags REQUIRED )
find_package( OpenMP REQUIRED )

ADD_SUBDIRECTORY(maxflow)
INCLUDE_DIRECTORIES(maxflow)

ADD_LIBRARY(
        CarpentryLib
        src/geometry/cgal/geom.h
        src/geometry/cgal/efficient_ransac.cpp
        src/geometry/cgal/min_bbox.cpp
        src/geometry/cgal/convex_hull.cpp
        src/geometry/cgal/polygon_triangulation.cpp
        src/geometry/primitives3/compute_basis.cpp
        src/geometry/primitives3/compute_basis.h
        src/geometry/fast_winding_number.cpp
        src/geometry/fast_winding_number.h
        src/utils/typedefs.hpp
        src/utils/sorted_data_structures.hpp
        src/reconstruction/CameraIntrinsics.cpp
        src/reconstruction/CameraIntrinsics.h
        src/reconstruction/Image.cpp
        src/reconstruction/Image.h
        src/reconstruction/Points3D.cpp
        src/reconstruction/Points3D.h
        src/math/integration.cpp
        src/math/integration.h
        src/math/robust_derivative.hpp
        src/math/GaussianMixture.cpp
        src/math/GaussianMixture.h
        src/math/poly34.cpp
        src/math/poly34.h
        src/utils/IntervalTree.h
        src/reconstruction/point_cloud_io.cpp
        src/reconstruction/point_cloud_io.h
        src/reconstruction/ReconstructionData.cpp
        src/reconstruction/ReconstructionData.h
        src/reconstruction/mesh_io.cpp
        src/reconstruction/mesh_io.h
        src/construction/FeatureExtractor.cpp
        src/construction/FeatureExtractor.h
        src/geometry/shapes2/Primitive.cpp
        src/geometry/shapes2/Primitive.h
        src/geometry/primitives3/BoundedPlane.cpp
        src/geometry/primitives3/BoundedPlane.h
        src/geometry/primitives3/Cylinder.cpp
        src/geometry/primitives3/Cylinder.h
        src/geometry/shapes2/VoxelGrid.cpp
        src/geometry/shapes2/VoxelGrid.hpp
        src/utils/top_k_indices.hpp
        src/geometry/shapes2/graphicsgems/FitCurves.cpp
        src/geometry/shapes2/graphicsgems/FitCurves.h
        src/geometry/shapes2/Curve.cpp
        src/geometry/shapes2/Curve.h
        src/geometry/shapes2/CombinedCurve.cpp
        src/geometry/shapes2/CombinedCurve.h
        src/geometry/shapes2/find_adjacent_ranges.cpp
        src/geometry/shapes2/find_adjacent_ranges.h
        src/geometry/shapes2/Ray2d.cpp
        src/geometry/shapes2/Ray2d.h
        src/construction/Construction.cpp
        src/construction/Construction.h
        src/construction/Solver.cpp
        src/construction/SolverInitialize.cpp
        src/construction/Solver.h
        src/utils/settings.cpp
        src/utils/settings.h
        src/geometry/shapes2/Segment2d.cpp
        src/geometry/shapes2/Segment2d.h
        src/math/fields/ScalarField.h
        src/math/fields/WindingNumberField3D.cpp
        src/math/fields/WindingNumberField3D.h
        src/math/fields/FieldSlice.cpp
        src/math/fields/FieldSlice.h
        src/math/fields/PointDensityField.cpp
        src/math/fields/PointDensityField.h
        src/geometry/WindingNumberData.h
        src/geometry/primitives3/intersect_planes.cpp
        src/geometry/primitives3/intersect_planes.h
        src/math/RunningAverage.cpp
        src/math/RunningAverage.h
        src/construction/optimization/alignmentProblem.cpp
        src/construction/optimization/alignmentProblem.hpp
        src/construction/optimization/test_problem.cpp
        src/construction/optimization/test_problem.h
        src/construction/optimization/simulated_annealing.cpp
        src/construction/optimization/simulated_annealing.hpp
        src/construction/optimization/discreteMultiObjective.cpp
        src/construction/optimization/discreteSingleObjective.cpp
        src/construction/optimization/maxOverlap.cpp
        src/construction/optimization/discreteProblems.hpp
        src/imgproc/multilabel/GCoptimization.cpp
        src/imgproc/multilabel/GCoptimization.h
        src/imgproc/multilabel/graph.cpp
        src/imgproc/multilabel/graph.h
        src/imgproc/multilabel/LinkedBlockList.cpp
        src/imgproc/multilabel/LinkedBlockList.h
        src/imgproc/multilabel/maxflow.cpp
        src/imgproc/multilabel_graph_cut.cpp
        src/imgproc/multilabel_graph_cut.h
        src/imgproc/display_segmentation.cpp
        src/imgproc/display_segmentation.h
#        src/imgproc/DepthSegmentation.cpp
#        src/imgproc/DepthSegmentation.h
        src/utils/visualization.cpp
        src/utils/visualization.hpp
        src/geometry/homographyFromPlane.cpp
        src/geometry/homographyFromPlane.h
        src/utils/eigenMatToCV.cpp
        src/utils/eigenMatToCV.h
#        src/imgproc/joint_segmentation.cpp
#        src/imgproc/joint_segmentation.h
        src/imgproc/matching_scores.cpp
        src/imgproc/matching_scores.h
        src/imgproc/graph_cut.cpp
        src/imgproc/graph_cut.h
        src/imgproc/histogram_matching.cpp
        src/imgproc/histogram_matching.h
        src/construction/joint_image_analysis.cpp
        src/imgproc/dda_foreach.h
        src/geometry/primitives3/SurfaceCompletion.cpp
        src/geometry/primitives3/SurfaceCompletion.h
        src/test/testUtils/displaySurfaceCompletion.cpp
        src/utils/io/csvFormat.h
        src/geometry/find_affine.cpp
        src/geometry/find_affine.h
        src/utils/vstack.h
        src/utils/io/csvFormat.cpp
        src/utils/sign.h
        src/utils/colorAtIndex.h
        src/utils/colorAtIndex.cpp
        src/geometry/primitives3/MultiRay3d.h
        src/geometry/primitives3/MultiRay3d.cpp
        src/geometry/csg1d.h
        src/geometry/shapes2/primitive_thickness.cpp
        src/geometry/shapes2/primitive_thickness.h
        src/geometry/primitives3/Ray3d.h
        src/geometry/shapes2/polygon_thickness.cpp
        src/geometry/shapes2/polygon_thickness.h
        src/test/testUtils/curveFitUtils.cpp
        src/test/testUtils/curveFitUtils.h
        src/utils/mesh_valid.cpp
        src/utils/mesh_valid.h
        src/construction/Constraints.h
        src/geometry/shapes2/convert_contour.cpp
        src/geometry/shapes2/convert_contour.h
        src/geometry/shapes2/density_contour.cpp
        src/geometry/shapes2/density_contour.h
        src/utils/macros.h
        src/construction/ConstructionIO.cpp
        src/geometry/primitives3/Multimesh.cpp
        src/geometry/primitives3/Multimesh.h)


TARGET_LINK_LIBRARIES( CarpentryLib Eigen3::eigen3 ${Boost_LIBRARIES} CGAL::CGAL igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::triangle ${OpenCV_LIBS} Pagmo::pagmo maxflow OpenMP::OpenMP_CXX)

ADD_EXECUTABLE( CarpentryOptimization src/applications/carpentry_optimization/main.cpp )
#ADD_EXECUTABLE( VisualizeSolution src/applications/carpentry_optimization/visualizeSolution.cpp )
TARGET_LINK_LIBRARIES( CarpentryOptimization CarpentryLib )
#TARGET_LINK_LIBRARIES( VisualizeSolution CarpentryLib )

#ADD_EXECUTABLE( StockSegment src/applications/stock-seg/main.cpp )
#TARGET_LINK_LIBRARIES( StockSegment CarpentryLib )

IF (BUILD_CLOUDTOOLS)
  ADD_EXECUTABLE(ConvertXYZtoPLY src/applications/cloudTools/xyz2ply.cpp)
  ADD_EXECUTABLE(Noisify src/applications/cloudTools/noisify.cpp src/math/NormalRandomVariable.cpp src/math/NormalRandomVariable.h)
  TARGET_LINK_LIBRARIES( ConvertXYZtoPLY ${PCL_LIBRARIES} CarpentryLib)
  TARGET_LINK_LIBRARIES( Noisify ${PCL_LIBRARIES} CarpentryLib)
ENDIF()

SET( BUILD_TEST true CACHE BOOL "build tests" )
SET( BUILD_PLANE false CACHE BOOL "build plane app" )
SET( BUILD_EDGE false CACHE BOOL "build edges app" )

IF (BUILD_TEST)
  ADD_EXECUTABLE( TestIntegrate src/test/test_integrate.cpp )
  TARGET_LINK_LIBRARIES( TestIntegrate CarpentryLib )
  ADD_EXECUTABLE( TestLoadReconstruction src/test/test_load_reconstruction.cpp )
  TARGET_LINK_LIBRARIES( TestLoadReconstruction CarpentryLib )
  ADD_EXECUTABLE( TestGMM src/test/test_gmm.cpp src/math/GaussianMixture.h src/math/GaussianMixture.cpp src/math/NormalRandomVariable.cpp src/math/NormalRandomVariable.h)
  TARGET_LINK_LIBRARIES( TestGMM ${OpenCV_LIBS} )
  ADD_EXECUTABLE( EnergyMinimizationExample src/imgproc/multilabel/example.cpp src/imgproc/multilabel/GCoptimization.cpp src/imgproc/multilabel/GCoptimization.h src/imgproc/multilabel/graph.cpp src/imgproc/multilabel/graph.h src/imgproc/multilabel/LinkedBlockList.cpp src/imgproc/multilabel/LinkedBlockList.h src/imgproc/multilabel/maxflow.cpp )
  ADD_EXECUTABLE( TestSegmentation src/test/test_multi_segmentation.cpp )
  TARGET_LINK_LIBRARIES( TestSegmentation CarpentryLib )
  ADD_EXECUTABLE( TestPolygon src/test/test_polygon.cpp )
  TARGET_LINK_LIBRARIES( TestPolygon CarpentryLib )
  ADD_EXECUTABLE( TestIntervalTree src/test/test_interval_tree.cpp src/utils/IntervalTree.h)
#  ADD_EXECUTABLE( TestMeshInterior src/test/test_mesh_interior.cpp )
#  TARGET_LINK_LIBRARIES( TestMeshInterior CarpentryLib )
  ADD_EXECUTABLE( TestMarchingSquares src/test/test_marching_squares.cpp )
  TARGET_LINK_LIBRARIES( TestMarchingSquares CarpentryLib )
  ADD_EXECUTABLE( TestUtils src/test/test_utils.cpp src/utils/sorted_data_structures.hpp src/utils/top_k_indices.hpp)
  ADD_EXECUTABLE( TestMath src/test/test_math.cpp src/math/robust_derivative.hpp src/geometry/find_affine.cpp)
  ADD_EXECUTABLE( TestCurveFit src/test/test_curve_fit.cpp)
  TARGET_LINK_LIBRARIES( TestCurveFit CarpentryLib )
  ADD_EXECUTABLE( TestTriangulate src/test/test_triangulate.cpp )
  TARGET_LINK_LIBRARIES( TestTriangulate igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::triangle )
  ADD_EXECUTABLE( TestScalarField src/test/test_scalar_field.cpp src/math/fields/FieldSlice.cpp)
  ADD_EXECUTABLE( TestMatching src/test/test_matching.cpp )
  TARGET_LINK_LIBRARIES( TestMatching CarpentryLib )
  ADD_EXECUTABLE( TestGraphCut src/test/test_graph_cut.cpp )
  TARGET_LINK_LIBRARIES( TestGraphCut CarpentryLib)
  ADD_EXECUTABLE( TestGraphCut8way src/test/testGraphCut8way.cpp )
  TARGET_LINK_LIBRARIES( TestGraphCut8way CarpentryLib )
  ADD_EXECUTABLE( TestHistogramMatching src/test/test_histogram_matching.cpp )
  TARGET_LINK_LIBRARIES( TestHistogramMatching CarpentryLib )
  ADD_EXECUTABLE( TestDDA src/test/test_dda.cpp)
  TARGET_LINK_LIBRARIES( TestDDA ${OpenCV_LIBS} igl::core igl::opengl_glfw igl::opengl_glfw_imgui)
  ADD_EXECUTABLE( TestDisparity src/test/test_disparities.cpp )
  TARGET_LINK_LIBRARIES( TestDisparity CarpentryLib )
  #ADD_EXECUTABLE( TestGeoCut src/test/test_geocut.cpp )
  #TARGET_LINK_LIBRARIES( TestGeoCut CarpentryLib )
ENDIF()

IF (BUILD_PLANE)
  ADD_EXECUTABLE( PlaneSegment src/applications/plane-seg/main.cpp )
  TARGET_LINK_LIBRARIES( PlaneSegment ${PCL_LIBRARIES} )
ENDIF()

IF (BUILD_EDGE)
  ADD_EXECUTABLE( EdgeFinder src/applications/region-edges/main.cpp )
  TARGET_LINK_LIBRARIES( EdgeFinder ${PCL_LIBRARIES} )
ENDIF()

ADD_EXECUTABLE( ExportColmap src/applications/exportToColmap.cpp )
TARGET_LINK_LIBRARIES( ExportColmap CarpentryLib )