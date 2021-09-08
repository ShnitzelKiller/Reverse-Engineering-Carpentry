//
// Created by James Noeckel on 10/21/20.
//

#include "displaySurfaceCompletion.h"

void displaySegmentation(SurfaceCompletion &surf, Visualizer &vis, int mode, bool use_distfield) {
    using namespace Eigen;
    Array3i resolution = surf.resolution();
    int N = resolution.prod();
    VectorXd field(N);
    MatrixX3d gridPoints(resolution.prod(), 3);
    int gridInd = 0;
    for (int i=0; i<resolution.x(); ++i) {
        for (int j=0; j<resolution.y(); ++j) {
            for (int k=0; k<resolution.z(); ++k, ++gridInd) {
                Vector3d point = surf.minPt() + surf.spacing() * Vector3d(i, j, k);
                gridPoints.row(gridInd) = point.transpose();
            }
        }
    }
    if (mode == VIS_SURFACE) {
        if (use_distfield) {
            const auto distanceFun = surf.distfun();
            for (size_t i = 0; i < N; ++i) {
                field(i) = -distanceFun[i];
            }
        } else {
            for (size_t i=0; i<N; ++i) {
                field(i) = surf.getSegmentation()[i] ? 1 : -1;
            }
        }
        MatrixX3d V(0, 3);
        MatrixX3i F(0, 3);
        igl::copyleft::marching_cubes(field, gridPoints, resolution.z(), resolution.y(), resolution.x(),
                                      0.0, V, F);
        for (int i=0; i<F.rows(); ++i) {
            std::swap(F(i, 0), F(i, 1));
        }
        vis.visualize_mesh(V, F);
        vis.align_camera(V);
    } else {
        /*if (mode == VIS_INSIDE) {
            field = VectorXd::Ones(N);
            for (int ind : surf.insideConstraints()) {
                field[ind] = -1;
            }
        } else if (mode == VIS_OUTSIDE) {
            field = -VectorXd::Ones(N);
            for (int ind : surf.outsideConstraints()) {
                field[ind] = 1;
            }
        }
        MatrixX3d V(0, 3);
        MatrixX3i F(0, 3);
        igl::copyleft::marching_cubes(field, gridPoints, resolution.z(), resolution.y(), resolution.x(),
                                      0.0, V, F);
        vis.visualize_mesh(V, F);
        vis.align_camera(V);*/
        PointCloud3 cloud;
        if (mode == VIS_INSIDE) {
            cloud.P.resize(surf.insideConstraints().size(), 3);
            for (int p=0; p<surf.insideConstraints().size(); ++p) {
                Vector3d pos = surf.getPosition(surf.insideConstraints()[p]);
                cloud.P.row(p) = pos.transpose();
            }
        } else if (mode == VIS_OUTSIDE) {
            cloud.P.resize(surf.outsideConstraints().size(), 3);
            for (int p=0; p<surf.outsideConstraints().size(); ++p) {
                Vector3d pos = surf.getPosition(surf.outsideConstraints()[p]);
                cloud.P.row(p) = pos.transpose();
            }
        }
        if (cloud.P.rows() > 0) {
            vis.visualize_points(cloud);
        }
    }

}