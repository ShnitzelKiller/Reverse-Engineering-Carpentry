//
// Created by James Noeckel on 11/3/20.
//

#include "find_affine.h"
#include <iostream>

using namespace Eigen;
void find_affine(const Ref<const MatrixX3d> &P, const Ref<const MatrixX3d> &Q, double &scale, Vector3d &trans, Quaterniond &rot) {
    RowVector3d Pmean = P.colwise().mean();
    RowVector3d Qmean = Q.colwise().mean();
    MatrixX3d Pdiff = P.rowwise() - Pmean;
    MatrixX3d Qdiff = Q.rowwise() - Qmean;
    Matrix3d S = Pdiff.transpose() * Qdiff;
//    std::cout << "S:\n" << S << std::endl;
    JacobiSVD svd(S, ComputeFullU | ComputeFullV);
    Vector3d c(0, 0, 0);
    for (int j=0; j<P.rows(); ++j) {
        c += ((Pdiff.row(j) * svd.matrixV()).array() * (Qdiff.row(j) * svd.matrixV()).array()).matrix();
    }

    Vector3i a;
    for (int i=0; i<3; ++i) {
        if (c(i) < 0) {
            a(i) = -1;
        } else if (c(i) > 0) {
            a(i) = 1;
        } else {
            a(i) = 0;
        }
    }

    DiagonalMatrix<double, 3> E(a(0), a(1), a(2));
    Matrix3d W = svd.matrixV() * E * svd.matrixU().transpose();
//    std::cout << "W: \n" << W << std::endl;
//    std::cout << "W norms: " << W.colwise().norm() << std::endl;
    if (W.determinant() < 0) {
        std::cout << "warning: transform is not right handed!" << std::endl;
    }

    double rhs = 0.0;
    for (int j=0; j<P.rows(); ++j) {
        rhs += (Pdiff.row(j) * W).dot(Qdiff.row(j));
    }
    /*double lhs = 0.0;
    for (int j=0; j<P.rows(); ++j) {
        lhs += Pdiff.row(j).squaredNorm();
    }
    scale = rhs / lhs;*/
    scale = Qdiff.rowwise().norm().sum() / Pdiff.rowwise().norm().sum();
    rot = W;
    trans = (Qmean - scale * (Pmean * W.transpose())).transpose();
}