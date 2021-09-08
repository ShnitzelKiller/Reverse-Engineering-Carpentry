//
// Created by James Noeckel on 9/10/20.
//

#include "alignmentProblem.hpp"

using namespace Eigen;

AlignmentProblem::AlignmentProblem(PointCloud3::Handle cloud, std::vector<double> offsets, std::vector<Quaterniond> rotations, std::vector<std::pair<size_t, size_t>> connections,
                                   std::vector<size_t> blockToClusterID,
                                   const std::vector<std::vector<int>> &clusters, double maxOffset, double offsetScale,
                                   double objectiveScale, int stride)
        : cloud_(std::move(cloud)), baseOffsets(std::move(offsets)), baseRotations(std::move(rotations)), orthogonalities_(std::move(connections)), dimToCluster(std::move(blockToClusterID)),
        clusters_(&clusters), stride_(stride), maxOffset_(maxOffset), offsetScale_(offsetScale),
        objectiveScale_(objectiveScale)
{

}

size_t AlignmentProblem::dim() const {
    return dimToCluster.size()*3;
}

std::pair<vector_double, vector_double> AlignmentProblem::get_bounds() const {
    size_t nDim = dim();
    vector_double minPt(nDim);
    vector_double maxPt(nDim);
    for (size_t i=0; i<dimToCluster.size(); ++i) {
        minPt[i*3] = -0.5;
        minPt[i*3+1] = -0.5;
        minPt[i*3+2] = -maxOffset_/offsetScale_;

        maxPt[i*3] = 0.5;
        maxPt[i*3+1] = 0.5;
        maxPt[i*3+2] = maxOffset_/offsetScale_;

    }
    return {minPt, maxPt};
}

vector_double::size_type AlignmentProblem::get_nobj() const { return 1; }

vector_double::size_type AlignmentProblem::get_nec() const {
    // orthogonality constraints
    return orthogonalities_.size();
}

vector_double::size_type AlignmentProblem::get_nic() const { return 0; }

Vector3d AlignmentProblem::unitvec(double x, double y) {
    return Vector3d(x, y, 1-0.5*x*x-0.5*y*y);
}

vector_double AlignmentProblem::fitness(const vector_double &params) const {
    double totalError = 0.0;
    vector_double output(1 + get_nec());
    for (size_t i=0; i<dimToCluster.size(); ++i) {
        double x = params[i*3];
        double y = params[i*3+1];
        double offset = params[i*3+2];
        Vector3d n = baseRotations[i] * unitvec(x, y);
        for (int r=0; r<(*clusters_)[dimToCluster[i]].size(); r+=stride_) {
            int row = (*clusters_)[dimToCluster[i]][r];
            double dist = n.dot(cloud_->P.row(row)) + offsetScale_ * offset + baseOffsets[i];
            totalError += dist*dist;
        }
    }
    output[0] = totalError * objectiveScale_;
    //orthogonality constraints
    for (size_t i=0; i<orthogonalities_.size(); ++i) {
        size_t i1 = orthogonalities_[i].first;
        size_t i2 = orthogonalities_[i].second;
        Vector3d n1 = baseRotations[i1] * unitvec(params[i1*3], params[i1*3+1]);
        Vector3d n2 = baseRotations[i2] * unitvec(params[i2*3], params[i2*3+1]);
        output[i+1] = n1.dot(n2);
    }
    return output;
}

bool AlignmentProblem::has_gradient() const { return true; }

vector_double AlignmentProblem::gradient(const vector_double &params) const {
    size_t nDims = dim();
    size_t nFunc = 1 + get_nec();
    double objectiveScale2 = 2 * objectiveScale_;
    // [df1/dx1 df1/dx2 ... df2/dx1 df2/dx2 ... ]
    vector_double grad(nDims * nFunc, 0.0);
    //objective function gradient
    for (size_t i=0; i<dimToCluster.size(); ++i) {
        double x = params[i*3];
        double y = params[i*3+1];
        double offset = params[i*3+2];
        Vector3d n = baseRotations[i] * unitvec(x, y);
        for (int r=0; r<(*clusters_)[dimToCluster[i]].size(); r+=stride_) {
            int row = (*clusters_)[dimToCluster[i]][r];
            double dist = n.dot(cloud_->P.row(row)) + offsetScale_ * offset + baseOffsets[i];
            Vector3d part = dist * cloud_->P.row(row).transpose();
            grad[i*3] += part.dot(baseRotations[i] * Vector3d(1, 0, -x));
            grad[i*3+1] += part.dot(baseRotations[i] * Vector3d(0, 1, -y));
            grad[i*3+2] += dist;
        }
        grad[i*3] *= objectiveScale2;
        grad[i*3+1] *= objectiveScale2;
        grad[i*3+2] *= offsetScale_ * objectiveScale2;
    }
    //orthogonality constraint derivatives
    for (size_t i=0; i<orthogonalities_.size(); ++i) {
        size_t offset = (i+1) * nDims;
        size_t i1 = orthogonalities_[i].first;
        size_t i2 = orthogonalities_[i].second;
        Vector3d n1 = baseRotations[i1] * unitvec(params[i1*3], params[i1*3+1]);
        Vector3d n2 = baseRotations[i2] * unitvec(params[i2*3], params[i2*3+1]);
        grad[offset + i1*3] = n2.dot(baseRotations[i1] * Vector3d(1, 0, -params[i1*3]));
        grad[offset + i1*3+1] = n2.dot(baseRotations[i1] * Vector3d(0, 1, -params[i1*3+1]));
        grad[offset + i2*3] = n1.dot(baseRotations[i2] * Vector3d(1, 0, -params[i2*3]));
        grad[offset + i2*3+1] = n1.dot(baseRotations[i2] * Vector3d(0, 1, -params[i2*3+1]));
    }
    return grad;
}

bool AlignmentProblem::has_gradient_sparsity() const { return false; }

std::string AlignmentProblem::get_name() const {
    return "plane alignment problem";
}

bool AlignmentProblem::has_hessians() const {
    return true;
}

std::vector<vector_double> AlignmentProblem::hessians(const vector_double &params) const {
    std::vector<vector_double> hessians(1+get_nec());
    hessians[0].resize(dimToCluster.size() * 6, 0.0);
    double objectiveScale2 = 2 * objectiveScale_;
    for (size_t d=0; d < dimToCluster.size(); ++d) {
        size_t numPts = (*clusters_)[dimToCluster[d]].size();
        double x = params[d*3];
        double y = params[d*3+1];
        double offset = params[d*3+2];
        Vector3d n = baseRotations[d] * unitvec(x, y);
        Matrix<double, 3, 2> D;
        D.col(0) = baseRotations[d] * Vector3d(1, 0, -x);
        D.col(1) = baseRotations[d] * Vector3d(0, 1, -y);
        //main hessian
        for (int r=0; r<numPts; r+=stride_) {
            int row = (*clusters_)[dimToCluster[d]][r];
            RowVector3d pt = cloud_->P.row(row);
            //normal and normal
            Vector2d g(pt.dot(D.col(0)),
                       pt.dot(D.col(1)));
            double dist = n.dot(cloud_->P.row(row)) + offsetScale_ * offset + baseOffsets[d];
            double h = dist * pt.dot(baseRotations[d] * Vector3d(0, 0, 1));
            size_t l=0;
            for (size_t i=0; i<2; ++i) {
                for (size_t j=0; j<=i; ++j) {
                    hessians[0][d*6 + l] += g(i) * g(j);
                    if (i == j) {
                        hessians[0][d*6 + l] -= h;
                    }
                    ++l;
                }
            }
            //cross terms
            for (size_t i=0; i<2; ++i) {
                hessians[0][d*6 + 3 + i] += pt.dot(D.col(i));
            }
        }
        //apply constant factors to accumulated normal block
        size_t l=0;
        for (size_t i=0; i<2; ++i) {
            for (size_t j=0; j<=i; ++j) {
                hessians[0][d*6 + l] *= objectiveScale2;
                ++l;
            }
        }
        //to accumulated cross terms
        for (size_t j=0; j<2; ++j) {
            hessians[0][d*6 + 3 + j] *= offsetScale_ * objectiveScale2;
        }
        //to do^2 element
        hessians[0][d*6+5] = static_cast<double>(numPts/stride_) * offsetScale_ * offsetScale_ * objectiveScale2;
    }
    //orthogonal constraint hessians
    for (size_t d=0; d < orthogonalities_.size(); ++d) {
        size_t orthConstraintIndex = d + 1;
        size_t i=orthogonalities_[d].first;
        size_t j=orthogonalities_[d].second;
        if (i < j) {
            std::swap(i, j);
        }
        double xi = params[i*3];
        double yi = params[i*3+1];
        double xj = params[j*3];
        double yj = params[j*3+1];
        Vector3d ni = baseRotations[i] * unitvec(xi, yi);
        Vector3d nj = baseRotations[j] * unitvec(xj, yj);
        Matrix<double, 3, 2> Dri;
        Dri.col(0) = baseRotations[i] * Vector3d(1, 0, -xi);
        Dri.col(1) = baseRotations[i] * Vector3d(0, 1, -yi);
        Matrix<double, 3, 2> Drj;
        Drj.col(0) = baseRotations[j] * Vector3d(1, 0, -xj);
        Drj.col(1) = baseRotations[j] * Vector3d(0, 1, -yj);
        Matrix2d crossblock = Dri.transpose() * Drj;
        hessians[orthConstraintIndex].reserve(8);

        //d^2f/dxj^2
        double df2dj2 = -(baseRotations[j].conjugate() * ni).z();
        double df2di2 = -(baseRotations[i].conjugate() * nj).z();

        hessians[orthConstraintIndex].push_back(df2dj2);
        hessians[orthConstraintIndex].push_back(df2dj2);
        //d^f/dxidxj
        hessians[orthConstraintIndex].push_back(crossblock(0, 0));
        hessians[orthConstraintIndex].push_back(crossblock(0, 1));
        //d^2f/dxi^2
        hessians[orthConstraintIndex].push_back(df2di2);
        //d^f/dxidxj
        hessians[orthConstraintIndex].push_back(crossblock(1, 0));
        hessians[orthConstraintIndex].push_back(crossblock(1, 1));
        //d^2f/dxi^2
        hessians[orthConstraintIndex].push_back(df2di2);
    }
    return hessians;
}

bool AlignmentProblem::has_hessians_sparsity() const {
    return true;
}

std::vector<sparsity_pattern> AlignmentProblem::hessians_sparsity() const {
    std::vector<sparsity_pattern> sparsity(1+get_nec());
    sparsity[0].reserve(dimToCluster.size() * 6);
    for (size_t d=0; d < dimToCluster.size(); ++d) {
        //main hessian
        for (size_t i=0; i<3; ++i) {
            for (size_t j=0; j<=i; ++j) {
                sparsity[0].emplace_back(i + 3*d, j + 3*d);
            }
        }
    }
    //orthogonal constraint hessians sparsity patterns
    for (size_t d=0; d < orthogonalities_.size(); ++d) {
        size_t orthConstraintIndex = d + 1;
        sparsity[orthConstraintIndex].reserve(8);
        size_t i=orthogonalities_[d].first;
        size_t j=orthogonalities_[d].second;
        if (i < j) {
            std::swap(i, j);
        }
        // => j<i
        //j,j strip
        sparsity[orthConstraintIndex].emplace_back(3*j, 3*j);
        sparsity[orthConstraintIndex].emplace_back(3*j+1, 3*j+1);

        //i,j block row 0
        sparsity[orthConstraintIndex].emplace_back(3*i, 3*j);
        sparsity[orthConstraintIndex].emplace_back(3*i, 3*j+1);

        //i,i strip 0
        sparsity[orthConstraintIndex].emplace_back(3*i, 3*i);

        //i,j block row 1
        sparsity[orthConstraintIndex].emplace_back(3*i+1, 3*j);
        sparsity[orthConstraintIndex].emplace_back(3*i+1, 3*j+1);

        //i,i strip 1
        sparsity[orthConstraintIndex].emplace_back(3*i+1, 3*i+1);
    }
    return sparsity;
}