#pragma once

#include <pagmo/pagmo.hpp>
#include "utils/typedefs.hpp"
#include "construction/Construction.h"

using namespace pagmo;

/**
 * Degrees of freedom: plane i parameters map to 4i, 4i+1, 4i+2 in the normal vector and 4i+3 is the offset
 */
struct AlignmentProblem {
    AlignmentProblem() = default;
    AlignmentProblem(PointCloud3::Handle cloud, std::vector<double> offsets, std::vector<Eigen::Quaterniond> rotations, std::vector<std::pair<size_t, size_t>> connections, std::vector<size_t> blockToClusterID, const std::vector<std::vector<int>> &clusters, double maxOffset, double offsetScale, double objectiveScale, int stride);

    /**
     * [obj, orthogonality constraints, unit vector constraints]
     */
    vector_double fitness(const vector_double &params) const;
    std::pair<vector_double, vector_double> get_bounds() const;
    vector_double::size_type get_nobj() const;
    vector_double::size_type get_nec() const;
    vector_double::size_type get_nic() const;
    bool has_gradient() const;
    vector_double gradient(const vector_double &params) const;
    bool has_gradient_sparsity() const;

    bool has_hessians() const;
    std::vector<vector_double> hessians(const vector_double &) const;
    bool has_hessians_sparsity() const;
    std::vector<sparsity_pattern> hessians_sparsity() const;
    std::string get_name() const;
    /**
    * 2nd order Taylor expansion of the unit vector $\hat{z}$
    */
    static Eigen::Vector3d unitvec(double x, double y);
private:
    size_t dim() const;
    PointCloud3::Handle cloud_;
    /** orthogonal pairs of parts */
    std::vector<std::pair<size_t, size_t>> orthogonalities_;
    const std::vector<std::vector<int>> *clusters_;
    int stride_;
    std::vector<size_t> dimToCluster;
    std::vector<double> baseOffsets;
    std::vector<Eigen::Quaterniond> baseRotations;
    double maxOffset_;
    double offsetScale_;
    double objectiveScale_;
};