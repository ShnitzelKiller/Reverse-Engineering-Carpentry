//
// Created by James Noeckel on 4/22/21.
//

#pragma once

#include <Eigen/Dense>
#include <vector>

class Multimesh {
public:
    void AddMesh(std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i> &&mesh);
    void AddMesh(const std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i> &mesh);
    std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i> GetTotalMesh() const;
private:
    std::vector<std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i>> meshes;
};

