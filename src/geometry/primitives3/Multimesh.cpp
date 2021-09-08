//
// Created by James Noeckel on 4/22/21.
//

#include "Multimesh.h"

void Multimesh::AddMesh(std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i> &&mesh) {
    meshes.push_back(std::move(mesh));
}

void Multimesh::AddMesh(const std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i> &mesh) {
    meshes.push_back(mesh);
}

std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i> Multimesh::GetTotalMesh() const {
    size_t totalVSize = 0;
    size_t totalFSize = 0;
    for (const auto &mesh : meshes) {
        totalVSize += mesh.first.rows();
        totalFSize += mesh.second.rows();
    }
    size_t VOffset = 0;
    size_t FOffset = 0;
    Eigen::MatrixX3d V(totalVSize, 3);
    Eigen::MatrixX3i F(totalFSize, 3);
    for (const auto &mesh : meshes) {
        V.block(VOffset, 0, mesh.first.rows(), 3) = mesh.first;
        F.block(FOffset, 0, mesh.second.rows(), 3) = mesh.second.array() + VOffset;
        VOffset += mesh.first.rows();
        FOffset += mesh.second.rows();
    }
    return {V, F};
}