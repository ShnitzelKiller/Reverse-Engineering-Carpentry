//
// Created by James Noeckel on 3/26/20.
//
#include <string>
#include <Eigen/Dense>

#pragma once
bool load_mesh(const std::string &filename, Eigen::MatrixX3d &V, Eigen::MatrixX3i &F);