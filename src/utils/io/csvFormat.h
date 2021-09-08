//
// Created by James Noeckel on 10/24/20.
//
#pragma once

#include <Eigen/Dense>
#include <string>

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

Eigen::MatrixXd openData(const std::string &fileToOpen);