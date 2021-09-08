//
// Created by James Noeckel on 10/28/20.
//

#pragma once
#include "geometry/shapes2/CombinedCurve.h"
#include <Eigen/Dense>
#include <string>
#include "opencv2/opencv.hpp"

Eigen::Vector2d getDims(const CombinedCurve &curve);

double computeScale(const CombinedCurve &curve, Eigen::Vector2d &minPt);

void draw_curve(cv::Mat &img, const CombinedCurve &curve, double scale, const Eigen::Vector2d &minPt, int thickness=3, const cv::Scalar &color=cv::Scalar(0, 255, 0));

cv::Mat display_curve(const std::string &name, const CombinedCurve &curve, double scale, const Eigen::Vector2d &minPt, int thickness=3, const cv::Scalar &color=cv::Scalar(0, 255, 0), bool display=false);

void display_curvatures(const std::string &name, const Eigen::MatrixX2d &d, int ksize, bool useEndpoints, bool save=true);

void display_fit(const std::string & name, const CombinedCurve &curve, const Eigen::MatrixX2d &d, int ksize=-1, bool save=true, const std::vector<LineConstraint> &edges={}, double curveThreshold=0, const std::vector<Eigen::MatrixX2d> &neighbors={}, const Eigen::Vector2d &minDir=Eigen::Vector2d(0, 0));

std::unique_ptr<Curve> test_bezier(const std::string &name, Eigen::MatrixX2d &d, int first, int last, int degree=3, const Eigen::Vector2d &leftTangent=Eigen::Vector2d(0, 0), const Eigen::Vector2d &rightTangent=Eigen::Vector2d(0, 0));

CombinedCurve test_fit(const std::string &name, Eigen::MatrixX2d &d, double bezier_cost, double line_cost, int knots=-1, int first=0, int last=-1, int ksize=5, const std::vector<LineConstraint> &edges={}, double curveThreshold=0, const std::vector<Eigen::MatrixX2d> &neighbors={});