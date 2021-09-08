//
// Created by James Noeckel on 4/14/20.
//

#pragma once
#include <Eigen/Dense>
#include <vector>

/*
 *  FitCurve :
 *  	Fit a Bezier curves to a set of digitized points using recursive splitting
 */
void FitCurve(const Eigen::Ref<Eigen::MatrixX2d> &d, double error, std::vector<Eigen::Matrix<double, 4, 2>> &curves);

/*
 *  Bezier :
 *  	Evaluate a Bezier curve at a particular parameter value
 *      V is a (degree+1, D) matrix where D is the dimension of the points
 */
Eigen::VectorXd BezierII(int degree, const Eigen::Ref<const Eigen::MatrixXd> &V, double t);

/*
 *  ChordLengthParameterize :
 *	Assign parameter values to digitized points
 *	using relative distances between points.
 */
std::vector<double> ChordLengthParameterize(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last);

/*
 *  GenerateBezier :
 *  Use least-squares method to find Bezier control points for region.
 *
 */
Eigen::Matrix<double, 4, 2>
GenerateBezier(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last, const std::vector<double> &uPrime,
               const Eigen::Vector2d &tHat1, const Eigen::Vector2d &tHat2);
Eigen::Matrix<double, -1, 2> FitBezier(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last, const std::vector<double> &uPrime, int degree=3,
                                       const Eigen::Vector2d &leftTangent= Eigen::Vector2d(0, 0), const Eigen::Vector2d &rightTangent= Eigen::Vector2d(
        0, 0));

/*
 *  ComputeMaxError :
 *	Find the maximum squared distance of digitized points
 *	to fitted curve.
*/
double ComputeMaxError(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last,
                       const Eigen::Matrix<double, 4, 2> &bezCurve, const std::vector<double> &u,
                       int &splitPointconst);
/*
 *  Reparameterize:
 *	Given set of points and their parameterization, try to find
 *   a better parameterization.
 *
 */
std::vector<double>
Reparameterize(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last, const std::vector<double> &u,
               const Eigen::Matrix<double, -1, 2> &bezCurve);
Eigen::MatrixX2d sampleCurve(const Eigen::Matrix<double, 4, 2> &bezierCurve, int numPts);
Eigen::MatrixX2d sampleCurves(const std::vector<Eigen::Matrix<double, 4, 2>> &bezierCurve, int subdivisions);