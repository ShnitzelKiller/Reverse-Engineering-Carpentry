//
// Created by James Noeckel on 4/15/20.
//

#include "geometry/shapes2/CombinedCurve.h"
#include <iostream>
#include <vector>
#include "utils/io/csvFormat.h"
#include "test/testUtils/curveFitUtils.h"

int main(int argc, char **argv) {
    int ksize = 2;
    if (argc > 1) {
        ksize=std::stoi(argv[1]);
    }
    double default_line_cost = 0.0004;
    double default_bezier_cost = 0.0008;
    std::mt19937 random{std::random_device{}()};
    using namespace Eigen;

    /*{
        //test single circle arc curve
        int N = 3;
        for (int i=0; i<10; ++i) {
            Eigen::MatrixX2d d(N, 2);
            d.setRandom();
            d.array() += 1;
            d /= 2;

            std::unique_ptr<Curve> arc(new CircularArc);
            arc->fit(d, 0, 2);
            CombinedCurve curve;
            curve.addCurve(std::move(arc));
            display_fit("circular_arc", curve, d, false);
        }
    }*/
    /*{
        MatrixX2d d = openData("../test_data/shapedetection-badhole/part_1_hole_1_segmentation_contour_0.csv").colwise().reverse();
        int dataKsize = 5;
        if (d.rows() == 0 || d.cols() != 2) {
            std::cout << "invalid matrix size: " << d.rows() << ", " << d.cols() << std::endl;
            return 1;
        }
        d.rowwise() -= d.colwise().minCoeff();
        double maxScale = d.maxCoeff();
        d /= maxScale;
        double errorScale = 1.0 / 250.0;
        double curveCost = 0.05 * errorScale * errorScale;
        std::vector<LineConstraint> edges;
        LineConstraint edge1;
        edge1.edge = Edge2d({0.1, 0.03}, {0.3, 0.03});
        edge1.threshold = 0.1;
        LineConstraint edge2;
        edge2.edge = Edge2d({0.4, 0.03}, {0.6, 0.03});
        edge2.threshold = 0.1;
        edges.push_back(edge1);
        edges.push_back(edge2);
        auto curve = test_fit("badHole_cost_" + std::to_string(curveCost), d, curveCost, curveCost * 0.5,
                 50, 0, -1, dataKsize, edges);
        curve.fixKnots(60/180.0*M_PI, d);
        display_fit("badHole_regularized_fixedknots", curve, d);
        return 0;
    }*/

    {
        //test erroneous alignment
        CombinedCurve curve;
        std::ifstream f("../test_data/curveData/part5_curveFit_5_constraints.txt");
        curve.loadPlaintext(f);
        Vector2d minPt;
        double scale = computeScale(curve, minPt);
        std::cout << "scale: " << getDims(curve).transpose() << std::endl;
        display_curve("badCurve", curve, scale, minPt);
    }

//#if false

    /*{
        //test segmentation data
        int dataKsize = 12;
        std::vector<Eigen::MatrixX2d> ds;
        double maxScale = 0;
        for (int c : {1, 2, 0, 3}) {
//            MatrixXd d = openData("../test_data/shapedetection-newpipeline_minsize/part_" + std::to_string(c) + "_segmentation_contour.csv");
//            MatrixXd d = openData("../test_data/shapedetection_newpipeline_avgsize/part_" + std::to_string(c) + "_segmentation_contour.csv");
            MatrixXd d = openData("../test_data/shapedetection_newpipeline_minsize20/part_" + std::to_string(c) + "_segmentation_contour.csv");
            if (d.rows() == 0 || d.cols() != 2) {
                std::cout << "invalid matrix size: " << d.rows() << ", " << d.cols() << std::endl;
                return 1;
            }
            d.rowwise() -= d.colwise().minCoeff();
            maxScale = std::max(maxScale, d.maxCoeff());
            ds.push_back(std::move(d));
        }
        for (int c = 0; c < ds.size(); ++c) {
            auto &d = ds[c];
            d /= maxScale;
            d *= 0.9;
            d.array() += 0.05;
//            std::cout << d << std::endl;
            std::string name = "realdata_" + std::to_string(c);
            display_curvatures(name, d, dataKsize, true, true);
            //for (size_t j=0; j<10; ++j) {
            //double curveCost = j * 0.00005;
            double errorScale = 1.0 / 250.0;
            double curveCost = 0.05 * errorScale * errorScale;
//            std::vector<Eigen::Matrix2d> edges(1, (Matrix2d() << -0.1, 0.1, 1, 0.1).finished());
            CombinedCurve curve = test_fit(name + "_cost_" + std::to_string(curveCost), d, curveCost, curveCost * 0.5,
                                           50, 0, -1, dataKsize);
                                           //, edges, 0.05);
            Vector2d dir(0, 1);
            for (int i=0; i<curve.size(); ++i) {
                if (curve.getCurve(i).type() == CurveTypes::LINE_SEGMENT) {
                    std::cout << "dotprod (before) " << i << " with dir: " << curve.getCurve(i).tangent(0.5).normalized().dot(dir.normalized()) << std::endl;
                }
            }
            curve.align(0.01, 10/180.0*M_PI, dir);
            for (int i=0; i<curve.size(); ++i) {
                if (curve.getCurve(i).type() == CurveTypes::LINE_SEGMENT) {
                    std::cout << "dotprod " << i << " with dir: " << curve.getCurve(i).tangent(0.5).normalized().dot(dir.normalized()) << std::endl;
                }
            }
//            curve.ransac(d, 0.1, M_PI / 6, errorScale, random);
            display_fit(name + "_regularized", curve, d);
            curve.fixKnots(60/180.0*M_PI);
            display_fit(name + "_regularized_fixedknots", curve, d);
        }
    }*/

    /*{
        //test degree 2 bezier
        int N = 3;
        Eigen::Matrix2d cpts;
        cpts <<    0.2, 0.1,
                0.6, 0.2;
        LineSegment line(cpts);
        Eigen::MatrixX2d d = line.uniformSample(10, 10);
        d.row(5) += RowVector2d(0, 0.2);
        d.row(9) += RowVector2d(0, 0.2);
        d.row(0) += RowVector2d(0, 0.2);
        test_bezier("Degree 2", d, 0, 9, 2);
    }

    {
        //test single bezier curve with free tangents
        int N = 5;
        Eigen::MatrixX2d d(N, 2);
        d << 0.1, 0.1,
            0.2, 0.5,
            0.4, 0.3,
            0.6, 0.2,
            0.8, 0.3;
        auto curve = test_bezier("Free tangents", d, 0, 4);
        std::cout << "curvature at 0: " << curve->curvature(0);
        std::cout << "curvature at 1: " << curve->curvature(1);
        {
            CombinedCurve cCurve;
            cCurve.addCurve(curve->clone());
            for (size_t i = 0; i < 10; i++) {
                double ang = static_cast<double>(i) / 10 * 2 * M_PI;
                Eigen::Vector2d dir(cos(ang), sin(ang));
                display_fit("Free tangents w/ projectedMin", cCurve, d, -1, true, {}, 0, {}, dir);
            }
        }
        {
            CombinedCurve cCurve;
            std::unique_ptr<Curve> line = std::make_unique<LineSegment>();
            line->fit(d, 0, 4);
            cCurve.addCurve(std::move(line));
            for (size_t i = 0; i < 10; i++) {
                double ang = static_cast<double>(i) / 10 * 2 * M_PI;
                Eigen::Vector2d dir(cos(ang), sin(ang));
                display_fit("Line segment w/ projectedMin", cCurve, d, -1, true, {}, 0, {}, dir);
            }
        }
        {
            auto curves = ((BezierCurve*) curve.get())->split(0.5);
            std::cout << "left curve: \n" << curves.first.points() << std::endl;
            std::cout << "right curve: \n" << curves.second.points() << std::endl;
            CombinedCurve left;
            left.addCurve(curves.first.clone());
            display_fit("left curve", left, d);
            CombinedCurve right;
            right.addCurve(curves.second.clone());
            display_fit("right curve", right, d);
        }
        test_bezier("Both constrained tangent", d, 0, 4, 3, Eigen::Vector2d(1, 1).normalized(), Eigen::Vector2d(-1, -1).normalized());
        test_bezier("Left constrained tangent", d, 0, 4, 3, Eigen::Vector2d(0, 1).normalized());
        test_bezier("Right constrained tangent", d, 0, 4, 3, Eigen::Vector2d(0, 0), Eigen::Vector2d(0, -1).normalized());
    }

    {
        //test single bezier curve with free tangents
        int N = 3;
        Eigen::MatrixX2d d(N, 2);
        d <<    0.2, 0.1,
                0.4, 0.3,
                0.6, 0.2;
        test_bezier("Degree 2", d, 0, 2);
    }
    {
        int N = 20;
        Eigen::MatrixX2d d(N, 2);
        for (int i = 0; i < N; i++) {
            double t = static_cast<double>(i) / N;
            d.row(i) = Eigen::RowVector2d(0.25 * std::sin(t * M_PI * 2) + 0.5, 0.25 * std::cos(t * M_PI * 2) + 0.5);
        }
        test_fit("circle ", d, 0.0001, 0.0001);
        //for (int i=0; i<N; i++) {
        //   test_bezier("circle_tangentPiece " + std::to_string(i), d, i, i + 5);
        //}
    }
    {
        int N = 20;
        Eigen::MatrixX2d d(N, 2);
        for (int i = 0; i < N; i++) {
            double t = static_cast<double>(i * i) / (N * N);
            d.row(i) = Eigen::RowVector2d(0.25 * std::sin(t * M_PI * 2) + 0.5, 0.25 * std::cos(t * M_PI * 2) + 0.5);
        }
        test_fit("nonuniform circle", d, 0.0001, 0.0001);
        test_fit("partial nonuniform circle", d, 0.0001, 0.0001, -1, 25, 35);
    }*/

    {
        CombinedCurve square;
        square.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() << 0.1, 0.1, 0.9, 0.1).finished()));
        square.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() << 0.9, 0.1, 0.9, 0.9).finished()));
        square.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() << 0.9, 0.9, 0.1, 0.9).finished()));
        square.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() << 0.1, 0.9, 0.1, 0.1).finished()));
        Eigen::MatrixX2d squarePoints = square.uniformSample(10, 10).colwise().reverse();
        std::cout << "square points size: " << squarePoints.rows() << std::endl;
        {
            std::vector<LineConstraint> edges;
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.1, 0.9, 0.1).finished(), 0.15);
            edges.emplace_back((Eigen::Matrix2d() << 0.9, 0.1, 0.9, 0.9).finished(), 0.15);
            edges.emplace_back((Eigen::Matrix2d() << 0.9, 0.9, 0.1, 0.9).finished(), 0.15);
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.9, 0.1, 0.1).finished(), 0.15);
            test_fit("square_fullconstrained", squarePoints, default_bezier_cost, default_line_cost, -1, 0, -1, ksize, edges);
        }
    }

    {
        int N = 40;
        Eigen::MatrixX2d d(N, 2);
        for (int i = 0; i < N; i++) {
            double t = static_cast<double>(i) / N;
            d.row(i) = Eigen::RowVector2d(0.3 * std::sin(t * M_PI * 2) + 0.5, 0.3 * std::cos(t * M_PI * 2) + 0.5);
        }
        {
            std::vector<LineConstraint> edges;
            edges.emplace_back((Eigen::Matrix2d() << 0.25, 0.1, 0.75, 0.1).finished(), 0.15);
            edges.emplace_back((Eigen::Matrix2d() << 0.9, 0.25, 0.9, 0.75).finished(), 0.15);
            edges.emplace_back((Eigen::Matrix2d() << 0.75, 0.9, 0.25, 0.9).finished(), 0.15);
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.75, 0.1, 0.25).finished(), 0.15);
            test_fit("circle_fullconstrained_notstitched", d, 0.0001, 0.0001, -1, 0, -1, ksize, edges);
        }
        {
            std::vector<LineConstraint> edges;
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.1, 0.9, 0.1).finished(), 0.15);
            edges.emplace_back((Eigen::Matrix2d() << 0.9, 0.1, 0.9, 0.9).finished(), 0.15);
            edges.emplace_back((Eigen::Matrix2d() << 0.9, 0.9, 0.1, 0.9).finished(), 0.15);
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.9, 0.1, 0.1).finished(), 0.15);
            test_fit("circle_fullconstrained_stitched", d, 0.0001, 0.0001, -1, 0, -1, ksize, edges);
        }
    }

    {
        CombinedCurve domeCurve;
        domeCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() << 0.1, 0.1, 0.9, 0.1).finished()));
        domeCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() << 0.9, 0.1, 0.9, 0.5).finished()));
        Eigen::Matrix<double, 4, 2> controlPoints;
        controlPoints << 0.9, 0.5,
                0.9, 0.9,
                0.1, 0.9,
                0.1, 0.5;
        domeCurve.addCurve(std::make_unique<BezierCurve>(controlPoints));
        domeCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() << 0.1, 0.5, 0.1, 0.1).finished()));
        Eigen::MatrixX2d domePoints = domeCurve.uniformSample(10, 10).colwise().reverse();

        {
            CombinedCurve domeTrans(domeCurve);
            display_curve("dome untranslated", domeTrans, 500, Vector2d(0, 0));
            domeTrans.transform(Vector2d(0, 0.1), M_PI/4, 1);
            display_curve("dome translated", domeTrans, 500, Vector2d(0, 0));
        }

        {
            test_fit("dome nolines", domePoints, default_bezier_cost, 1000, 4, 0, -1, ksize);
        }

        {
            CombinedCurve fitCurve = test_fit("dome", domePoints, default_bezier_cost, default_line_cost, 4, 0, -1, ksize);
//            for (size_t i = 0; i < 10; i++) {
//                double ang = static_cast<double>(i) / 10 * 2 * M_PI;
//                Eigen::Vector2d dir(cos(ang), sin(ang));
//                display_fit("Dome w/ projectedMin", fitCurve, domePoints, 5, true, {}, 0, {}, dir);
//            }
        }

        {
            std::vector<LineConstraint> edges;
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.1, 0.9, 0.1).finished(), 0.01);
            test_fit("dome_constrained", domePoints, default_bezier_cost, default_line_cost, -1, 0, -1, ksize, edges);
            //test_bezier("dome_constrained_tangentPiece", d, 1, 3);
            //test_bezier("dome_constrained_tangentPiece2", d, 6, 8);
        }
        {
            std::vector<LineConstraint> edges;
//            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.1, 0.9, 0.1).finished(), 0.01);
            edges.emplace_back((Eigen::Matrix2d() << 0.9, 0.1, 0.9, 0.5).finished(), 0.01);
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.1, 0.1, 0.5).finished(), 0.01);
            test_fit("dome_multi_constrained", domePoints, default_bezier_cost, default_line_cost, -1, 0, -1, ksize, edges);
        }
        {
            Eigen::MatrixX2d d = domePoints;
//            d(3, 1) += 0.5;
            std::vector<LineConstraint> edges;
            edges.emplace_back((Eigen::Matrix2d() << 0.35, 0.15, 0.95, 0.15).finished(), 0.07);
            auto curve = test_fit("dome_projected", d, default_bezier_cost, default_line_cost, -1, 0, -1, ksize,edges);
            curve.fixKnots(0.45);
            display_fit("dome_projected_aligned", curve, domePoints, -1, true, edges);
        }

        {
            std::vector<LineConstraint> edges;
//            edges.emplace_back((Eigen::Matrix2d() << 0.05, 0.15, 0.95, 0.15).finished(), 0.07);
            edges.emplace_back((Eigen::Matrix2d() << 0.84, 0.1, 0.84, 0.5).finished(), 0.07);
            edges.emplace_back((Eigen::Matrix2d() << 0.16, 0.1, 0.16, 0.5).finished(), 0.07);
            auto curve = test_fit("dome_multi_projected", domePoints, default_bezier_cost, default_line_cost, -1, 0, -1, ksize, edges);
            curve.fixKnots(0.45);
            display_fit("dome_multi_projected_aligned", curve, domePoints, -1, true, edges);
        }

        {
            std::vector<LineConstraint> edges;
            edges.emplace_back((Eigen::Matrix2d() << 0.05, 0.15, 0.95, 0.15).finished(), 0.07);
            edges.emplace_back((Eigen::Matrix2d() << 0.87, 0.1, 0.87, 0.5).finished(), 0.07);
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.1, 0.1, 0.5).finished(), 0.07);
            edges.emplace_back((Eigen::Matrix2d() << 0.05, 0.5, 0.95, 0.5).finished(), 0.07);
            test_fit("dome_all_projected", domePoints, default_bezier_cost, default_line_cost, -1, 0, -1, ksize, edges);
        }

        {
            std::vector<LineConstraint> edges;
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.15, 0.9, 0.15).finished(), 0.02);
            edges.emplace_back((Eigen::Matrix2d() << 0.87, 0.1, 0.87, 0.5).finished(), 0.02);
            edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.1, 0.1, 0.5).finished(), 0.02);
            test_fit("dome_multi_projected_large_threshold", domePoints, default_bezier_cost, default_line_cost, -1, 0,
                     -1, ksize, edges);
        }
        {
            std::vector<LineConstraint> edges;
            //edges.emplace_back((Eigen::Matrix2d() << 0.1, 0.15, 0.9, 0.15).finished(), 0.05);
            edges.emplace_back((Eigen::Matrix2d() << 0.87, 0.1, 0.87, 0.5).finished(), 0.05);
            edges.emplace_back((Eigen::Matrix2d() << 0.15, 0.1, 0.15, 0.5).finished(), 0.05);
            test_fit("dome_side_overlapping_projected", domePoints, default_bezier_cost, default_line_cost, -1, 0, -1, ksize,
                     edges);
        }
    }
    {
        CombinedCurve trueCurve;
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() <<0.1, 0.1, 0.9, 0.1).finished()));
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() << 0.9, 0.1, 0.9, 0.5).finished()));
        Eigen::Matrix<double, 4, 2> controlPoints;
        controlPoints << 0.9, 0.5,
                0.9, 0.9,
                0.1, 0.9,
                0.1, 0.5;
        trueCurve.addCurve(std::make_unique<BezierCurve>(controlPoints));
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() <<0.1, 0.5, 0.1, 0.1).finished()));
        Eigen::MatrixX2d constraint = trueCurve.getCurve(2).uniformSample(10, 10);//.block(2, 0, 6, 2);
        std::vector<Eigen::MatrixX2d> constraints;
        std::vector<LineConstraint> edges;
        constraints.push_back(std::move(constraint));
        Eigen::MatrixX2d d = trueCurve.uniformSample(20, 20);
        test_fit("dome curve constraint", d, default_bezier_cost, default_line_cost, -1, 0, -1, ksize, edges, 0.1, constraints);
    }
    {
        CombinedCurve trueCurve;
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() <<0.1, 0.1, 0.9, 0.1).finished()));
        Eigen::Matrix<double, 4, 2> controlPoints0;
        controlPoints0 << 0.9, 0.1,
                0.7, 0.2,
                0.7, 0.3,
                0.9, 0.5;
        trueCurve.addCurve(std::make_unique<BezierCurve>(controlPoints0));
        Eigen::Matrix<double, 4, 2> controlPoints;
        controlPoints << 0.9, 0.5,
                0.9, 0.9,
                0.1, 0.9,
                0.1, 0.5;
        trueCurve.addCurve(std::make_unique<BezierCurve>(controlPoints));
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() <<0.1, 0.5, 0.1, 0.1).finished()));
        Eigen::MatrixX2d d = trueCurve.uniformSample(20, 20);
        test_fit("dome multi curve", d, default_bezier_cost, default_line_cost, -1, 0, -1, ksize);
    }
    {
        CombinedCurve trueCurve;
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() <<0.1, 0.1, 0.9, 0.1).finished()));
        Eigen::Matrix<double, 4, 2> controlPoints0;
        controlPoints0 << 0.9, 0.1,
                0.7, 0.2,
                0.7, 0.3,
                0.9, 0.5;
        trueCurve.addCurve(std::make_unique<BezierCurve>(controlPoints0));
        Eigen::Matrix<double, 4, 2> controlPoints;
        controlPoints << 0.9, 0.5,
                0.9, 0.9,
                0.1, 0.9,
                0.1, 0.5;
        trueCurve.addCurve(std::make_unique<BezierCurve>(controlPoints));
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() <<0.1, 0.5, 0.1, 0.1).finished()));
        Eigen::MatrixX2d constraint0 = trueCurve.getCurve(1).uniformSample(10, 10);
        Eigen::MatrixX2d constraint = trueCurve.getCurve(2).uniformSample(10, 10);//.block(2, 0, 6, 2);
        std::vector<Eigen::MatrixX2d> constraints;
        std::vector<LineConstraint> edges;
        constraints.push_back(std::move(constraint0));
        constraints.push_back(std::move(constraint));
        Eigen::MatrixX2d d = trueCurve.uniformSample(20, 20);
        test_fit("dome multi curve constraint", d, default_bezier_cost, default_line_cost, -1, 0, -1, ksize, edges,
                 0.1, constraints);
    }
    {
        CombinedCurve trueCurve;
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() <<0.1, 0.1, 0.9, 0.1).finished()));
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() << 0.9, 0.1, 0.9, 0.5).finished()));
        Eigen::Matrix<double, 4, 2> controlPoints;
        controlPoints << 0.9, 0.5,
                0.9, 0.9,
                0.1, 0.9,
                0.1, 0.5;
        trueCurve.addCurve(std::make_unique<BezierCurve>(controlPoints));
        trueCurve.addCurve(std::make_unique<LineSegment>((Eigen::Matrix2d() <<0.1, 0.5, 0.1, 0.1).finished()));
        Eigen::MatrixX2d constraint = trueCurve.getCurve(2).uniformSample(10, 10);//.block(2, 0, 6, 2);
        std::vector<Eigen::MatrixX2d> constraints;
        std::vector<LineConstraint> edges;
        edges.emplace_back((Eigen::Matrix2d() << 0.9, 0.1, 0.9, 0.5).finished(), 0.1);
        constraints.push_back(std::move(constraint));
        Eigen::MatrixX2d d = trueCurve.uniformSample(20, 20);
        test_fit("dome curve line and curve constraint", d, default_bezier_cost, default_line_cost, -1, 0, -1, ksize, edges,
                 0.1, constraints);
    }
    /*{
        int N = 20;
        Eigen::MatrixX2d d(N, 2);
        for (int i = 0; i < N; i++) {
            double t = static_cast<double>(i) / N;
            d.row(i) = Eigen::RowVector2d(0.25 * std::sin(t * M_PI * 2) + 0.5, 0.25 * std::cos(t * M_PI * 2) + 0.5);
        }
        for (int j=0; j<N; j++) {
            Eigen::MatrixX2d d2(N, 2);
            for (int i = 0; i < N; i++) {
                d2.row(i) = d.row((i + j) % N);
            }
            std::vector<LineConstraint> edges;
            edges.push_back((Eigen::Matrix2d() << 0.3, 0, 0.3, 1).finished());
            edges.push_back((Eigen::Matrix2d() << 0.7, 0, 0.7, 1).finished());
            test_fit("constrained circle circle " + std::to_string(j), d2, 0.0001, 0.0001, -1, 0, -1, edges, 0.2);
        }
    }*/

    return 0;
}