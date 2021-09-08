//
// Created by James Noeckel on 10/28/20.
//

#include "curveFitUtils.h"
#include "math/nonmax_suppression.h"
#include "utils/printvec.h"
#include "geometry/shapes2/find_adjacent_ranges.h"

#define IMGSCALE 1200
#define MARGIN 100

Eigen::Vector2d getDims(const CombinedCurve &curve) {
    Eigen::MatrixX2d d = curve.uniformSample(10);
    return (d.colwise().maxCoeff() - d.colwise().minCoeff()).transpose();
}

double computeScale(const CombinedCurve &curve, Eigen::Vector2d &minPt) {
    Eigen::MatrixX2d d = curve.uniformSample(10);
    minPt = d.colwise().minCoeff().transpose();
    double SCALE = (IMGSCALE - 2*MARGIN) / ((d.colwise().maxCoeff().transpose() - minPt).maxCoeff());
    minPt.array() -= MARGIN / SCALE;
    return SCALE;
}

void draw_curve(cv::Mat &img, const CombinedCurve &curve, double SCALE, const Eigen::Vector2d &minPt, int thickness, const cv::Scalar &color) {
    if (curve.size() != 0) {
        //draw control points
//        for (int i = 0; i < curve.size(); i++) {
//            if (curve.getCurve(i).type() == CurveTypes::BEZIER_CURVE) {
//                const auto *bezierCurve = (const BezierCurve *) &curve.getCurve(i);
//                for (int k = 0; k < bezierCurve->points().rows() - 1; k++) {
//                    std::cout << "drawing curve control line " << k << " for curve " << i << std::endl;
//                    Eigen::RowVector2d pt1 = (bezierCurve->points().row(k) - minPt.transpose()) * SCALE;
//                    Eigen::RowVector2d pt2 = (bezierCurve->points().row(k + 1) - minPt.transpose()) * SCALE;
//                    cv::line(img, cv::Point2d(pt1.x(), pt1.y()), cv::Point2d(pt2.x(), pt2.y()),
//                             cv::Scalar(255, 255, 0), thickness/2);
//                }
//            }
//        }
        //draw curves
        Eigen::MatrixX2d samples = curve.uniformSample(1000, 1000);
        samples.rowwise() -= minPt.transpose();
        samples *= SCALE;
        for (int i = 0; i < samples.rows() - 1; i++) {
            cv::line(img, cv::Point2d(samples(i, 0), samples(i, 1)),
                     cv::Point2d(samples(i + 1, 0), samples(i + 1, 1)), color, thickness);
        }
        //draw knots
        for (int i = 0; i < curve.size(); i++) {
            Eigen::Vector2d pt = (curve.getCurve(i).sample(0) - minPt) * SCALE;
            Eigen::Vector2d pt2 = (curve.getCurve(i).sample(1) - minPt) * SCALE;
            cv::circle(img, cv::Point2d(pt(0), pt(1)), 10,
                       cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(img, cv::Point2d(pt2(0), pt2(1)), thickness*4,
                        cv::Scalar(0, 0, 255), cv::FILLED);
        }
    }
}

void display_curvatures(const std::string &name, const Eigen::MatrixX2d &d, int ksize, bool useEndpoints, bool save) {
    std::cout << "drawing curvature-based corner selection for " << name << std::endl;
    Eigen::Vector2d minPt = d.colwise().minCoeff().transpose();
    double SCALE = (IMGSCALE - 2*MARGIN) / ((d.colwise().maxCoeff().transpose() - minPt).maxCoeff());
    minPt.array() -= MARGIN / SCALE;
    cv::Mat img(IMGSCALE, IMGSCALE, CV_8UC3);
    img = cv::Scalar(255, 255, 255);
    int N = d.rows();
    std::vector<double> angleDiffs(N+2);
    std::vector<CombinedCurve> allCurves;
    for (int i=0; i<N; ++i) {
        int left = i-ksize;
        int mid = i;
        int right = i+ksize;
        if (left < 0) {
            left += N;
            mid += N;
            right += N;
        }
        std::unique_ptr<Curve> leftCurve(new BezierCurve(2));
        std::unique_ptr<Curve> rightCurve(new BezierCurve(2));
        leftCurve->fit(d, left, mid);
        rightCurve->fit(d, mid, right);

        Eigen::Vector2d t1 = leftCurve->tangent(0.5).normalized();
        Eigen::Vector2d t2 = rightCurve->tangent(0.5).normalized();

        double angcos = t1.dot(t2);
        double ang = std::acos(angcos);
        angleDiffs[i+1] = ang;
        uchar col = cv::saturate_cast<uchar>(2*ang / M_PI * 255);
        cv::circle(img, SCALE * cv::Point2d(d(i, 0) - minPt.x(), d(i, 1) - minPt.y()), 5, cv::Scalar(col, col, col), cv::FILLED);
        CombinedCurve combinedCurve;
        combinedCurve.addCurve(std::move(leftCurve));
        combinedCurve.addCurve(std::move(rightCurve));
        allCurves.push_back(std::move(combinedCurve));
    }
//    std::cout << "angleDiffs before nonmax suppression: " << angleDiffs << std::endl;
    //cyclic nonmax suppression
    angleDiffs[0] = angleDiffs[N];
    angleDiffs[N+1] = angleDiffs[1];
    nonmax_suppression<double>(angleDiffs.cbegin(), angleDiffs.cend(), angleDiffs.begin());
    angleDiffs.erase(angleDiffs.begin());
    angleDiffs.resize(N);
    {
        //draw each curve
        cv::Mat imgOrig = img.clone();
        for (int i = 0; i < N; ++i) {
            if (angleDiffs[i] > M_PI/6) {
                cv::circle(img, SCALE * cv::Point2d(d(i, 0) - minPt.x(), d(i, 1) - minPt.y()), 10, cv::Scalar(127, 0, 255), cv::FILLED);
                cv::putText(img, std::to_string(i) + ": " + std::to_string(angleDiffs[i]),
                            SCALE * cv::Point2d(d(i, 0) - minPt.x(), d(i, 1) - minPt.y()) + cv::Point2d(5, 10), cv::FONT_HERSHEY_DUPLEX, 1.5,
                            cv::Scalar(0, 0, 255));

                //display each corner curve individually
                /*cv::Mat img2 = imgOrig.clone();
                cv::circle(img2, SCALE * cv::Point2d(d(i, 0), d(i, 1)), 10, cv::Scalar(127, 0, 255), cv::FILLED);
                cv::putText(img2, std::to_string(i) + ": " + std::to_string(angleDiffs[i]),
                            SCALE * cv::Point2d(d(i, 0), d(i, 1)) + cv::Point2d(5, 10), cv::FONT_HERSHEY_DUPLEX, 1.5,
                            cv::Scalar(0, 0, 255));
                draw_curve(img2, allCurves[i], 5);
                cv::imshow(name, img2);
                cv::waitKey();
                cv::destroyWindow(name);*/
            }
        }
    }

    if (save) {
        cv::imwrite(name + ".png", img);
    }
    cv::imshow(name, img);
    cv::waitKey();
    cv::destroyWindow(name);
}

cv::Mat display_curve(const std::string &name, const CombinedCurve &curve, double scale, const Eigen::Vector2d &minPt, int thickness, const cv::Scalar &color, bool display) {
    cv::Mat img(IMGSCALE, IMGSCALE, CV_8UC3);
    img = cv::Scalar(255, 255, 255);
    draw_curve(img, curve, scale, minPt, thickness, color);
    if (display) {
        cv::imshow(name, img);
        cv::waitKey();
        cv::destroyWindow(name);
    }
    return img;
}

void display_fit(const std::string & name, const CombinedCurve &curve, const Eigen::MatrixX2d &d, int ksize, bool save, const std::vector<LineConstraint> &edges, double curveThreshold, const std::vector<Eigen::MatrixX2d> &neighbors, const Eigen::Vector2d &minDir) {
    int N = d.rows();
    Eigen::Vector2d minPt = d.colwise().minCoeff().transpose();
    double SCALE = (IMGSCALE - 2*MARGIN) / ((d.colwise().maxCoeff().transpose() - minPt).maxCoeff());
    minPt.array() -= MARGIN / SCALE;
    cv::Mat img(IMGSCALE, IMGSCALE, CV_8UC3);
    img = cv::Scalar(255, 255, 255);
    /*{
      //draw ranges
        auto ranges = find_adjacent_ranges(d, edges, neighbors, threshold);
        for (const auto &range : ranges) {
            cv::Mat img2 = img.clone();
            int start = range.first.first;
            int end = range.first.second;
            for (int i=0; i<N; i++) {
                cv::circle(img2, SCALE * cv::Point2d(d(i, 0), d(i, 1)), 2, cv::Scalar(100, 0, 0), cv::FILLED);
                if (i % 10 == 0) {
                    cv::putText(img2, std::to_string(i),
                                SCALE * cv::Point2d(d(i, 0), d(i, 1)) + cv::Point2d(5, 10), cv::FONT_HERSHEY_DUPLEX, 1.5,
                                cv::Scalar(0, 0, 255));
                    cv::line(img2, SCALE * cv::Point2d(d(i,0),d(i,1)),cv::Point2d(SCALE * d(i, 0) + 5, SCALE * d(i, 1) + 10), cv::Scalar(0, 0, 255), 1);
                }
            }
            for (int i=start; i <= end; ++i) {
                cv::circle(img2, SCALE * cv::Point2d(d(i % N, 0), d(i % N, 1)), 10, cv::Scalar(100, 255, 0), cv::FILLED);
            }
            std::string windowName = "neighbor_"+std::to_string(range.second);
            cv::imshow(windowName, img2);
            cv::waitKey();
            cv::destroyWindow(windowName);
        }
    }*/
    //draw constraints
    for (int i=0; i<edges.size(); i++) {
        Eigen::RowVector2d n = edges[i].edge.second - edges[i].edge.first;
        n = Eigen::RowVector2d(-n.y(), n.x()).normalized();
        Eigen::Matrix2d edgeOffset0;// = edges[i].rowwise() + n * thresholds[i];
        edgeOffset0 << (edges[i].edge.first.transpose() + n * edges[i].threshold),
                (edges[i].edge.second.transpose() + n * edges[i].threshold);
        Eigen::Matrix2d edgeOffset1;// = edges[i].rowwise() - n * thresholds[i];
        edgeOffset1 << (edges[i].edge.first.transpose() - n * edges[i].threshold),
                (edges[i].edge.second.transpose() - n * edges[i].threshold);
        cv::line(img, SCALE * cv::Point2d(edges[i].edge.first.x() - minPt.x(), edges[i].edge.first.y() - minPt.y()), SCALE * cv::Point2d(edges[i].edge.second.x() - minPt.x(), edges[i].edge.second.y() - minPt.y()), cv::Scalar(255, 100, 255), 12);
        cv::line(img, SCALE * cv::Point2d(edgeOffset0(0, 0) - minPt.x(), edgeOffset0(0, 1) - minPt.y()), SCALE * cv::Point2d(edgeOffset0(1, 0) - minPt.x(), edgeOffset0(1, 1) - minPt.y()), cv::Scalar(255, 200, 255), 3);
        cv::line(img, SCALE * cv::Point2d(edgeOffset1(0, 0) - minPt.x(), edgeOffset1(0, 1) - minPt.y()), SCALE * cv::Point2d(edgeOffset1(1, 0) - minPt.x(), edgeOffset1(1, 1) - minPt.y()), cv::Scalar(255, 200, 255), 3);
    }
    for (int i=0; i<neighbors.size(); i++) {
        for (int j=0; j<neighbors[i].rows(); j++) {
            cv::circle(img, SCALE * cv::Point2d(neighbors[i](j, 0) - minPt.x(), neighbors[i](j, 1) - minPt.y()), 8, cv::Scalar(255, 127, 127));
        }
    }
    //draw input points
    for (int i=0; i<N; i++) {
        cv::circle(img, SCALE * cv::Point2d(d(i, 0) - minPt.x(), d(i, 1) - minPt.y()), 5, cv::Scalar(100, 0, 0), cv::FILLED);
//        if (i % 10 == 0) {
//            cv::putText(img, std::to_string(i),
//                        SCALE * cv::Point2d(d(i, 0) - minPt.x(), d(i, 1) - minPt.y()) + cv::Point2d(5, 10), cv::FONT_HERSHEY_DUPLEX, 1.5,
//                        cv::Scalar(0, 0, 255));
//            cv::line(img, SCALE * cv::Point2d(d(i,0) - minPt.x(),d(i,1) - minPt.y()),cv::Point2d(SCALE * (d(i, 0) - minPt.x()) + 5, SCALE * (d(i, 1) - minPt.y()) + 10), cv::Scalar(0, 0, 255), 2);
//        }
    }
    if (curve.size() != 0) {
        /*if (ksize > 0) {
            for (int c = 0; c < curve.size(); ++c) {
                int i = curve.getInterval(c).first;
                int left = i - ksize;
                int mid = i;
                int right = i + ksize;
                if (left < 0) {
                    left += N;
                    mid += N;
                    right += N;
                }
                std::unique_ptr<Curve> leftCurve(new BezierCurve(2));
                std::unique_ptr<Curve> rightCurve(new BezierCurve(2));
                leftCurve->fit(d, left, mid);
                rightCurve->fit(d, mid, right);

                Eigen::Vector2d t1 = leftCurve->tangent(0.5).normalized();
                Eigen::Vector2d t2 = rightCurve->tangent(0.5).normalized();

                double angcos = t1.dot(t2);
                double ang = std::acos(angcos);
                CombinedCurve combinedCurve;
                combinedCurve.addCurve(std::move(leftCurve));
                combinedCurve.addCurve(std::move(rightCurve));

                cv::putText(img, std::to_string(i) + ": " + std::to_string(ang),
                            SCALE * cv::Point2d(d(i % d.rows(), 0), d(i % d.rows(), 1)) + cv::Point2d(5, 10),
                            cv::FONT_HERSHEY_DUPLEX, 1.5,
                            cv::Scalar(0, 0, 255));
                draw_curve(img, combinedCurve, 2, cv::Scalar(255, 255, 50));
            }
        }*/
        //draw the curve itself
        Eigen::Vector2d loopPoint = curve.sample(0);
        draw_curve(img, curve, SCALE, minPt);
        //draw start point
//        cv::circle(img, SCALE * cv::Point2d(loopPoint.x() - minPt.x(), loopPoint.y() - minPt.y()), 20, cv::Scalar(50, 0, 0), cv::FILLED);
        //draw min pt
        if (minDir.squaredNorm() > 0) {
            double t = curve.projectedMinPt(minDir);
            std::cout << "minPt: " << t << std::endl;
            Eigen::Vector2d minimumPt = curve.sample(t);
            cv::circle(img, SCALE * cv::Point2d(minimumPt(0) - minPt.x(), minimumPt(1) - minPt.y()), 2, cv::Scalar(127, 127, 255), cv::FILLED);
            Eigen::Vector2d minPtOffset(minDir.y(), -minDir.x());
            minPtOffset *= 0.2;
            Eigen::Vector2d handle0 = minimumPt - minPtOffset;
            Eigen::Vector2d handle1 = minimumPt + minPtOffset;
            Eigen::Vector2d handle3 = minimumPt + minDir * 0.1;
            cv::line(img, SCALE * cv::Point2d(handle0.x() - minPt.x(), handle0.y() - minPt.y()), SCALE * cv::Point2d(handle1.x() - minPt.x(), handle1.y() - minPt.y()),
                     cv::Scalar(0, 0, 255));
            cv::line(img, SCALE * cv::Point2d(minimumPt.x() - minPt.x(), minimumPt.y() - minPt.y()), SCALE * cv::Point2d(handle3.x() - minPt.x(), handle3.y() - minPt.y()),
                     cv::Scalar(0, 0, 255));
        }
        //draw curvature
        /*for (int i=0; i<curve.size(); i++) {
            if (curve.getCurve(i).type() == CurveTypes::BEZIER_CURVE) {
                for (int j=0; j<20; j++) {
                    double t = j * 0.05;
                    double curvature = curve.getCurve(i).curvature(t) * 3;
                    //if (curvature > 2) {
                        //std::cout << "curvature at " << t << ": " << curvature << std::endl;
                        Eigen::Vector2d n = curve.getCurve(i).tangent(t).normalized();
                        n = Eigen::Vector2d(n.y(), -n.x());
                        Eigen::Vector2d pt = curve.getCurve(i).sample(t) + 0.002 * curvature * n;
                        cv::circle(img, SCALE * cv::Point2d(pt.x(), pt.y()), 5,
                                   50 * cv::Scalar(curvature, 10-curvature, curvature), 1);
                    //}
                }
            }
        }*/
        //draw tangents
        /*for (int i=0; i<d.rows(); i++) {
            double curvature;
            Eigen::Vector2d tangent = curveTangent(d, i, curvature, true, 1, 8);
            Eigen::Vector2d pt = d.row(i).transpose();
            Eigen::Vector2d pt2 = pt + tangent * 0.25;
            cv::line(img, SCALE * cv::Point2d(pt.x(), pt.y()), SCALE * cv::Point2d(pt2.x(), pt2.y()), cv::Scalar(255, 255, 0));
        }*/
        /*for (int i=0; i<curve.size(); i++) {
            if (curve.getCurve(i).type() == CurveTypes::BEZIER_CURVE) {
                for (int j=0; j<5; j++) {
                    double t = j * 0.2;
                    Eigen::Vector2d tangent = curve.getCurve(i).tangent(t);
                    Eigen::Vector2d pt = curve.getCurve(i).sample(t);
                    Eigen::Vector2d pt2 = pt + tangent * 0.25;
                    cv::line(img, SCALE * cv::Point2d(pt.x(), pt.y()), SCALE * cv::Point2d(pt2.x(), pt2.y()), cv::Scalar(255, 0, 255));
                }
            }
        }*/
    }
    if (save) {
        cv::imwrite(name + ".png", img);
    } else {
        cv::imshow(name, img);
        cv::waitKey();
        cv::destroyWindow(name);
    }
}
std::unique_ptr<Curve> test_bezier(const std::string &name, Eigen::MatrixX2d &d, int first, int last, int degree, const Eigen::Vector2d &leftTangent, const Eigen::Vector2d &rightTangent) {
    auto bezierCurve = std::make_unique<BezierCurve>(degree);
    double error = bezierCurve->fit(d, first, last, leftTangent, rightTangent);
    std::cout << name << " error: " << error << std::endl;
    //std::cout << "bezierPoints: " << bezierCurve->points() << std::endl;
    CombinedCurve curve;
    curve.addCurve(bezierCurve->clone());
    display_fit(name, curve, d);
    return bezierCurve;
}

CombinedCurve test_fit(const std::string &name, Eigen::MatrixX2d &d, double bezier_cost, double line_cost, int knots, int first, int last, int ksize, const std::vector<LineConstraint> &edges, double curveThreshold, const std::vector<Eigen::MatrixX2d> &neighbors) {
    std::cout << "testing fit for " << name << std::endl;
    CombinedCurve curve;
    double minAngle = 0.34;
    //error = curve.fitConstrained(d, edges, neighbors, threshold, 0, knots, bezier_cost, line_cost, 1.0);
    double error = curve.fit(d, minAngle, knots, bezier_cost, line_cost, 1.0, first, last, ksize, Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0), edges, neighbors, curveThreshold);

    int num_bezier = 0, num_line=0;
    std::cout << "final knots: ";
    for (int i=0; i<curve.size(); i++) {
        if (curve.getCurve(i).type() == CurveTypes::BEZIER_CURVE) {
            num_bezier++;
        } else if (curve.getCurve(i).type() == CurveTypes::LINE_SEGMENT) {
            //std::cout << "line " << i << " points: " << std::endl << curve.getCurve(i).points() << std::endl;
            num_line++;
        }
        std::cout << curve.getInterval(i).first % d.rows() << ", ";
        if (i == curve.size()-1) std::cout << curve.getInterval(i).second % d.rows();
    }
    std::cout << std::endl << "found " << num_bezier << " bezier curves and " << num_line << " lines with error " << error << std::endl;
    display_fit(name, curve, d, ksize, true, edges, curveThreshold, neighbors);
    return curve;
}