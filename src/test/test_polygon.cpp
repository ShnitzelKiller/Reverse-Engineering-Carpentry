//
// Created by James Noeckel on 2/13/20.
//

#include "geometry/shapes2/Primitive.h"
#include "opencv2/opencv.hpp"
#include <ctime>

//#include "math/NormalRandomVariable.h"
#define WIDTH 256
#define HEIGHT 256
using namespace Eigen;

void display(const Primitive &poly, const std::string &name) {
    cv::Mat canvas(HEIGHT, WIDTH, CV_8UC1);
    auto start_t = clock();
    for (int i=0; i<WIDTH; i++) {
        for (int j=0; j<HEIGHT; j++) {
            if (poly.contains(Eigen::Vector2d(j, i))) {
                canvas.at<uchar>(i, j) = 255;
            } else {
                canvas.at<uchar>(i, j) = 0;
            }
        }
    }
    auto total_t = clock() - start_t;
    float time_sec = static_cast<float>(total_t)/CLOCKS_PER_SEC;
    std::cout << "total inference time: " << time_sec << std::endl;
    cv::imwrite(name, canvas);
}

Polygon test_circle(size_t num_points) {
    Eigen::MatrixX2d points(num_points, 2);
    for (int i=0; i<num_points; i++) {
        float ang = M_PI * 2 * (static_cast<float>(i)/num_points);
        float x = WIDTH/2 * (1 + cos(ang));
        float y = HEIGHT/2 * (1 + sin(ang));
        points(i, 0) = x;
        points(i, 1) = y;
    }
    auto start_t = clock();
    Polygon poly(points);
    auto total_t = clock() - start_t;
    float time_sec = static_cast<float>(total_t)/CLOCKS_PER_SEC;
    std::cout << "time to construct with " << num_points << " points: " << time_sec << std::endl;
    display(poly, "test_polygon_circle_" + std::to_string(num_points) + ".png");
    return poly;
}

int main(int argc, char **argv) {

    {
        MatrixX2d points(5, 2);
        points <<
                0, 0,
                0, 1,
                0.7f, 0.7f,
                0.5f, 0.2f,
                1, 0;
        points = (points.array() + 0.2) * WIDTH * 0.5;

        Polygon poly(points);
        display(poly, "test_polygon_1.png");
    }

    {
        test_circle(16);
        test_circle(10000);
        Polygon poly = test_circle(1000000);
        Polygon poly_copied(poly);
        MatrixX2d extrapoints(4, 2);
        extrapoints << 30, 30,
                30, 200,
                200, 200,
                200, 30;
        poly_copied.addPoints(extrapoints);
        display(poly_copied, "test_copied.png");
        display(poly, "test_original.png");
        poly_copied.clear();
        display(poly_copied, "test_cleared.png");
        poly_copied.addPoints(extrapoints);
        display(poly_copied, "test_re-added.png");
    }

    {
        MatrixX2d points(4, 2);
        points <<
               0, 0,
                0, 1,
                1, 1,
                1, 0;
        points = (points.array() + 0.2) * WIDTH * 0.5;

        PolygonWithHoles poly(points, {});
        display(poly, "test_polygon_with_holes_no_holes.png");
    }

    {
        MatrixX2d points(4, 2);
        points <<
               0, 0,
                0, 1,
                1, 1,
                1, 0;
        points = (points.array() + 0.2) * WIDTH * 0.5;

        MatrixX2d holepoints = points.array() * 0.5 + 0.25;
        MatrixX2d holepoints2 = holepoints.array() + WIDTH/4;
        PolygonWithHoles poly(points, {std::make_shared<Polygon>(holepoints), std::make_shared<Polygon>(holepoints2)});
        display(poly, "test_polygon_with_holes.png");
    }

    {
        MatrixX2d points(4, 2);
        points <<
               0, 0,
                0, 1,
                1, 1,
                1, 0;
        points = (points.array() + 0.2) * WIDTH * 0.5;

        MatrixX2d holepoints = points.array() * 0.5 + 0.25;
        MatrixX2d holepoints2 = holepoints.array() + WIDTH/4;
        std::cout << "untransformed poly: " << std::endl;
        PolygonWithHoles poly(points, {std::make_shared<Polygon>(holepoints), std::make_shared<Polygon>(holepoints2)});
        std::cout << poly.points() << std::endl;
        std::cout << "transformed poly: " << std::endl;
        poly.transform(Vector2d(WIDTH * 0.3, WIDTH * 0.3), 0, 1);
        std::cout << poly.points() << std::endl;
        display(poly,"test_polygon_with_holes_transformed.png");
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
        domeCurve.transform(Vector2d(0, 0), 0, WIDTH/2);
        PolyCurveWithHoles polycurve(domeCurve);
        display(polycurve, "dome.png");

        polycurve.transform(Vector2d(WIDTH/8, 0), M_PI/4, 1.1);
        domeCurve.transform(Vector2d(WIDTH/8, 0), M_PI/4, 1.1);
        PolyCurveWithHoles polycurve2(std::move(domeCurve));
        display(polycurve, "dome_transformed_poly.png");
        display(polycurve2, "dome_transformed_curve.png");
    }

    {
        MatrixX2d points(5, 2);
        points <<
               0, 0,
                0, 1,
                0.7f, 0.7f,
                0.5f, 0.2f,
                1, 0;
        /*MatrixX2d points(4, 2);
        points <<
            0, 0,
            1, 0,
            1, 1,
            0, 1;*/
        points = (points.array() + 0.2) * WIDTH * 0.5;
        Polygon poly(points.colwise().reverse());
        cv::Mat canvas(HEIGHT, WIDTH, CV_8UC3);
        for (int i = 0; i < WIDTH; i++) {
            for (int j = 0; j < HEIGHT; j++) {
                if (poly.contains(Eigen::Vector2d(j, i))) {
                    canvas.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
                } else {
                    canvas.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                }
            }
        }
        Eigen::Vector2d ray_origin(0.352 * WIDTH, 0.9 * HEIGHT);
        //Eigen::Vector2d ray_origin = points.row(2).transpose();
        Eigen::Vector2d ray_direction(0, 1);
        Ray2d ray(ray_origin, ray_direction);
        auto intersections = poly.intersect(ray, 0);
        std::cout << "intersections: " << intersections.size() << std::endl;
        for (auto & t : intersections) {
            std::cout << '(' << t.t << ", " << t.curveIndex << ", " << t.curveDist << ", entering=" << t.entering << "), ";
            Eigen::Vector2d pt = ray_origin + ray_direction * t.t;
            cv::circle(canvas, cv::Point(pt.x(), pt.y()), 2, cv::Vec3b(255, 0, 0), cv::FILLED);
        }
        cv::imwrite("test_polygon_intersection.png", canvas);

        auto newpolys = poly.split(ray);
        for (size_t i=0; i<newpolys.first.size(); ++i) {
            display(*(newpolys.first[i]), "left_split_" + std::to_string(i) + ".png");
        }
        for (size_t i=0; i<newpolys.second.size(); ++i) {
            display(*(newpolys.second[i]), "right_split_" + std::to_string(i) + ".png");
        }
    }

    return 0;
}