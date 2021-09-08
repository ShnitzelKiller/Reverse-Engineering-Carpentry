//
// Created by James Noeckel on 10/5/20.
//

#include "graph_cut.h"
#include "maxflow.h"
#include <memory>
#include "geometry/shapes2/Segment2d.h"
//#define CAP_PRECISION 100

static void error_fn(const char *msg) {
    std::cerr << msg << std::endl;
}

static const std::vector<float> edgeWeights = {1.0f, 1.0f/std::sqrt(2.0f)};
static int imageId = 0;

float graph_cut(const cv::Mat &data_costs, cv::Mat &labels, float smoothness, const std::vector<cv::Mat> &contrast_image, float sigma, const std::vector<cv::Mat> &contrast_masks, const std::vector<Edge2d> &guides, double guideExtension, bool diagonalEdges, float precision) {
//    cv::Mat debugImg = cv::Mat::zeros(labels.rows, labels.cols, CV_8UC3);
    std::cout << "running with";
    if (!diagonalEdges) {std::cout << "out";}
    std::cout << " diagonal edges" << std::endl;
    std::vector<Ray2d> guideRays;
    guideRays.reserve(guides.size());
    for (const auto &edge : guides) {
        Eigen::Vector2d dir = edge.second - edge.first;
        double len = dir.norm();
        dir /= len;
        guideRays.emplace_back(edge.first, dir, 0, len);
        if (std::isfinite(guideExtension) && guideExtension > 0) {
            guideRays.back().start -= guideExtension;
            guideRays.back().end += guideExtension;
        }
    }
    if (!contrast_image.empty()) {
        if (contrast_image[0].type() != CV_8UC3 && contrast_image[0].type() != CV_32FC1) {
            std::cout << "unsupported contrast image type" << std::endl;
            return std::numeric_limits<float>::max();
        }
        if (!contrast_masks.empty() && contrast_masks.size() != contrast_image.size()) {
            std::cout << "invalid number of masks" << std::endl;
            return std::numeric_limits<float>::max();
        }
    }
    float sigma2 = sigma*sigma;
    uchar numDirections = diagonalEdges ? 4 : 2;
    int num_pixels = labels.total();
    int numEdgesMax = (labels.rows - 1) * (labels.cols - 1) * 4 + labels.rows + labels.cols;
    auto *graph = new maxflow::Graph_III(num_pixels, numEdgesMax, &error_fn);
    graph->add_node(num_pixels);
    std::cout << "adding edges" << std::endl;
    for (int i=0; i<num_pixels; ++i) {
        int col = i % labels.cols;
        int row = i / labels.cols;
        const auto &E_d = data_costs.at<cv::Vec<float, 2>>(row, col);
        float iE_d0 = E_d[0] * precision;
        float iE_d1 = E_d[1] * precision;
        graph->add_tweights(i, iE_d1, iE_d0);
        for (uchar dir=0; dir<numDirections; ++dir) {
            float edgeWeight;
            int row2 = row, col2 = col;
            if (dir == 0) {
                if (row < labels.rows-1) {
                    row2 += 1;
                    edgeWeight = edgeWeights[0];
                } else continue;
            } else if (dir == 1) {
                if (col < labels.cols-1) {
                    col2 += 1;
                    edgeWeight = edgeWeights[0];
                } else continue;
            } else if (dir == 2) {
                //down-right diagonal
                if (row < labels.rows-1 && col < labels.cols-1) {
                    row2 += 1;
                    col2 += 1;
                    edgeWeight = edgeWeights[1];
                } else continue;
            } else if (dir == 3) {
                //down-left diagonal
                if (row < labels.rows-1 && col > 0) {
                    row2 += 1;
                    col2 -= 1;
                    edgeWeight = edgeWeights[1];
                } else continue;
            }
            float E_pq;
            if (contrast_image.empty()) {
                E_pq = smoothness;
            } else {
                float diff = 0.0f;
                float count = 1e-10;
                for (int im=0; im<contrast_image.size(); ++im) {
                    const cv::Mat &img = contrast_image[im];
                    if (contrast_masks.empty() || (contrast_masks[im].at<uchar>(row, col) && contrast_masks[im].at<uchar>(row2, col2))) {
                        count += 1;
                        if (img.type() == CV_8UC3) {
                            cv::Vec3b colA = img.at<cv::Vec3b>(row, col);
                            cv::Vec3b colB = img.at<cv::Vec3b>(row2, col2);
                            float diff0 = colB[0] - colA[0];
                            float diff1 = colB[1] - colA[1];
                            float diff2 = colB[2] - colA[2];
                            diff += diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
                        } else {
                            diff += img.at<float>(row, col) - img.at<float>(row2, col2);
                        }
                    }
                }
                diff /= count;
                //edgeWeight is 1/L, so multiplying with diff computes the rate of change
                float fac = std::exp(-diff * edgeWeight/sigma2);
                E_pq = smoothness * fac;
            }
            bool intersected = false;
            for (const auto &edge : guideRays) {
                Segment2d pixelSegment(Eigen::Vector2d(col, row), Eigen::Vector2d(col2, row2));
                double t, t2;
                bool entering;
                if (pixelSegment.intersect(edge, t, t2, entering)) {
                    intersected = true;
                    break;
                }
            }
            if (!intersected) {
                E_pq *= edgeWeight;
                int i1 = row * labels.cols + col;
                int i2 = row2 * labels.cols + col2;

                float iE_pq = E_pq * precision;
//                std::cout << "adding edge " << i1 << "-" << i2 << std::endl;
                graph->add_edge(i1, i2, iE_pq, iE_pq);
            } /*else {
                debugImg.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255);
                debugImg.at<cv::Vec3b>(row2, col2) = cv::Vec3b(0, 255, 0);
            }*/
        }
        /*if (col < labels.cols - 1) {
            graph->add_edge(i, i+1, smoothness, smoothness);
        }
        if (row < labels.rows - 1) {
            graph->add_edge(i, i+labels.cols, smoothness, smoothness);
        }*/
    }
    std::cout << "running flow" << std::endl;
    float flow = graph->maxflow();

    for ( int i = 0; i < num_pixels; i++ ) {
        int col = i % labels.cols;
        int row = i / labels.cols;
        cv::Point point(col, row);
        labels.at<uchar>(point) = graph->what_segment(i);
    }
//    cv::imwrite("graphCut_debug_img_" + std::to_string(imageId++) + ".png", debugImg);
    delete graph;
    return flow;
}