//
// Created by James Noeckel on 1/16/21.
//

#include "convert_contour.h"
#include "utils/io/csvFormat.h"
#include "utils/vstack.h"
#include <fstream>

using namespace Eigen;

std::shared_ptr<Primitive> convertContour(const std::vector<std::vector<Vector2d>> &marching_squares_contours, std::vector<std::vector<int>> &hierarchy, double minSizeRatio) {
    if (!hierarchy.back().empty()) {
        int max_outer_contour_index = *std::max_element(hierarchy.back().begin(), hierarchy.back().end(),
                                                        [&](int a, int b) {
                                                            return marching_squares_contours[a].size() <
                                                                   marching_squares_contours[b].size();
                                                        });
        const std::vector<Vector2d> &max_outer_contour = marching_squares_contours[max_outer_contour_index];
        MatrixX2d contour_eig = vstack(max_outer_contour);
//        if (visualize) {
//            std::ofstream file("part_" + std::to_string(partID) + ".csv");
//            file << contour_eig.format(CSVFormat);
//        }
        //VISUALIZE
//        vis_shapes_raw[partID].emplace_back();
//        for (int p = 0; p < contour_eig.rows(); p++) {
//            vis_shapes_raw[partID].back().push_back(planes[partID].points3D(contour_eig.row(p)).transpose());
//        }
        std::vector<std::shared_ptr<Primitive>> holes;
        for (int child_ind : hierarchy[max_outer_contour_index]) {
            const std::vector<Vector2d> &child_contour = marching_squares_contours[child_ind];
            //TODO: use a configurable threshold for finding holes inside contours, and maybe use areas
            if (child_contour.size() * minSizeRatio > max_outer_contour.size()) {
                MatrixX2d hole_eig = vstack(child_contour);
                holes.emplace_back(new Polygon(hole_eig));
//                if (visualize) {
//                    vis_shapes_raw[partID].emplace_back();
//                    for (int p = 0; p < hole_eig.rows(); p++) {
//                        vis_shapes_raw[partID].back().push_back(planes[partID].points3D(hole_eig.row(p)).transpose());
//                    }
//                }
            }
        }
        if (holes.empty()) {
            return std::make_shared<Polygon>(std::move(contour_eig));
        } else {
            //std::cout << "found " << holes.size() << " inner contours";
            return std::make_shared<PolygonWithHoles>(std::move(contour_eig), std::move(holes));
        }
    }
    return std::shared_ptr<Primitive>();
}