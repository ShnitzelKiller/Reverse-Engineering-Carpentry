//
// Created by James Noeckel on 10/21/20.
//

#include "geometry/primitives3/SurfaceCompletion.h"
#include <iostream>
/*#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>*/
#include "utils/timingMacros.h"
#include "utils/visualization.hpp"
#include "test/testUtils/displaySurfaceCompletion.h"


int main(int argc, char **argv) {
    using namespace Eigen;
    double spacing = 0.05;
    Vector3d minPt(-2, -1.5, -1.75);
    Vector3d maxPt(2, 1.5, 1.75);
    /*{
        SurfaceCompletion surf(minPt, maxPt, spacing);
        auto resolution = surf.resolution();
        std::cout << "resolution: " << resolution.transpose() << std::endl;
        std::cout << "setting segmentation" << std::endl;
        std::vector<bool> segmentation(resolution.prod());
        int num_cells = 0;
        for (int i = 0; i < resolution.x(); ++i) {
            for (int j = 0; j < resolution.y(); ++j) {
                for (int k = 0; k < resolution.z(); ++k) {
                    Vector3d pos = minPt + spacing * Vector3d(i, j, k);
                    int ind = i * resolution.y() * resolution.z() + j * resolution.z() + k;
                    if (pos.norm() > 1) {
                        segmentation[ind] = false;
                    } else {
                        segmentation[ind] = true;
                        ++num_cells;
                    }
                }
            }
        }
        std::cout << "set " << num_cells << " inside cells" << std::endl;
        std::cout << "constructing problem" << std::endl;
        surf.constructProblem();
        std::cout << "finished" << std::endl;

        if (!surf.setSegmentation(segmentation)) {
            std::cout << "failed to set segmenation" << std::endl;
            return 1;
        }
        std::cout << "computing cost" << std::endl;
        double cost = surf.getCurrentCost();
        double sphereArea = 4 * M_PI;
        std::cout << "cost of unit sphere: " << cost << std::endl;
        std::cout << "percent error: " << (cost - sphereArea) / sphereArea << std::endl;
        std::cout << "ratio: " << cost / sphereArea << std::endl;
        displaySegmentation(surf);

        std::cout << "setting segmentation" << std::endl;
        for (int i = 0; i < resolution.x(); ++i) {
            for (int j = 0; j < resolution.y(); ++j) {
                for (int k = 0; k < resolution.z(); ++k) {
                    Vector3d pos = minPt + spacing * Vector3d(i, j, k);
                    int ind = i * resolution.y() * resolution.z() + j * resolution.z() + k;
                    if (pos.head(2).norm() > 1 || std::abs(pos.z()) > 1) {
                        segmentation[ind] = false;
                    } else {
                        segmentation[ind] = true;
                        ++num_cells;
                    }
                }
            }
        }

        if (!surf.setSegmentation(segmentation)) {
            std::cout << "failed to set segmenation" << std::endl;
            return 1;
        }
        std::cout << "computing cost" << std::endl;
        cost = surf.getCurrentCost();
        double cylinderArea = 6 * M_PI;
        std::cout << "cost of cylinder (radius 1, height 2: " << cost << std::endl;
        std::cout << "percent error: " << (cost - cylinderArea) / cylinderArea << std::endl;
        std::cout << "ratio: " << cost / cylinderArea << std::endl;
        displaySegmentation(surf);
    }*/
    {
        SurfaceCompletion surf(minPt, maxPt, spacing);
        std::vector<BoundedPlane> primitives;
        std::shared_ptr<Primitive> shape(new Bbox(Vector2d(-0.8, -0.8), Vector2d(0.8, 0.8)));
        primitives.emplace_back(shape->clone(), (Matrix3d() << 1, 0, 0,
                                                                0, 1, 0,
                                                                0, 0, 1).finished(), -1);
        primitives.emplace_back(shape->clone(), (Matrix3d() << 0, 1, 0,
                0, 0, 1,
                1, 0, 0).finished(), -1);
        primitives.emplace_back(shape->clone(), (Matrix3d() << 0, 0, 1,
                1, 0, 0,
                0, 1, 0).finished(), -1);

        //backwards
        primitives.emplace_back(shape->clone(), (Matrix3d() << 1, 0, 0,
                0, -1, 0,
                0, 0, -1).finished(), -1);
        primitives.emplace_back(shape->clone(), (Matrix3d() << 0, 1, 0,
                0, 0, -1,
                -1, 0, 0).finished(), -1);
        primitives.emplace_back(shape->clone(), (Matrix3d() << 0, 0, 1,
                -1, 0, 0,
                0, -1, 0).finished(), -1);
        for (auto &primitive : primitives) {
            primitive.setCurrentShape(0);
        }
        Visualizer visualizer;
        visualizer.visualize_primitives(primitives, std::vector<Cylinder>());
        surf.setPrimitives(std::move(primitives));
        auto resolution = surf.resolution();
        std::cout << "constructing problem" << std::endl;
        surf.constructProblem();
        std::cout << "inside constraints: " << surf.insideConstraints().size() << std::endl;
        std::cout << "outside constraints: " << surf.outsideConstraints().size() << std::endl;
        //displaySegmentation(surf, visualizer, VIS_INSIDE);
        //visualizer.launch();
        //displaySegmentation(surf, visualizer, VIS_OUTSIDE);
        //visualizer.launch();
        std::cout << "running max flow algorithm" << std::endl;
        DECLARE_TIMING(maxflow);
        START_TIMING(maxflow);
        double cost = surf.maxflow();
        STOP_TIMING(maxflow);
        std::cout << "finished in " << GET_TIMING(maxflow) << "s with " << cost << " max flow" << std::endl;
        displaySegmentation(surf, visualizer, VIS_SURFACE, true);
        visualizer.launch();
    }
    return 0;
}