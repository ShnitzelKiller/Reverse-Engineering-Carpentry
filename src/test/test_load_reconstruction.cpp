//
// Created by James Noeckel on 1/21/20.
//
//#include "reconstruction/CameraIntrinsics.h"
//#include "reconstruction/Image.h"
//#include "reconstruction/Points3D.h"
#include "reconstruction/ReconstructionData.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

int main(int argc, char **argv) {
    using namespace Eigen;
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " reconstructionpath [ depthpath [ imagepath [ targetReconstruction ] ]]" << std::endl;
        return 1;
    }
    std::string reconstruction_path = argv[1];
    std::string depth_path="";
    std::string image_path="";
    std::string reconstruction_path2="";
    if (argc >= 3) {
        depth_path = argv[2];
    }
    if (argc >= 4) {
        image_path = argv[3];
    }
    if (argc >= 5) {
        reconstruction_path2 = argv[4];
    }
    ReconstructionData reconstruction;
    bool success;
    if (reconstruction_path.rfind(".out") != std::string::npos) {
        success = reconstruction.load_bundler_file(reconstruction_path, depth_path);
    } else {
        if (image_path.empty()) {
            std::cout << "colmap requires image path" << std::endl;
            return 1;
        }
        success = reconstruction.load_colmap_reconstruction(reconstruction_path, image_path, depth_path);
    }
    if (!success) {
        std::cout << "loading failed" << std::endl;
        return 1;
    }
    if (!reconstruction_path2.empty()) {
        ReconstructionData target;
        if (!target.load_bundler_file(reconstruction_path2)) {
            std::cout << "loading target reconstruction failed" << std::endl;
            return 1;
        }
        double scale;
        Eigen::Quaterniond rot;
        Eigen::Vector3d trans;
        reconstruction.findAffineTransform(target, scale, rot, trans);
        std::cout << "scale: " << scale << std::endl;
        {
            int ind = 0;
            MatrixX3d P(reconstruction.images.size(), 3);
            for (const auto &pair : reconstruction.images) {
//                P.row(ind) = (scale * pair.second.origin() + trans).transpose();
                P.row(ind) = scale * (rot * pair.second.origin()).transpose() + trans.transpose();
                ++ind;
            }
            ind = 0;
            MatrixX3d Q(target.images.size(), 3);
            for (const auto &pair : target.images) {
                Q.row(ind) = pair.second.origin().transpose();
                ++ind;
            }
            igl::opengl::glfw::Viewer viewer;
            viewer.data().add_points(P, RowVector3d(1, 0, 0));
            viewer.data().add_points(Q, RowVector3d(0, 1, 0));
            auto matches = reconstruction.correspondingImages(target);
            for (const auto &pair : matches) {
                viewer.data().add_edges(scale * (rot * reconstruction.images[pair.first].origin()).transpose() + trans.transpose(), target.images[pair.second].origin().transpose(), RowVector3d(1,1,1));
            }
            viewer.launch();
        }
    }
    for (auto &pair : reconstruction.images) {
        cv::Mat img = pair.second.getImage().clone();
        for (auto pid : pair.second.point3D_ids_) {
            Eigen::Vector3d pt = reconstruction.points[pid].xyz_;
            Eigen::Vector2d pt_proj = reconstruction.project(pt.transpose(), pair.first).transpose();
            cv::circle(img, cv::Point(pt_proj.x(), pt_proj.y()), 5, cv::Scalar(0, 255, 0), cv::FILLED);
        }
        for (const auto &xy : pair.second.xys_) {
            cv::circle(img, cv::Point(xy.x(), xy.y()), 3, cv::Scalar(0, 0, 255), cv::FILLED);
        }
        std::string windowname = "image_" + std::to_string(pair.first);
        cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowname, img);
        cv::waitKey(0);
        cv::destroyWindow(windowname);
        cv::Mat depthMap = pair.second.getDepthGeometric();
        if (!depthMap.empty()) {
            cv::Mat depthMapConverted;
            depthMap.convertTo(depthMapConverted, CV_8UC1, 10);
            std::string depthname = windowname + "_depth";
            cv::namedWindow(depthname, cv::WINDOW_AUTOSIZE);
            cv::imshow(depthname, depthMapConverted);
            cv::waitKey(0);
            cv::destroyWindow(depthname);
        }
    }
    return 0;
}