//
// Created by James Noeckel on 1/18/21.
//

#include "utils/settings.h"
#include <iostream>
#include "reconstruction/point_cloud_io.h"
#include "reconstruction/ReconstructionData.h"

#include "construction/Construction.h"
#include <igl/png/writePNG.h>

using namespace Eigen;
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <config>.txt" << std::endl;
        return 1;
    }

    Settings settings;
    if (!settings.parse_file(argv[1])) {
        return 1;
    }

    std::cout << "============= settings =============\n" << settings << "============================\n\n";

    // load data reconstruction data
    ReconstructionData::Handle reconstruction(new ReconstructionData);
    if (!settings.reconstruction_path.empty()) {
        if (settings.reconstruction_path.rfind(".out") != std::string::npos) {
            if (!reconstruction->load_bundler_file(settings.reconstruction_path, settings.depth_path)) {
                std::cerr << "failed to load bundler file " << settings.reconstruction_path << std::endl;
                return 1;
            }
        } else {
            if (!reconstruction->load_colmap_reconstruction(settings.reconstruction_path, settings.image_path,
                                                            settings.depth_path)) {
                std::cerr << "failed to load reconstruction in path " << settings.reconstruction_path << std::endl;
                return 1;
            }
        }
    }

    reconstruction->export_image_resolutions(settings.result_path + "_image_resolutions.txt");

    std::string plaintextPath = settings.result_path + ".txt";
    Construction construction;
    bool success = construction.loadPlaintext(plaintextPath, false, true);
    if (!success) {
        std::cout << "failed to load " << plaintextPath << std::endl;
        return 1;
    }

    construction.computeMeshes(false);
    auto mesh = construction.mesh;
    RowVector3d minPt = std::get<0>(mesh).colwise().minCoeff();
    RowVector3d maxPt = std::get<0>(mesh).colwise().maxCoeff();
    RowVector3d dims = maxPt - minPt;
    double diameter = dims.maxCoeff();
    std::cout << "diameter: " << diameter << std::endl;

    construction.visualize(0);
    return 0;

//    igl::opengl::glfw::Viewer viewer;
//    viewer.data().set_mesh(std::get<0>(mesh), std::get<1>(mesh));
//    viewer.core().align_camera_center(std::get<0>(mesh));
//    viewer.data().set_face_based(true);
//    viewer.launch();
    /*viewer.core().camera_eye = Eigen::Vector3f(0.f, 0.f, 3.f);

    {
        int outWidth=1280, outHeight=800;
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(outWidth,outHeight);
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(outWidth,outHeight);
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(outWidth,outHeight);
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(outWidth,outHeight);
        viewer.launch_init();
        viewer.draw();
        viewer.core().draw_buffer(viewer.data(),false,R,G,B,A);
        std::cout << (int)R.minCoeff() << ", " << (int)R.maxCoeff() << std::endl;
        std::cout << (int)G.minCoeff() << ", " << (int)G.maxCoeff() << std::endl;
        std::cout << (int)B.minCoeff() << ", " << (int)B.maxCoeff() << std::endl;
        std::cout << (int)A.minCoeff() << ", " << (int)A.maxCoeff() << std::endl;
        igl::png::writePNG(R,G,B,A,"outTest.png");
        viewer.launch_shut();
    }*/
//    viewer.core().camera_dnear = diameter/100;
//    viewer.core().camera_dfar = diameter * 10;
//
//    std::cout << "rendering images" << std::endl;
//    for (const auto &pair : reconstruction->images) {
//        if (pair.first == 69) continue; //DEBUG
//        std::cout << "view " << pair.first << std::endl;
//        const CameraIntrinsics &camera = reconstruction->cameras[pair.second.camera_id_];
//        int width = camera.width_;
//        int height = camera.height_;
//        Eigen::Array2d fdist(camera.params_.data());
//        float fov = 2 * std::atan(static_cast<float>(width)/2/static_cast<float>(fdist[0])) * 180/static_cast<float>(M_PI);
//        std::cout << "fov: " << fov << std::endl;
//        Eigen::Array2d principal_point(camera.params_.data() + 2);
//        const Image &image = pair.second;
//        Vector3d target = image.origin() + image.direction() * diameter/2;
//        Vector3d up = image.rot_.conjugate() * Eigen::Vector3d(0, 0, 1);
//        viewer.core().camera_eye = image.origin().cast<float>();
//        std::cout << "origin: " << viewer.core().camera_eye.transpose() << std::endl;
//        viewer.core().camera_center = target.cast<float>();
//        viewer.core().camera_up = up.cast<float>();
//        viewer.core().camera_view_angle = fov;
//        viewer.core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_NO_ROTATION);
//        viewer.launch();

        /*int outWidth=width/8, outHeight=height/8;
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(outWidth,outHeight);
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(outWidth,outHeight);
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(outWidth,outHeight);
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(outWidth,outHeight);
        std::cout << "drawing buffer" << std::endl;

        viewer.launch_init();
        viewer.draw();

        viewer.core().draw_buffer(viewer.data(),false,R,G,B,A);
        igl::png::writePNG(R,G,B,A,"out"+std::to_string(pair.first)+".png");
        viewer.launch_shut();*/
//    }


    return 0;
}