//
// Created by James Noeckel on 9/29/20.
//

#include "homographyFromPlane.h"
#include <iostream>

using namespace Eigen;

Matrix3d homographyFromPlane(const Quaterniond &planeRot, const Ref<const Vector3d> &planeTrans,
                             const Ref<const Vector2d> &planeOrigin,
                             const Ref<const Vector2d> &planeDims,
                             const Ref<const Vector2i> &planeRes,
                             const Quaterniond &camRot, const Ref<const Vector3d> &camTrans,
                             const CameraIntrinsics &intrinsics,
                             double scale) {
    Array2d planeScale = planeDims.array() / planeRes.array().cast<double>();
    Array2d fdist(intrinsics.params_.data());
    Vector2d principal_point(intrinsics.params_.data() + 2);
    if (scale != 1) {
        fdist *= scale;
        principal_point *= scale;
    }
    {
        Matrix4d Mcam = Matrix4d::Zero();
        Mcam.block<3, 3>(0, 0) = camRot.matrix();
        Mcam.block<3, 1>(0, 3) = camTrans;
        Mcam(3, 3) = 1;
//        std::cout << "Mcam: " << std::endl << Mcam << std::endl;

        Matrix4d Mplane = Matrix4d::Zero();
        Mplane.block<3, 3>(0, 0) = planeRot.matrix();
        Mplane.block<3, 1>(0, 3) = planeTrans;
        Mplane(3, 3) = 1;
//        std::cout << "Mplane: " << std::endl << Mplane << std::endl;

        Matrix4d planeUnproj = Matrix4d::Zero();
        planeUnproj(0, 0) = planeScale.x();
        planeUnproj(1, 1) = planeScale.y();
        planeUnproj.block<2, 1>(0, 3) = planeOrigin;
        planeUnproj(3, 3) = 1;
//        std::cout << "planeUnproj: " << std::endl << planeUnproj << std::endl;

        Matrix4d T = Mcam * Mplane * planeUnproj;
//        std::cout << "T: " << std::endl << T << std::endl;

        Matrix3d H;
        H.block<3, 2>(0, 0) = T.block<3, 2>(0, 0);
        H.col(2) = T.block<3, 1>(0, 3);
//        std::cout << "H: " << std::endl << H << std::endl;

        Matrix3d proj = Matrix3d::Zero();
        proj(0, 0) = fdist.x();
        proj(1, 1) = fdist.y();
        proj.block<2, 1>(0, 2) = principal_point;
        proj(2, 2) = 1;
//        std::cout << "proj: " << std::endl << proj << std::endl;
        Matrix3d Htrans = proj * H;
        Htrans /= Htrans(2, 2);
        //std::cout << "Htrans: " << std::endl << Htrans << std::endl;
        return Htrans;

    }
    /*Matrix3d R = (camRot * planeRot).matrix();
    Vector3d t = camRot * planeTrans + camTrans;
    //account for plane pixel coordinates
    R.block<3, 2>(0, 0).array().rowwise() *= planeScale.transpose();
    t += R.block<3, 2>(0, 0) * planeOrigin;
    //convert R to a homography
    R.col(2) = t;
    //apply projection matrix
    R.block<2, 3>(0, 0).array().colwise() *= fdist;
    R.block<2, 3>(0, 0) += principal_point * R.row(2);
    R /= R(2, 2);
    std::cout << "R: " << std::endl << R << std::endl;
    return R;*/
}