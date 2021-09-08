//
// Created by James Noeckel on 9/29/20.
//

#pragma once
#include <Eigen/Dense>
#include "reconstruction/CameraIntrinsics.h"

/**
 * Compute homography matrix mapping points on a plane to pixel locations given camera pose and plane position
 * @param planeRot plane->world
 * @param planeTrans plane->world
 * @param planeOrigin origin in plane coordinates
 * @param planeDims dimensions of the plane in world units
 * @param planeRes resolution of plane image
 * @param camRot world->camera
 * @param camTrans world->camera
 * @param intrinsics camera intrinsics
 * @param scale additional scale factor of images
 * @return
 */
Eigen::Matrix3d homographyFromPlane(const Eigen::Quaterniond &planeRot, const Eigen::Ref<const Eigen::Vector3d> &planeTrans,
                                    const Eigen::Ref<const Eigen::Vector2d> &planeOrigin,
                                    const Eigen::Ref<const Eigen::Vector2d> &planeDims,
                                    const Eigen::Ref<const Eigen::Vector2i> &planeRes,
                                    const Eigen::Quaterniond &camRot, const Eigen::Ref<const Eigen::Vector3d> &camTrans,
                                    const CameraIntrinsics &intrinsics,
                                    double scale=1.0);