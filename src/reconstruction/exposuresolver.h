//
// Created by James Noeckel on 10/12/20.
//

#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

enum {
    LOSS_L2,
    LOSS_CAUCHY,
    LOSS_HUBER,
};

/**
 *
 * @param images rectified images as CV_8UC3
 * @param occlusionMasks rectified masks where 0 means the pixel is occluded in this view
 * @param constraintMask rectified mask of pixels to use across all (unoccluded) images to optimize the radiance and exposure
 * @param imageIndices list of indices of image, mask pairs to consider as observations of the same radiance
 * @param exposures per-channel exposures for each image (index = i + ch * n for i=0:n-1, ch=0:2)
 * @param radiances
 * @param lossfn
 * @param scale
 * @return
 */
double solveExposure(
        //MeshManager* manager,
        const std::vector<cv::Mat> &images,
        const std::vector<cv::Mat> &occlusionMasks,
        const std::vector<int> &imageIndices,
        const cv::Mat &constraintMask,
        double* exposures,
        double* radiances,
        int lossfn=LOSS_L2,
        float scale=5.0
);