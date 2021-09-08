//
// Created by James Noeckel on 10/8/20.
//

#include "Solver.h"
#include "geometry/homographyFromPlane.h"
#include "utils/eigenMatToCV.h"
#include <geometry/primitives3/BoundedPlane.h>
#include <utils/timingMacros.h>
#include <imgproc/graph_cut.h>
#include <utils/io/csvFormat.h>
#include "math/GaussianMixture.h"
#include "geometry/shapes2/VoxelGrid.hpp"
#include "utils/vstack.h"
//#include "imgproc/matching_scores.h"
#include "imgproc/dda_foreach.h"
//#include "reconstruction/exposuresolver.h"

//#define MIN_CORR 0.3
//#define GAMMA 2.2f
//#define MAX_DISPARITY 10.0
//#define EXPOSURE_SAMPLES 1000
//#define MIN_PROB 1e-11f
#define MAX_ENERGY 1000000
//#define CONTOUR_SCALE 1.0
#define NUM_ITERATIONS 2

using namespace Eigen;
using namespace boost;

void displayProbabilityMap(const cv::Mat &energy, const std::string &name) {
    std::vector<cv::Mat> avgProbImgs(2, cv::Mat(energy.rows, energy.cols, CV_32FC1));
    for (int cluster = 0; cluster <= 1; ++cluster) {
        for (int pixel=0; pixel<energy.total(); ++pixel) {
            avgProbImgs[cluster].at<float>(pixel) = std::exp(-energy.at<cv::Vec<float, 2>>(pixel)[cluster]);
        }
        cv::Mat debugAvgProb2;
        double minVal;
        double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;
        cv::minMaxLoc(avgProbImgs[cluster], &minVal, &maxVal, &minLoc, &maxLoc);

        avgProbImgs[cluster].convertTo(debugAvgProb2, CV_8UC1, 255.0 / maxVal);
        cv::imwrite(name + std::to_string(cluster) +
                           "_avg_scale_maxval_" +
                           std::to_string(maxVal) + ".png", debugAvgProb2);
    }
}

/**
 *
 * @tparam T
 * @param img
 * @param inValue inside value
 * @param outValue outside value
 * @param ptA start point in pixel coordinates
 * @param ptB end point in pixel coordinates
 * @param spacing space between inside and outside constraint
 * @param inFatness
 * @param outFatness
 */
template <typename T>
void paintConstraint(cv::Mat &img, const T& inValue, const T& outValue, const cv::Point2d &ptA, const cv::Point2d &ptB, float spacing, float inFatness, float outFatness) {
    Vector2d dir(ptB.x - ptA.x, ptB.y - ptA.y);
//    std::cout << "dir: " << dir.transpose() << std::endl;
    double length = dir.norm();
    dir /= length;
    //right facing normal (outward)
    Vector2d n(dir.y(), -dir.x());
//    std::cout << "n: " << n << std::endl;
    cv::Point2d nCv(n.x(), n.y());
//    std::cout << "ncv: " << nCv << std::endl;
    if (outFatness > 0) {
        cv::Point2d ptAO = ptA + nCv * (spacing / 2);
        cv::Point2d ptBO = ptB + nCv * (spacing / 2);
        std::vector<cv::Point> outsidePts = {ptAO, ptBO, ptBO + nCv * outFatness, ptAO + nCv * outFatness};
//        std::cout << "outFatness: " << outFatness << std::endl;
//        std::cout << "outsidePts: " << outsidePts[0] << ", " << outsidePts[1] << ", " << outsidePts[2] << ", " << outsidePts[3] << std::endl;
        cv::fillConvexPoly(img, outsidePts.data(), outsidePts.size(), outValue);
    }
    if (inFatness > 0) {
        cv::Point2d ptAO = ptA - nCv * (spacing / 2);
        cv::Point2d ptBO = ptB - nCv * (spacing / 2);
        std::vector<cv::Point> insidePts = {ptAO, ptBO, ptBO - nCv * inFatness, ptAO - nCv * inFatness};
        cv::fillConvexPoly(img, insidePts.data(), insidePts.size(), inValue);
    }
}

int Solver::shapeFromImages(bool useConstraints) {
//    segmentationResults.clear();
    reconstruction_->setImageScale(1);
    double maskScale = 0.5; //TODO: Parameter
    double maskScaleInv = 1./maskScale;
    double voxel_width = diameter_ / settings_.voxel_resolution;
    double margin = 5 * voxel_width;
    double imageMargin = 7 * voxel_width;
    const float maxEnergy = 10 * MAX_ENERGY;

    std::unordered_map<Construction::Vertex, std::vector<std::shared_ptr<Polygon>>> newShapes;

//    Construction::IndexMap index = get(vertex_index, construction.g);
    Construction::vertex_iter vi, vend;
    //TODO: run outer loop in parallel
    size_t N = num_vertices(construction.g);
    for (boost::tie(vi, vend) = vertices(construction.g); vi != vend; ++vi) {
        Construction::Vertex v = *vi;
        size_t partIdx = construction.g[v].partIdx;
        const PartData &pd = construction.partData[partIdx];
        if (pd.groundPlane) continue;
        auto &shape = construction.getShape(partIdx);
        auto &connectionConstraints = shape.connectionConstraints;
        auto &shapeConstraints = shape.shapeConstraints;
        auto &guideConstraints = shape.guides;
        if (shape.dirty) {
            std::cout << "analyzing part " << partIdx << std::endl;
            Vector3d dirU = pd.rot * Vector3d(1, 0, 0);
            Vector3d dirV = pd.rot * Vector3d(0, 1, 0);
            Vector2d minPt;
            Vector2d maxPt;
            {
                MatrixX2d points = shape.cutPath->points();
                minPt =
                        points.colwise().minCoeff().transpose().array() -
                        imageMargin;
                maxPt =
                        points.colwise().maxCoeff().transpose().array() +
                        imageMargin;
            }
            double thickness = construction.getStock(partIdx).thickness;
            Vector3d normal = pd.normal();
            Vector3d posBack = pd.pos - normal * thickness;
            Vector3d faceCenter = pd.unproject(((minPt + maxPt)/2).transpose()).transpose();
            Vector3d backFaceCenter = faceCenter - normal * thickness;
            double offset = pd.offset();
            double offsetBack = offset + thickness;
            Vector2d planeDims = maxPt - minPt;

            /** all views, front and back, containing (view, dotprod, backside) */
            std::vector<std::tuple<int32_t, double, bool>> allViews;
            /** indices into allViews */
            std::vector<size_t> viewIndices;
            for (const auto &view : cluster_visibility[partIdx]) {
                allViews.emplace_back(view.first, view.second, false);
            }
            //if (construction.partData[partIdx].bothSidesObserved) {
                std::cout << "adding backface views" << std::endl;
                for (const auto &view : cluster_visibility[partIdx + construction.partData.size()]) {
                    allViews.emplace_back(view.first, view.second, true);
                }
            //}
            std::sort(allViews.begin(), allViews.end(),
                      [](const auto &a, const auto &b) { return std::get<1>(a) > std::get<1>(b); });

            {
                std::cout << "choosing best views..." << std::endl;
                int previewWidth = 100;
                int previewHeight = static_cast<int>(std::round(previewWidth / planeDims.x() * planeDims.y()));
                double imgToWorldScale = planeDims.x() / previewWidth;

                for (size_t viewIndex = 0;
                     viewIndex < allViews.size() && viewIndices.size() < settings_.max_views_per_cluster; ++viewIndex) {
                    const auto &view = allViews[viewIndex];
                    if (std::get<1>(view) < settings_.max_view_angle_cos) continue;
                    auto &image = reconstruction_->images[std::get<0>(view)];
                    Vector3d rayOrigin = image.origin();
                    BoundedPlane plane(normal, std::get<2>(view) ? offsetBack : offset);

                    //TODO: skip if views are occluded
                    size_t shapeTotal = 0;
                    size_t occluded = 0;
                    size_t nloop = previewWidth * previewHeight;
                    for (size_t ind = 0; ind < nloop; ++ind) {
                        size_t i = ind / previewWidth;
                        size_t j = ind % previewWidth;
                        Vector2d pt2d(j * imgToWorldScale + minPt.x(),
                                      i * imgToWorldScale + minPt.y());

                        if (shape.cutPath->contains(pt2d)) {
                            Vector3d pt3d = pd.pos + pd.rot * Vector3d(pt2d.x(), pt2d.y(),
                                                                       std::get<2>(view) ? -construction.getStock(
                                                                               partIdx).thickness : 0);
                            ++shapeTotal;

                            Vector3d rayDir = (pt3d - rayOrigin).normalized();
                            double raycos = std::abs(rayDir.dot(plane.normal()));
                            igl::Hit hit;
                            bool intersected = construction.aabb.intersect_ray(std::get<0>(construction.mesh),
                                                                               std::get<1>(construction.mesh),
                                                                               rayOrigin.transpose(),
                                                                               rayDir.transpose(), hit);
                            if (intersected) {
                                int intersectedPart = std::get<2>(construction.mesh)(hit.id);
                                if (intersectedPart != partIdx) {
                                    double t;
                                    if (plane.intersectRay(rayOrigin, rayDir, t)) {
                                        if (hit.t * raycos < t * raycos - voxel_width * 3) {
                                            ++occluded;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    std::cout << "view " << std::get<0>(view) << ": " << occluded << '/' << shapeTotal << " occluded"
                              << std::endl;
                    if (occluded * 2 > shapeTotal) {
                        continue;
                    }
                    viewIndices.push_back(viewIndex);
                }
            }

            if (viewIndices.empty()) {
                std::cout << "no viable views found for part " << partIdx << "; skipping" << std::endl;
                continue;
            }
            std::cout << "views chosen: ";
            for (const auto &viewIndex : viewIndices) {
                std::cout << std::get<0>(allViews[viewIndex]) << ", ";
            }
            std::cout << std::endl;

            double meanPixelRate = 0.0;
            double totalWeight = 0;
            // use pixel derivatives at shape center to determine resolution of plane image
            for (auto viewIndex : viewIndices) {
                const auto &view = allViews[viewIndex];
                const auto &eitherFaceCenter = std::get<2>(view) ? backFaceCenter : faceCenter;
                double dxdu = reconstruction_->directionalDerivative(eitherFaceCenter, dirU, std::get<0>(view)).norm();
                double dxdv = reconstruction_->directionalDerivative(eitherFaceCenter, dirV, std::get<0>(view)).norm();
                double weight = std::abs(
                        (reconstruction_->images[std::get<0>(view)].origin() - eitherFaceCenter).normalized().dot(
                                normal));
                meanPixelRate += weight * (dxdu + dxdv) * 0.5;
                totalWeight += weight;
            }

            meanPixelRate /= totalWeight;
            double worldToImgScale = meanPixelRate * settings_.image_scale;
            Vector2i resolution = (planeDims.array() * worldToImgScale).ceil().cast<int>();
            {
                if (planeDims.x() / resolution.x() > voxel_width / settings_.min_pixels_per_voxel) {
                    double newX = planeDims.x() / voxel_width * settings_.min_pixels_per_voxel;
                    double fac = newX / resolution.x();
                    resolution = Vector2i(static_cast<int>(std::ceil(fac * resolution.x())),
                                          static_cast<int>(std::ceil(fac * resolution.y())));
                    worldToImgScale *= fac;
                }
            }
            {
                int maxres = std::max(resolution.x(), resolution.y());
                if (maxres > settings_.max_image_resolution) {
                    double fac = settings_.max_image_resolution / static_cast<double>(maxres);
                    resolution = Vector2i(static_cast<int>(std::ceil(fac * resolution.x())),
                                          static_cast<int>(std::ceil(fac * resolution.y())));
                    worldToImgScale *= fac;
                }
            }
            cv::Size imageSize(resolution.x(), resolution.y());
            double imgToWorldScale = 1.0 / worldToImgScale;
//        double imgToWorldNoiseScale = imgToWorldScale / (diameter_/settings_.master_resolution);
//        double imgToWorldNoiseScale2 = imgToWorldNoiseScale * imgToWorldNoiseScale;
            std::cout << "image to world scale: " << imgToWorldScale << std::endl;

            //initial guess mask using eroded point density shape (which is conservative and rarely overestimates boundaries)
            /** 0 = part, 1 = background, 2 = unknown */
            cv::Mat initMask = cv::Mat::ones(imageSize, CV_8UC1) * 2;
            cv::Mat knownBGMask;
            cv::Mat knownFGMask;

            size_t numForegroundPixels = 0;
            {
                /** 1 = known foreground */
                cv::Mat foregroundMask = cv::Mat::zeros(imageSize, CV_8UC1);
                /** 1 = known background */
                cv::Mat bgMask = cv::Mat::zeros(imageSize, CV_8UC1);

                /** 1 = known hole */
                cv::Mat holeMask = cv::Mat::zeros(imageSize, CV_8UC1);
                for (int i = 0; i < initMask.rows; ++i) {
                    for (int j = 0; j < initMask.cols; ++j) {
                        Vector2d pt = Array2d(j, i) * imgToWorldScale + minPt.array();
                        if (shape.cutPath->contains(pt)) {
                            foregroundMask.at<uchar>(i, j) = 1;
                            ++numForegroundPixels;
                        } else {
                            bgMask.at<uchar>(i, j) = 1;
                        }
                        for (const auto &child : shape.cutPath->children()) {
                            if (child->contains(pt)) {
                                holeMask.at<uchar>(i, j) = 1;
                            }
                        }
                    }
                }
                {
                    int dilation = static_cast<int>(std::ceil(voxel_width * worldToImgScale * 0.5));
                    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT,
                                                                           cv::Size(2 * dilation + 1, 2 * dilation + 1),
                                                                           cv::Point(dilation, dilation));
                    cv::erode(foregroundMask, foregroundMask, structuringElement);
                    cv::erode(bgMask, bgMask, structuringElement);
//                    cv::erode(holeMask, holeMask, structuringElement);
                }
                //remove bgmask in guide regions (leave unknown to allow solution to choose)
                for (const auto &pair : guideConstraints) {
                    if (pair.first < 0 || pair.first > construction.partData.size()) continue;
                    for (const auto &guide : pair.second) {
                        std::cout << "erasing BGMask " << partIdx << " using guide from part " << pair.first
                                  << std::endl;
                        Vector2d pixelStart = (guide.edge.first - minPt) * worldToImgScale;
                        Vector2d pixelEnd = (guide.edge.second - minPt) * worldToImgScale;
                        cv::Point2d ptA(pixelStart.x(), pixelStart.y());
                        cv::Point2d ptB(pixelEnd.x(), pixelEnd.y());
                        float inFatness = static_cast<float>(construction.getStock(pair.first).thickness * worldToImgScale * 2);
                        paintConstraint(bgMask, static_cast<uchar>(0), static_cast<uchar>(0), ptA, ptB,
                                        0, inFatness, 0);
                    }
                }
                //add insets due to connections to fg and bg masks
                for (const auto &pair : connectionConstraints) {
                    for (const auto &constraint : pair.second) {
                        if (constraint.outside) {
                            std::cout << "erasing initMask " << partIdx << " using connection constraint from part " << pair.first
                                      << std::endl;
                            double pixelMargin = std::max(1.0, constraint.margin * worldToImgScale);
                            Vector2d pixelStart = (constraint.edge.first - minPt) * worldToImgScale;
                            Vector2d pixelEnd = (constraint.edge.second - minPt) * worldToImgScale;
                            cv::Point2d ptA(pixelStart.x(), pixelStart.y());
                            cv::Point2d ptB(pixelEnd.x(), pixelEnd.y());
                            double outFatness = construction.getStock(pair.first).thickness * worldToImgScale * 2;
                            paintConstraint(foregroundMask, static_cast<uchar>(0), static_cast<uchar>(0), ptA, ptB,
                                            pixelMargin, 0, outFatness);
                            paintConstraint(bgMask, static_cast<uchar>(0), static_cast<uchar>(1), ptA, ptB,
                                            pixelMargin, 0, outFatness);
                        }
                    }
                }
                //add back holes
                if (settings_.debug_visualization) {
                    cv::imwrite("part_" + std::to_string(partIdx) + "_holeMask.png", holeMask * 127);
                }
                //combine
                initMask = 2 * (1 - (bgMask | foregroundMask)) + bgMask;
                if (settings_.debug_visualization) {
                    cv::imwrite("part_" + std::to_string(partIdx) + "_initMask.png", initMask * 127);
                }
                knownBGMask = (initMask == 1) / 255;
                knownFGMask = (initMask == 0) / 255;
                {
                    int dilation = static_cast<int>(std::ceil(margin * worldToImgScale));
                    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT,
                                                                           cv::Size(2 * dilation + 1, 2 * dilation + 1),
                                                                           cv::Point(dilation, dilation));
                    cv::erode(knownBGMask, knownBGMask, structuringElement);
                }
                {
                    int dilation = static_cast<int>(std::ceil(voxel_width * worldToImgScale));
                    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT,
                                                                           cv::Size(2 * dilation + 1, 2 * dilation + 1),
                                                                           cv::Point(dilation, dilation));
                    cv::erode(knownFGMask, knownFGMask, structuringElement);
                }
                knownBGMask = knownBGMask | holeMask;
                if (settings_.debug_visualization) {
                    cv::imwrite("part_" + std::to_string(partIdx) + "_knownBGMask_final.png", knownBGMask * 127);
                    cv::imwrite("part_" + std::to_string(partIdx) + "_knownFGMask_final.png", knownFGMask * 127);
                }
            }
//        std::vector<cv::Mat> linearizedImages;
            //data for each viewIndex in viewIndices
            std::vector<cv::Mat> allMasks;
            /** 1 if ray hits no part of the model */
            std::vector<cv::Mat> approxBGMasks;
            std::vector<Matrix3d> homographies;
            std::vector<cv::Mat> allImages;
            {
                for (auto viewIndex : viewIndices) {
                    auto viewIdx = std::get<0>(allViews[viewIndex]);
                    bool backside = std::get<2>(allViews[viewIndex]);
                    std::stringstream name;
                    name << "part_" << partIdx << "_view_" << viewIdx;
                    if (backside) name << "_backside";
                    std::cout << "processing " << name.str() << std::endl;
                    Image &cameraView = reconstruction_->images[viewIdx];
                    cv::Mat img = cameraView.getImage();
                    Vector3d rayOrigin = cameraView.origin();

                    Matrix3d H = homographyFromPlane(pd.rot, backside ? posBack : pd.pos, minPt, planeDims,
                                                     resolution,
                                                     cameraView.rot_,
                                                     cameraView.trans_,
                                                     reconstruction_->cameras[cameraView.camera_id_]);
                    BoundedPlane plane(normal, backside ? offsetBack : offset);

                    cv::Mat cvH = eigenMatToCV(H);
                    //std::cout << "homography: " << std::endl << cvH << std::endl;
                    cv::Mat imgWarped;
                    cv::warpPerspective(img, imgWarped, cvH, imageSize,
                                        cv::INTER_CUBIC | cv::WARP_INVERSE_MAP);
                    cv::Mat maskWarped = cv::Mat::ones(
                            cv::Size(resolution.x() * maskScale, resolution.y() * maskScale),
                            CV_8UC1);
                    cv::Mat approxBGMaskWarped = cv::Mat::zeros(
                            cv::Size(resolution.x() * maskScale, resolution.y() * maskScale),
                            CV_8UC1);
                    {
                        std::cout << "computing occlusion mask" << std::endl;
                        DECLARE_TIMING(occlusion);
                        START_TIMING(occlusion);
//                            const size_t nthreads = omp_get_max_threads();
//                            std::cout << "parallel (" << nthreads << " threads):" << std::endl;
                        //std::vector<size_t> shapeCounts(nthreads, 0);
                        size_t nloop = maskWarped.rows * maskWarped.cols;
#pragma omp parallel for default(none) shared(maskWarped, approxBGMaskWarped, H, maskScaleInv, img, rayOrigin, plane, partIdx, viewIdx, nloop, voxel_width, initMask)
                        for (size_t ind = 0; ind < nloop; ++ind) {
                            size_t i = ind / maskWarped.cols;
                            size_t j = ind % maskWarped.cols;
                            Vector3d planePix(j * maskScaleInv, i * maskScaleInv, 1.0);
                            Vector3d imgPix = H * planePix;
                            imgPix /= imgPix(2);
                            if (imgPix.x() < 0 || imgPix.y() < 0 || imgPix.x() >= img.cols ||
                                imgPix.y() >= img.rows) {
                                maskWarped.at<uchar>(i, j) = 0;
                                approxBGMaskWarped.at<uchar>(i, j) = 1;
                            }
                            Vector3d rayDir = reconstruction_->initRay(imgPix.y(), imgPix.x(),
                                                                       viewIdx);
                            double raycos = std::abs(rayDir.dot(plane.normal()));
                            igl::Hit hit;
                            bool intersected = construction.aabb.intersect_ray(std::get<0>(construction.mesh),
                                                                               std::get<1>(construction.mesh),
                                                                               rayOrigin.transpose(),
                                                                               rayDir.transpose(), hit);
                            if (intersected) {
                                int intersectedPart = std::get<2>(construction.mesh)(hit.id);
                                if (construction.partData[intersectedPart].groundPlane) {
                                    approxBGMaskWarped.at<uchar>(i, j) = 1;
                                } else if (intersectedPart != partIdx) {
                                    double t;
                                    if (plane.intersectRay(rayOrigin, rayDir, t)) {
                                        if (hit.t * raycos < t * raycos - voxel_width * 3) {
                                            maskWarped.at<uchar>(i, j) = 0;
                                        }
                                    }
                                }
                            } else {
                                approxBGMaskWarped.at<uchar>(i, j) = 1;
                            }
                        }
                        STOP_TIMING(occlusion);
                        PRINT_TIMING(occlusion);
                    }
                    cv::resize(maskWarped, maskWarped, imageSize, 0, 0, cv::INTER_NEAREST);
                    cv::resize(approxBGMaskWarped, approxBGMaskWarped, imageSize, 0, 0, cv::INTER_NEAREST);
                    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5),
                                                                cv::Point(2, 2));
                    cv::Mat occlusionMaskEroded;
                    cv::erode(maskWarped, occlusionMaskEroded, element);
                    //if too much occlusion, discard view
                    {
                        size_t numOccludedForegroundPixels = 0;
                        for (int row = 0; row < resolution.y(); ++row) {
                            for (int col = 0; col < resolution.x(); ++col) {
                                if (maskWarped.at<uchar>(row, col) == 0 && initMask.at<uchar>(row, col) == 0) {
                                    ++numOccludedForegroundPixels;
                                }
                            }
                        }
                        if ((numForegroundPixels - numOccludedForegroundPixels) * 10 < numForegroundPixels) {
                            std::cout << "less than 10% unoccluded, skipping" << std::endl;
                            continue;
                        }
                    }
                    if (settings_.debug_visualization) {
                        cv::imwrite(name.str() + "_warped.png", imgWarped);
                        cv::imwrite(name.str() + "_mask_warped_eroded.png", occlusionMaskEroded * 255);
                        cv::imwrite(name.str() + "_approxbgmask_warped_eroded.png", approxBGMaskWarped * 255);
                    }
                    cv::cvtColor(imgWarped, imgWarped, cv::COLOR_BGR2Luv, 0);
                    allMasks.push_back(std::move(occlusionMaskEroded));
                    approxBGMasks.push_back(std::move(approxBGMaskWarped));
                    allImages.push_back(std::move(imgWarped));
                    homographies.push_back(std::move(H));
                }
            }

            // SEGMENTATION
            {
                int num_pixels = initMask.total();
                for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
                    std::cout << "learning GMMS..." << std::endl;
                    DECLARE_TIMING(learning);
                    START_TIMING(learning);
                    std::vector<std::vector<GaussianMixture>> gmms(allImages.size(), std::vector<GaussianMixture>(2));
                    for (int imageInd = 0; imageInd < allImages.size(); ++imageInd) {
                        for (int cluster = 0; cluster <= 1; ++cluster) {
                            const cv::Mat &img = allImages[imageInd];
                            std::vector<int> indices;
                            indices.reserve(img.total());
                            //add all pixels that are both inside the known subset and unoccluded in this view to the training set
                            for (int p = 0; p < img.total(); ++p) {
                                if (initMask.at<uchar>(p) == cluster && allMasks[imageInd].at<uchar>(p)) {
                                    indices.push_back(p);
                                }
                            }
                            size_t n = indices.size();
                            MatrixX3d colors(n, 3);
                            for (size_t j = 0; j < n; ++j) {
                                int p = indices[j];
                                const auto &col = img.at<cv::Vec3b>(p);
                                colors.row(j) = RowVector3d(col[0], col[1], col[2]);
                            }
                            gmms[imageInd][cluster] = GaussianMixture(settings_.segmentation_gmm_components, 3,
                                                                      settings_.segmentation_gmm_min_variance);
                            int iters = gmms[imageInd][cluster].learn(colors);
                            if (!gmms[imageInd][cluster].success()) {
                                std::cout << "warning: gmm " << imageInd << '-' << cluster << " failed to learn from "
                                          << n
                                          << " points after " << iters << " iterations" << std::endl;
                            }
                        }
                    }
                    STOP_TIMING(learning);
                    PRINT_TIMING(learning);
                    std::cout << "inferring probabilities..." << std::endl;
                    DECLARE_TIMING(inferring);
                    START_TIMING(inferring);
                    std::vector<cv::Mat> avgLogProbImgs(2);
                    avgLogProbImgs[0] = cv::Mat::zeros(initMask.rows, initMask.cols, CV_32FC1);
                    avgLogProbImgs[1] = cv::Mat::zeros(initMask.rows, initMask.cols, CV_32FC1);
                    cv::Mat viewCount = cv::Mat::ones(initMask.rows, initMask.cols, CV_32FC1) * 0.0001;
                    for (int imgInd = 0; imgInd < allImages.size(); ++imgInd) {
                        for (int cluster = 0; cluster <= 1; ++cluster) {
                            if (gmms[imgInd][cluster].success()) {
                                size_t nloop = initMask.rows * initMask.cols;
#pragma omp parallel for default(none) shared(nloop, initMask, allMasks, approxBGMasks, allImages, gmms, cluster, viewCount, avgLogProbImgs, imgInd)
                                for (size_t ind = 0; ind < nloop; ++ind) {
                                    size_t i = ind / initMask.cols;
                                    size_t j = ind % initMask.cols;
                                    if (allMasks[imgInd].at<uchar>(i, j)) {
                                        const auto &col = allImages[imgInd].at<cv::Vec3b>(i, j);
                                        RowVector3d color(col[0], col[1], col[2]);
                                        float energy = -static_cast<float>(gmms[imgInd][cluster].logp_data(color)(0));
                                        double weight = approxBGMasks[imgInd].at<uchar>(i, j) > 0 ? 3 : 1;
                                        avgLogProbImgs[cluster].at<float>(i, j) += weight * energy;
                                        viewCount.at<float>(i, j) += weight;
                                    }
                                }
                            }
                        }
                    }
                    STOP_TIMING(inferring);
                    PRINT_TIMING(inferring);
                    avgLogProbImgs[0] /= viewCount;
                    avgLogProbImgs[1] /= viewCount;

                    if (settings_.debug_visualization) {
                        for (int cluster = 0; cluster <= 1; ++cluster) {
                            cv::Mat debugAvgProb2;
                            double minVal;
                            double maxVal;
                            cv::Point minLoc;
                            cv::Point maxLoc;
                            cv::minMaxLoc(avgLogProbImgs[cluster], &minVal, &maxVal, &minLoc, &maxLoc);

                            avgLogProbImgs[cluster].convertTo(debugAvgProb2, CV_8UC1, 255.0 / maxVal);
                            cv::imwrite("prob_img_" + std::to_string(partIdx) + '-' + std::to_string(cluster) +
                                        "_avg_scale_maxval_" +
                                        std::to_string(maxVal) + "_iteration_" + std::to_string(iteration) + ".png",
                                        debugAvgProb2);
                        }
                    }

                    cv::Mat energy(initMask.rows, initMask.cols, CV_32FC2);
                    for (int row = 0; row < initMask.rows; ++row) {
                        for (int col = 0; col < initMask.cols; ++col) {
                            float energy0 = std::min((float)MAX_ENERGY, avgLogProbImgs[0].at<float>(row, col));
                            float energy1 = std::min((float)MAX_ENERGY, avgLogProbImgs[1].at<float>(row, col));
                            energy.at<cv::Vec<float, 2>>(row, col) = cv::Vec<float, 2>(energy0, energy1);
                        }
                    }

                    std::cout << "applying constraints..." << std::endl;
                    std::vector<Edge2d> guides;
                    if (useConstraints) {
                        for (const auto &guide : guideConstraints) {
                            std::cout << "adding " << guide.second.size() << " guides from part " << guide.first << std::endl;
                            for (const auto &guideEdge : guide.second) {
                                Vector2d pixelStart = (guideEdge.edge.first - minPt) * worldToImgScale;
                                Vector2d pixelEnd = (guideEdge.edge.second - minPt) * worldToImgScale;
                                guides.emplace_back(pixelStart, pixelEnd);
                            }
                        }
                        for (const auto &sc : shapeConstraints) {
                            std::cout << "adding " << sc.second.size() << " shape constraints from part " << sc.first
                                      << std::endl;
                            for (const auto &constraint : sc.second) {
                                Vector2d pixelStart = (constraint.edge.first - minPt) * worldToImgScale;
                                Vector2d pixelEnd = (constraint.edge.second - minPt) * worldToImgScale;
                                cv::Point2d ptA(pixelStart.x(), pixelStart.y());
                                cv::Point2d ptB(pixelEnd.x(), pixelEnd.y());
                                paintConstraint(energy, cv::Vec<float, 2>(0, maxEnergy / 2),
                                                cv::Vec<float, 2>(maxEnergy / 2, 0), ptA, ptB, 1,
                                                constraint.inside ? 1 : 0,
                                                constraint.outside ? margin * worldToImgScale : 0);
                            }
                        }
                        //0 = no constraint, 1 = inside, 2 = outside
//                    cv::Mat constraintLabels = cv::Mat::zeros(imageSize, CV_8UC1);
                        for (const auto &pair : connectionConstraints) {
                            std::cout << "adding " << pair.second.size() << " connection constraints from part "
                                      << pair.first << std::endl;
                            for (const auto &constraint : pair.second) {
                                double pixelMargin = std::max(1.0, constraint.margin * worldToImgScale);
                                Vector2d pixelStart = (constraint.edge.first - minPt) * worldToImgScale;
                                Vector2d pixelEnd = (constraint.edge.second - minPt) * worldToImgScale;

                                cv::Point2d ptA(pixelStart.x(), pixelStart.y());
                                cv::Point2d ptB(pixelEnd.x(), pixelEnd.y());
                                double outFatness = construction.getStock(pair.first).thickness * worldToImgScale * 2;
//                                double inFatness = construction.getStock(pair.first).thickness * worldToImgScale;
                                std::cout << "outFatness: " << outFatness << std::endl;
                                paintConstraint(energy, cv::Vec<float, 2>(0, maxEnergy / 2),
                                                cv::Vec<float, 2>(maxEnergy / 2, 0), ptA, ptB, pixelMargin,
                                                constraint.inside ? outFatness : 0,
                                                constraint.outside ? outFatness : 0);
                            }
                        }
                    }
                    for (int row = 0; row < initMask.rows; ++row) {
                        for (int col = 0; col < initMask.cols; ++col) {
                            if (row == 0 || col == 0 || row == initMask.rows - 1 || col == initMask.cols - 1 ||
                                knownBGMask.at<uchar>(row, col) == 1) {
                                energy.at<cv::Vec<float, 2>>(row, col) = cv::Vec<float, 2>(maxEnergy, 0);
                            } else if (knownFGMask.at<uchar>(row, col) == 1) {
                                energy.at<cv::Vec<float, 2>>(row, col) = cv::Vec<float, 2>(0, maxEnergy);
                            }
                        }
                    }
                    if (settings_.debug_visualization) {
                        displayProbabilityMap(energy, "part_" + std::to_string(partIdx) + "_energy_");
                    }
                    cv::Mat labels(imageSize, CV_8UC1);
                    std::cout << "running maxflow..." << std::endl;
                    DECLARE_TIMING(flow);
                    START_TIMING(flow);
                    float maxflow = graph_cut(energy, labels, settings_.segmentation_smoothing_weight, allImages,
                                              settings_.segmentation_sigma, allMasks, guides, 0, settings_.segmentation_8way, settings_.segmentation_precision);
                    STOP_TIMING(flow);
                    PRINT_TIMING(flow);
                    std::cout << maxflow << " max flow" << std::endl;
                    if (settings_.debug_visualization) {
                        cv::Mat labelsCopy(labels.rows, labels.cols, CV_8UC3);
                        for (int i = 0; i < labels.rows; ++i) {
                            for (int j = 0; j < labels.cols; ++j) {
                                labelsCopy.at<cv::Vec3b>(i, j) = labels.at<uchar>(i, j) ? cv::Vec3b(255, 255, 255)
                                                                                        : cv::Vec3b(0, 0, 0);
                            }
                        }
                        if (useConstraints) {
                            for (const auto &pair : guideConstraints) {
                                for (const auto &guideEdge : pair.second) {
                                    Vector2d pixelStart = (guideEdge.edge.first - minPt) * worldToImgScale;
                                    Vector2d pixelEnd = (guideEdge.edge.second - minPt) * worldToImgScale;
                                    Vector2d ptA = pixelStart;
                                    Vector2d ptB = pixelEnd;
                                    cv::line(labelsCopy, cv::Point(ptA.x(), ptA.y()), cv::Point(ptB.x(), ptB.y()),
                                             cv::Vec3b(255, 0, 255));
                                }
                            }
                            for (const auto &pair : connectionConstraints) {
                                for (const auto &constraint : pair.second) {
                                    Vector2d pixelStart = (constraint.edge.first - minPt) * worldToImgScale;
                                    Vector2d pixelEnd = (constraint.edge.second - minPt) * worldToImgScale;
                                    Vector2d dir = (constraint.edge.second - constraint.edge.first).normalized();
                                    double pixelMargin = std::max(1.0, constraint.margin * worldToImgScale);
//                                std::cout << "pixel margin for constraint " << pair.first << ": " << pixelMargin << std::endl;
                                    Vector2d n(dir.y(), -dir.x());
                                    n *= pixelMargin;
//
                                    Vector2d ptA = pixelStart + n;
                                    Vector2d ptB = pixelEnd + n;
                                    if (constraint.outside) {
                                        cv::line(labelsCopy, cv::Point(ptA.x(), ptA.y()), cv::Point(ptB.x(), ptB.y()),
                                                 cv::Vec3b(0, 255, 255));
                                    }
                                    if (constraint.inside) {
                                        ptA = pixelStart - n;
                                        ptB = pixelEnd - n;
                                        cv::line(labelsCopy, cv::Point(ptA.x(), ptA.y()), cv::Point(ptB.x(), ptB.y()),
                                                 cv::Vec3b(255, 255, 0));
                                    }
                                }
                            }
                            for (const auto &sc : shapeConstraints) {
                                for (const auto &constraint : sc.second) {
                                    Vector2d pixelStart = (constraint.edge.first - minPt) * worldToImgScale;
                                    Vector2d pixelEnd = (constraint.edge.second - minPt) * worldToImgScale;
                                    Vector2d dir = (constraint.edge.second - constraint.edge.first).normalized();
                                    Vector2d n(dir.y(), -dir.x());

                                    Vector2d ptA = pixelStart + n;
                                    Vector2d ptB = pixelEnd + n;
                                    if (constraint.outside) {
                                        cv::line(labelsCopy, cv::Point(ptA.x(), ptA.y()), cv::Point(ptB.x(), ptB.y()),
                                                 cv::Vec3b(0, 0, 255));
                                    }
                                    if (constraint.inside) {
                                        ptA = pixelStart - n;
                                        ptB = pixelEnd - n;
                                        cv::line(labelsCopy, cv::Point(ptA.x(), ptA.y()), cv::Point(ptB.x(), ptB.y()),
                                                 cv::Vec3b(255, 0, 0));
                                    }
                                }
                            }
                        }
                        cv::imwrite("part_" + std::to_string(partIdx) + "_graphCutOutput_iteration_" +
                                    std::to_string(iteration) + "_flow_" + std::to_string(maxflow) + ".png", labelsCopy * 255);
                    }
                    initMask = labels;
                }

                initMask *= 255;
//                cv::resize(initMask, initMask, cv::Size(), CONTOUR_SCALE, CONTOUR_SCALE);
                std::vector<double> gridData(initMask.total());
                for (int pix = 0; pix < initMask.total(); ++pix) {
                    gridData[pix] = (255 - initMask.at<uchar>(pix)) / 255.0;
                }
                double spacing = planeDims.x() / initMask.cols;
                VoxelGrid2D grid(std::move(gridData), initMask.cols, initMask.rows, minPt.x(), minPt.y(), spacing);
                std::vector<std::vector<int>> hierarchy;
                std::vector<std::vector<Vector2d>> contours = grid.marching_squares(hierarchy, 0.5);
                std::cout << "found " << contours.size() << " contours in segmentation for part " << partIdx
                          << std::endl;
                if (!hierarchy.back().empty()) {
//                int max_outer_contour_index = *std::max_element(hierarchy.back().begin(), hierarchy.back().end(),
//                                                                [&](int a, int b) {
//                                                                    return contours[a].size() <
//                                                                           contours[b].size();
//                                                                });
//                const std::vector<Vector2d> &max_outer_contour = contours[max_outer_contour_index];
                    for (auto outer_contour_index : hierarchy.back()) {
                        std::vector<const std::vector<Vector2d> *> allContours;
                        const std::vector<Vector2d> &outer_contour = contours[outer_contour_index];
                        allContours.push_back(&outer_contour);
                        MatrixX2d contour_eig = vstack(outer_contour);
                        if (settings_.debug_visualization) {
                            std::ofstream file("part_" + std::to_string(partIdx) + "_segmentation_contour.csv");
                            file << contour_eig.format(CSVFormat);
                        }
                        std::vector<std::shared_ptr<Primitive>> holes;
                        for (int child_ind : hierarchy[outer_contour_index]) {
                            const std::vector<Vector2d> &child_contour = contours[child_ind];
                            allContours.push_back(&child_contour);
                            if (child_contour.size() * settings_.max_contour_hole_ratio > outer_contour.size()) {
                                MatrixX2d hole_eig = vstack(child_contour);
                                if (settings_.debug_visualization) {
                                    std::ofstream file(
                                            "part_" + std::to_string(partIdx) + "_hole_" + std::to_string(child_ind) +
                                            "_segmentation_contour_" + std::to_string(outer_contour_index) + ".csv");
                                    file << hole_eig.format(CSVFormat);
                                }
                                holes.emplace_back(new Polygon(hole_eig));
                            }
                        }
                        std::shared_ptr<Polygon> poly;
                        if (holes.empty()) {
                            poly = std::make_shared<Polygon>(std::move(contour_eig));
                        } else {
                            poly = std::make_shared<PolygonWithHoles>(
                                    std::move(contour_eig), std::move(holes));
                        }
                        bool contained = false;
                        for (const auto &pt : outer_contour) {
                            if (shape.cutPath->contains(pt)) {
                                contained = true;
                                break;
                            }
                        }
                        if (!contained) {
                            auto points = shape.cutPath->points();
                            for (int r=0; r<points.rows(); ++r) {
                                if (poly->contains(points.row(r).transpose())) {
                                    contained = true;
                                    break;
                                }
                            }
                        }
                        if (contained && poly->area() > thickness * thickness) {
                            newShapes[v].push_back(poly);
                        }
                    }
                    shape.gridSpacing = spacing;
//                SegmentationResult segmentationResult;
//                segmentationResult.spacing = spacing;
//                segmentationResult.width = initMask.cols;
//                segmentationResult.height = initMask.rows;
//                segmentationResult.minPt = minPt;
//                segmentationResult.grid.resize(initMask.total());
//                for (size_t pix=0; pix<initMask.total(); ++pix) {
//                    segmentationResult.grid[pix] = initMask.at<uchar>(pix) < 127;
//                }
//                segmentationResults[partIdx] = std::move(segmentationResult);
                }
            }
            shape.dirty = false;
        }
    }

    std::cout << "updating model with new shapes" << std::endl;
    int numSplit = 0;
    for (const auto &pair : newShapes) {
        Construction::Vertex oldV = pair.first;
        size_t partIdx = construction.g[oldV].partIdx;
        auto &shape = construction.getShape(partIdx);
        if (!pair.second.empty()) {
            shape.cutPath = pair.second[0];

            if (pair.second.size() > 1) {
                ++numSplit;
                std::cout << "splitting " << partIdx << " into " << pair.second.size() << " parts" << std::endl;
                for (size_t i = 1; i < pair.second.size(); ++i) {
                    PartData newPart = construction.partData[partIdx];
                    newPart.shapeIdx = construction.shapeData.size();
                    ShapeData newShape;
                    newShape.stockIdx = shape.stockIdx;
                    newShape.gridSpacing = shape.gridSpacing;
                    newShape.cutPath = pair.second[i];
                    newShape.shapeConstraints = shape.shapeConstraints;
                    newShape.dirty = false;

                    Construction::Vertex v = add_vertex(construction.g);
                    construction.g[v].partIdx = construction.partData.size();
                    construction.partData.push_back(std::move(newPart));
                    construction.w_.push_back(true);
                    construction.shapeData.push_back(std::move(newShape));

                    Construction::out_edge_iter out_i, out_end;
//                std::cout << "copying connections from part " << partIdx << " to " << construction.g[v].partIdx << std::endl;
                    for (tie(out_i, out_end) = out_edges(oldV, construction.g); out_i != out_end; ++out_i) {
                        Construction::Edge e = *out_i;
                        Construction::Vertex adj =
                                target(e, construction.g) == oldV ? source(e, construction.g) : target(e,
                                                                                                       construction.g);
                        auto ebool = add_edge(v, adj, construction.g);
//                    std::cout << "copying connection to " << construction.g[adj].partIdx << std::endl;
                        if (ebool.second) {
                            construction.g[ebool.first] = construction.g[e];
                            if (construction.g[ebool.first].part1 == get(vertex_index, construction.g, oldV))
                                construction.g[ebool.first].part1 = get(vertex_index, construction.g, v);
                            if (construction.g[ebool.first].part2 == get(vertex_index, construction.g, oldV))
                                construction.g[ebool.first].part2 = get(vertex_index, construction.g, v);
                        } else {
                            std::cout << "adding edge " << v << ", " << adj << " failed!" << std::endl;
                        }
                    }
                }
            }
        } else {
            std::cout << "Warning: empty segmentation mask found! for part " << partIdx << '!' << std::endl;
        }
    }
    return numSplit;
}

double Solver::getDiameter() const {
    return diameter_;
}
