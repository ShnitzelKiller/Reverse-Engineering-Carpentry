//
// Created by James Noeckel on 10/12/20.
//

#include "exposuresolver.h"
#include "ceres/ceres.h"

using namespace std;
using namespace ceres;

struct OverexposedFunctor {
    explicit OverexposedFunctor(float lambda) : l(lambda) {}
    template <typename T>
    bool operator()(
            const T* const camera,
            const T* const point,
            T* residual) const
    {
        residual[0] = camera[0]*point[0] - T(1.f);
        if (camera[0]*point[0] > T(1.f)) residual[0] *= T(l);
        return true;
    }

    float l;
};

struct CostFunctor {
    explicit CostFunctor(float pixelvalue) : v(pixelvalue) {}
    template <typename T>
    bool operator()(
            const T* const camera,
            const T* const point,
            T* residual) const
    {
        residual[0] = camera[0]*point[0] - T(v);
        return true;
    }
    float v;
};

const float lambda = 0.f;

CostFunction* Create(float pixelvalue) {
    if (pixelvalue >= 1.f) {
        return (new ceres::AutoDiffCostFunction<OverexposedFunctor, 1, 1, 1>(
                new OverexposedFunctor(lambda)));
    } else {
        return (new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1>(
                new CostFunctor(pixelvalue)));
    }
}


double solveExposure(
        const std::vector<cv::Mat> &images,
        const std::vector<cv::Mat> &occlusionMasks,
        const std::vector<int> &imageIndices,
        const cv::Mat &constraintsMask,
        double* exposures,
        double* radiances,
        int lossfn,
        float scale)
{
    int n = images.size();
    double cost = 0;
    for (int ch = 0; ch < 3; ch++) {
        double* exps = exposures + ch*n;
        double *rads = radiances + ch*n;
        if (ch) {
            memcpy(exps,
                   exposures + (ch-1)*n,
                   sizeof(double)*n);
            memcpy(rads,
                   radiances + (ch-1)*n,
                   sizeof(double)*n);
        }

        ceres::Problem problem;
        int bestIm = imageIndices[0];
        int maxViewCount = 0;
        for (int im : imageIndices) {
            bool seen = false;
            size_t viewCount = 0;
            for (int i=0; i<images[im].total(); ++i) {
                if (occlusionMasks[im].at<uchar>(i) && constraintsMask.at<uchar>(i)) {
                    ceres::CostFunction *costfn = Create(static_cast<float>(images[im].at<cv::Vec3b>(i)[ch])/255.0f);
                    if (lossfn == LOSS_CAUCHY) {
                        problem.AddResidualBlock(
                                costfn, new ceres::CauchyLoss(scale), &exps[im], &rads[i]);
                    } else if (lossfn == LOSS_HUBER) {
                        problem.AddResidualBlock(
                                costfn, new ceres::HuberLoss(scale), &exps[im], &rads[i]);
                    } else {
                        problem.AddResidualBlock(
                                costfn, NULL, &exps[im], &rads[i]);
                    }
                    problem.SetParameterLowerBound(&rads[i], 0, 0);
                    seen = true;
                    ++viewCount;
                }
            }
            if (seen) {
                problem.SetParameterLowerBound(&exps[im], 0, 0);
            }
            if (viewCount > maxViewCount) {
                maxViewCount = viewCount;
                bestIm = im;
            }
        }
        problem.SetParameterBlockConstant(&exps[bestIm]);

        ceres::Solver::Options options;
        //options.linear_solver_type = ceres::DENSE_SCHUR;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.use_inner_iterations = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.FullReport() << "\n";
        cost = max(cost, summary.final_cost);
    }
    return cost;
}