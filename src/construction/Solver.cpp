//
// Created by James Noeckel on 7/8/20.
//
//#define MULTI
//#define TESTGRAD
//#define DEBUG_VIS

#include "Solver.h"
#include "optimization/discreteProblems.hpp"
#include "optimization/simulated_annealing.hpp"
#include "optimization/alignmentProblem.hpp"
#include "geometry/shapes2/primitive_thickness.h"
#include "math/integration.h"
#include "math/RunningAverage.h"
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/algorithm.hpp>
#include <utility>
//#include <igl/opengl/glfw/Viewer.h>
//#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
//#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <sstream>
#include "utils/printvec.h"
#include "utils/timingMacros.h"
#include "utils/vstack.h"
#include "test/testUtils/curveFitUtils.h"
#include "geometry/shapes2/density_contour.h"
#include "geometry/shapes2/convert_contour.h"
//#include <pagmo/algorithms/ipopt.hpp>
#ifdef MULTI
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/utils/multi_objective.hpp>
#endif
#ifdef TESTGRAD
#include <construction/pagmo/test_problem.h>
#endif

#define ANGLE_THRESHOLD (10/180.0*M_PI)
#define DEFAULT_CONNECTION_SCORE 10
#define MIN_VISIBLE_LENGTH 5
#define MAX_PARALLEL_ANGlE 0.03

using namespace pagmo;
using namespace Eigen;
using namespace boost;

void Solver::recomputeMeshes(bool recomputeBVH) {
    construction.computeMeshes(recomputeBVH);
}

std::vector<std::pair<int32_t, double>> Solver::getVisibility(const Eigen::Vector3d &origin, const Eigen::Vector3d &normal) {
    std::vector<int32_t> views;
    std::vector<double> dotprods;
    for (const auto &pair : reconstruction_->images) {
        double dotprod = 1.0;
        Vector3d camera_pos = -(pair.second.rot_.conjugate() * pair.second.trans_);
        Vector3d camera_offset = (camera_pos - origin).normalized();
        dotprod = camera_offset.dot(normal);

        if (dotprod > 0) {
            views.push_back(pair.first);
            dotprods.push_back(dotprod);
//                visible_clusters[pair.first].push_back(c);
        }
    }
    std::vector<int> view_indices(views.size());
    std::iota(view_indices.begin(), view_indices.end(), 0);
    std::sort(view_indices.begin(), view_indices.end(), [&](int a, int b) {return dotprods[a] > dotprods[b];});
    std::vector<std::pair<int32_t, double>> sorted_views;
    sorted_views.reserve(views.size());
    for (int view_indice : view_indices) {
        sorted_views.emplace_back(views[view_indice], dotprods[view_indice]);
    }
    return sorted_views;
}

void Solver::computeVisibility() {
    size_t numParts = construction.partData.size();
    cluster_visibility.clear();
    cluster_visibility.reserve(numParts * 2);
//    visible_clusters.reserve(reconstruction_->images.size());
    //for each bbox, for each point in the bbox, increment scores for each view
    for (size_t c=0; c<numParts * 2; ++c) {
        std::vector<int32_t> views;
        std::vector<double> dotprods;
        int realC = c < numParts ? c : c-numParts;
        Vector3d plane_center = construction.partData[realC].unproject(construction.getShape(realC).cutPath->points().colwise().mean()).transpose();
        Vector3d n = construction.partData[realC].normal();
        if (c >= numParts) {
            //backside
            plane_center -= n * construction.getStock(realC).thickness;
            n = -n;
        }

        auto sorted_views = getVisibility(plane_center, n);
        cluster_visibility.push_back(std::move(sorted_views));
    }
//    for (auto &pair : visible_clusters) {
//        std::sort(pair.second.begin(), pair.second.end());
//    }
}

void mapAdd(std::map<std::vector<bool>, int> &inout, std::map<std::vector<bool>, int> &in) {
    for (const auto &pair : in) {
        auto it = inout.find(pair.first);
        if (it != inout.end()) {
            it->second += pair.second;
        } else {
            inout[pair.first] = pair.second;
        }
    }
}

void Solver::pruneOpposing() {
    double voxel_width = diameter_/settings_.voxel_resolution;
    /** ID of part each part was merged into (-1 if it is left intact) */
    std::vector<int> mergedIds(construction.partData.size(), -1);
    std::vector<bool> opposite(construction.partData.size());
    std::vector<bool> visited(construction.partData.size(), false);
    for (size_t partIdx=0; partIdx < construction.partData.size(); ++partIdx) {
        const auto &pd = construction.partData[partIdx];
        if (!visited[partIdx]) {
            // (idx, opposing)
            std::cout << "finding parts to merge with " << partIdx << std::endl;
            std::vector<size_t> overlappingIds;
            {
                std::vector<std::pair<size_t, bool>> stack(1, {partIdx, false});
                while (!stack.empty()) {
                    auto nextPart = stack.back();
                    stack.pop_back();
                    if (!visited[nextPart.first]) {
                        std::cout << "expanding " << nextPart.first << " (" << (nextPart.second ? "opposite" : "parallel") << ")" << std::endl;
                        if (nextPart.first != partIdx) {
                            mergedIds[nextPart.first] = partIdx;
                        }
                        visited[nextPart.first] = true;
                        opposite[nextPart.first] = nextPart.second;
                        overlappingIds.push_back(nextPart.first);
                        std::cout << "adding neighbors from part " << nextPart.first << ": ";
                        for (auto idx : construction.partData[nextPart.first].opposingPartIds) {
                            std::cout << idx << ", ";
                            stack.emplace_back(idx, !nextPart.second);
                        }
                        std::cout << std::endl;
                    }
                }
            }
            assert(overlappingIds.size() >= 1);
            if (overlappingIds.size() > 1) {
                std::cout << "merging " << overlappingIds.size() << " shapes into part " << partIdx << ": ";
                size_t totalPts = 0;
                auto &shapeConstraints = construction.getShape(partIdx).shapeConstraints;
                for (auto idx : overlappingIds) {
                    std::cout << idx << ", ";
                    totalPts += construction.partData[idx].pointIndices.size();
                    //inherit any shape constraints
                    auto &otherShapeConstraints = construction.getShape(idx).shapeConstraints;
                    for (const auto &pair : otherShapeConstraints) {
                        const auto &otherPD = construction.partData[idx];
                        auto it = shapeConstraints.find(pair.first);
                        if (it == shapeConstraints.end()) {
                            auto &constraintList = shapeConstraints[pair.first];
                            for (const auto &cc : pair.second) {
                                Edge3d edge3d(otherPD.unproject(
                                        cc.edge.first.transpose()).transpose(),
                                              otherPD.unproject(
                                                      cc.edge.second.transpose()).transpose());
                                ShapeConstraint ccNew(cc);
                                ccNew.edge = Edge2d(
                                        pd.project(edge3d.first.transpose()).transpose(),
                                        pd.project(edge3d.second.transpose()).transpose());
                                if (opposite[idx]) {
                                    std::swap(ccNew.edge.first, ccNew.edge.second);
                                    ccNew.opposing = true;
                                }
                                constraintList.push_back(ccNew);
                            }
                        }
                    }
                }
                std::cout << std::endl;
                PointCloud2::Handle cloud2d(new PointCloud2);
                {
                    std::vector<Vector2d> points2d;
                    points2d.reserve(totalPts);
                    for (auto idx : overlappingIds) {
                        for (auto p : construction.partData[idx].pointIndices) {
                            points2d.emplace_back(pd.project(cloud_->P.row(p)).transpose());
                        }
                    }
                    cloud2d->P = vstack(points2d);
                }
                std::vector<std::vector<int>> hierarchy;
                std::vector<std::vector<Vector2d>> contours = density_contour(cloud2d, hierarchy, settings_.contour_threshold, voxel_width);
                std::shared_ptr<Primitive> shape = convertContour(contours, hierarchy, settings_.max_contour_hole_ratio);
                construction.getShape(partIdx).cutPath = std::move(shape);
            }
        }
    }

    //prune part data
    std::vector<int> oldToNewIndex(construction.partData.size());
    int newSize=0;
    for (size_t partIdx=0; partIdx<construction.partData.size(); ++partIdx) {
        if (mergedIds[partIdx] < 0 && construction.getStock(partIdx).thickness > voxel_width) {
            oldToNewIndex[partIdx] = newSize;
            if (newSize != partIdx) {
                construction.partData[newSize] = std::move(construction.partData[partIdx]);
            }
            ++newSize;
        } else {
            if (construction.getStock(partIdx).thickness <= voxel_width) {
                std::cout << "part too thin: " << construction.getStock(partIdx).thickness << " vs " << voxel_width << std::endl;
            }
            oldToNewIndex[partIdx] = -1;
        }
    }
    construction.partData.resize(newSize);

    std::cout << "oldToNewIndex: " << oldToNewIndex << std::endl;
    std::cout << "mergeIndices: " << mergedIds << std::endl;
    std::cout << "opposite: " << opposite << std::endl;

    std::cout << "reassigning shape constraints" << std::endl;
    for (size_t partIdx=0; partIdx<construction.partData.size(); ++partIdx) {
        std::cout << "part " << partIdx << std::endl;
        std::unordered_map<int, std::vector<ShapeConstraint>> newShapeConstraints;
        auto &shapeConstraints = construction.getShape(partIdx).shapeConstraints;
        for (auto &sc : shapeConstraints) {
            auto newIndex = oldToNewIndex[sc.first];
            std::cout << "mapping " << sc.first << " to " << newIndex << std::endl;
            if (newIndex >= 0) {
                std::cout << "adding " << sc.first << "'s shape constraints to " << newIndex << std::endl;
                std::copy(sc.second.begin(), sc.second.end(), std::back_inserter(newShapeConstraints[newIndex]));
            } else if (mergedIds[sc.first] >= 0) {
                auto baseNew = oldToNewIndex[mergedIds[sc.first]];
                if (baseNew >= 0) {
                    std::cout << "adding " << sc.first << "'s shape constraints to opposing " << baseNew << std::endl;
                    std::for_each(sc.second.begin(), sc.second.end(), [&](ShapeConstraint &constraint) {constraint.otherOpposing = opposite[sc.first];});
                    std::copy(sc.second.begin(), sc.second.end(), std::back_inserter(newShapeConstraints[baseNew]));
                }
            } else {
                std::cout << "adding " << sc.first << "'s shape constraints to miscellaneous index" << std::endl;
                std::copy(sc.second.begin(), sc.second.end(), std::back_inserter(newShapeConstraints[-1]));
            }
        }
        shapeConstraints = std::move(newShapeConstraints);
    }
    construction.computeMeshes();
}

#pragma omp declare reduction(mapAdd: std::map<std::vector<bool>, int>: \
    mapAdd(omp_out, omp_in))

bool Solver::optimizeW(int alg) {
    //find overlap volumes for constraints
    std::cout << "precomputing overlap volumes by sampling" << std::endl;
    std::map<std::vector<bool>, int> canonicalCounts;
    size_t nParts = construction.partData.size();
    RowVector3d bboxMin = cloud_->P.colwise().minCoeff();
    RowVector3d bboxMax = cloud_->P.colwise().maxCoeff();
    RowVector3d bbox = bboxMax-bboxMin;
    std::uniform_real_distribution<double> x_sampler(bboxMin.x(), bboxMax.x());
    std::uniform_real_distribution<double> y_sampler(bboxMin.y(), bboxMax.y());
    std::uniform_real_distribution<double> z_sampler(bboxMin.z(), bboxMax.z());
    {
        DECLARE_TIMING(sampling);
        START_TIMING(sampling);
//        auto t_start = clock();
//        auto t_start_w = std::chrono::high_resolution_clock::now();
        //const size_t nthreads = std::thread::hardware_concurrency();
        int nthreads = omp_get_max_threads();
        //std::vector<std::map<std::vector<bool>, int>> canonicalCountsBlock(nthreads);
        std::vector<int> seeds(nthreads);
        {
            std::uniform_int_distribution<int> seedSampler{std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max()};
            for (int t = 0; t < nthreads; ++t) {
                seeds[t] = seedSampler(random_);
            }
        }
#pragma omp parallel default(none) firstprivate(x_sampler, y_sampler, z_sampler) shared(seeds, nParts) reduction(mapAdd:canonicalCounts)
        {
            int threadId = omp_get_thread_num();
            std::mt19937 localrandom(seeds[threadId]);
#pragma omp for
            for (size_t r = 0; r < settings_.sample_count; ++r) {
                std::vector<bool> key(nParts);
                RowVector3d pt;
                pt.x() = x_sampler(localrandom);
                pt.y() = y_sampler(localrandom);
                pt.z() = z_sampler(localrandom);
                for (size_t idx = 0; idx < nParts; ++idx) {
                    key[idx] = construction.contains(idx, pt);
                }
                auto it = canonicalCounts.find(key);
                if (it == canonicalCounts.end()) {
                    canonicalCounts[key] = 1;
                } else {
                    ++it->second;
                }
            }
        }
        STOP_TIMING(sampling);
        PRINT_TIMING(sampling);
//        std::cout << "sampling finished in " << time_sec << " CPU seconds (" << std::chrono::duration<double>(total_t_w).count() << "s wall clock time" << std::endl;
    }
    std::cout << "number of canonical intersection terms: " << canonicalCounts.size() << std::endl;
    for (const auto &pair : canonicalCounts) {
        std::cout << '[';
        for (const auto b : pair.first) {
            std::cout << b << ", ";
        }
        std::cout << "]: " << pair.second << std::endl;
    }
    /** overlaps(i, j) = vol(i\cap j) / vol(i) */
    MatrixXd overlaps(nParts, nParts);
    {
        //upper triangular matrix; overlapsi(i, j) = overlap of i and j where i < j
        //overlapsi(i, i) = volume of part i
        MatrixXi overlapsi = MatrixXi::Zero(nParts, nParts);
        for (const auto &pair : canonicalCounts) {
            for (int i = 0; i < nParts; ++i) {
                if (pair.first[i]) {
                    overlapsi(i, i) += pair.second;
                }
                for (int j = i + 1; j < nParts; ++j) {
                    if (pair.first[i] && pair.first[j]) {
                        overlapsi(i, j) += pair.second;
                    }
                }
            }
        }
        for (int i=0; i<nParts; ++i) {
            for (int j=i+1; j<nParts; ++j) {
                auto overlap = static_cast<double>(overlapsi(i, j));
                overlaps(i, j) = overlapsi(i, i) == 0 ? 0 : overlap / overlapsi(i, i);
                overlaps(j, i) = overlapsi(j, j) == 0 ? 0 : overlap / overlapsi(j, j);
            }
            overlaps(i, i) = overlapsi(i, i);
        }
    }
    std::cout << "overlap matrix: " << std::endl << overlaps << std::endl;
    for (int row=0; row<overlaps.rows(); ++row) {
        for (int col=0; col<overlaps.cols(); ++col) {
            if (std::isnan(overlaps(row, col))) {
                std::cout << "WARNING: NaN at " << row << ", " << col << std::endl;
                return false;
            }
        }
    }
    std::cout << "precomputing knowns" << std::endl;
    //find non-overlapping candidates and hold them fixed
    // 0 = not known, 1 = true, 2 = false
    std::vector<int> known(nParts, 0);
    for (int c=0; c<nParts; ++c) {
        double overlap = 0;
        for (int c2=0; c2<nParts; ++c2) {
            if (c == c2) continue;
            overlap += overlaps(c, c2);
        }
        if (overlap < settings_.max_overlap_ratio) {
            std::cout << "fixing part " << c << " true with " << overlap * 100 << " percent overlap" << std::endl;
            known[c] = 1;
        }
    }
    for (int c=0; c<nParts; ++c) {
        if (known[c]) continue;
        double overlapWithKnown = 0.0;
        for (int c2=0; c2<nParts; ++c2) {
            if (known[c2]) {
                overlapWithKnown += overlaps(c, c2);
            }
        }
        if (overlapWithKnown > settings_.max_overlap_ratio) {
            std::cout << "fixing part " << c << " false with " << overlapWithKnown << " percent overlap with known parts" << std::endl;
            known[c] = 2;
        }
    }
    int numFree = 0;
    for (int c=0; c<nParts; ++c) {
        if (known[c] == 0) ++numFree;
    }
    std::vector<int> indexMap(numFree);
    {
        int currIndex = 0;
        for (int c = 0; c < nParts; ++c) {
            if (known[c] == 0) {
                indexMap[currIndex++] = c;
            }
        }
    }
    std::cout << "index map (" << indexMap.size() << "): ";
    std::cout << indexMap << std::endl;

    //populate solution using both knowns and results of optimization
    std::vector<bool> w(nParts, false);
    for (int c=0; c<nParts; ++c) {
        if (known[c] == 1) {
            w[c] = true;
        } else if (known[c] == 2) {
            w[c] = false;
        }
    }

    if (!indexMap.empty()) {
        MatrixXd reducedOverlaps(numFree, numFree);
        for (int row = 0; row < numFree; ++row) {
            for (int col = 0; col < numFree; ++col) {
                reducedOverlaps(row, col) = overlaps(indexMap[row], indexMap[col]);
            }
        }
        MatrixXd allDistances;
        {
            std::cout << "precomputing part point distances... ";
            DECLARE_TIMING(distances);
            START_TIMING(distances);
            allDistances = construction.allDistances(cloud_, settings_.alignment_stride);
            STOP_TIMING(distances);
            PRINT_TIMING(distances);
            allDistances /= diameter_; //to ensure that energy function is scale-independent
        }
        if (alg == 1) {
            std::cout << "running naiive selection" << std::endl;
            std::vector<double> areas(indexMap.size());
            for (int i=0; i<indexMap.size(); ++i) {
                areas[i] = construction.getShape(indexMap[i]).cutPath->area();
            }
            std::vector<int> index_indices(indexMap.size());
            std::iota(index_indices.begin(), index_indices.end(), 0);
            std::sort(index_indices.begin(), index_indices.end(), [&](int a, int b) {return areas[a] > areas[b];});
            for (int ind=0; ind<index_indices.size(); ++ind) {
                double totalOverlap = 0.0;
                int i = indexMap[index_indices[ind]];
                for (int j = 0; j < w.size(); ++j) {
                    if (i == j) continue;
                    if (w[j]) {
                        totalOverlap += overlaps(i, j);
                    }
                }
                if (totalOverlap <= settings_.max_overlap_ratio) {
                    w[i] = true;
                }
            }
        } else if (alg == 0) {
            problem p{CarpentryOptimizationSOProblem(overlaps, allDistances, settings_.max_overlap_ratio, indexMap,
                                                     known)};
            algorithm algo{
                    HypercubeSimulatedAnnealing(settings_.generations, &reducedOverlaps, settings_.max_overlap_ratio)};
            // 3 - Instantiate a population
            population pop(p);
            pop.push_back(std::vector<double>(numFree, 0.0));
            // 4 - Evolve the population
//    std::cout << pop << std::endl;
            DECLARE_TIMING(mcmc);
            START_TIMING(mcmc);
            pop = algo.evolve(pop);
            STOP_TIMING(mcmc);
            PRINT_TIMING(mcmc);
            const auto &vec = pop.get_x()[pop.best_idx()];
            const auto &f = pop.get_f()[pop.best_idx()];
            std::cout << "]; f[0]=" << f[0] << "; f[1]=" << f[1] << std::endl;
            for (size_t i = 0; i < vec.size(); ++i) {
                w[indexMap[i]] = vec[i] > 0.5;
            }
        } else if (alg == 2) {
            std::cout << "running greedy distance-based selection" << std::endl;

            std::vector<bool> visited(indexMap.size(), false);

            for (int iter=0; iter<indexMap.size(); ++iter) {

                int nextInd = -1;
                double minDist = std::numeric_limits<double>::max();

                for (int ind = 0; ind < indexMap.size(); ++ind) {
                    int i = indexMap[ind];
                    if (!w[i] && !visited[ind]) {
                        w[i] = true;

                        VectorXd minDistances = VectorXd::Constant(allDistances.rows(),
                                                                   std::numeric_limits<double>::max());
                        for (size_t k = 0; k < w.size(); ++k) {
                            if (w[k]) {
                                minDistances = minDistances.cwiseMin(allDistances.col(k));
                            }
                        }
                        double avgPointDistance = minDistances.mean();
                        if (avgPointDistance < minDist) {
                            minDist = avgPointDistance;
                            nextInd = ind;
                        }
                        w[i] = false;
                    }
                }

                if (nextInd < 0) break;
                visited[nextInd] = true;
                double totalOverlap = 0.0;
                int i = indexMap[nextInd];
                for (int j = 0; j < w.size(); ++j) {
                    if (i == j) continue;
                    if (w[j]) {
                        totalOverlap += overlaps(i, j);
                    }
                }
                if (totalOverlap <= settings_.max_overlap_ratio) {
                    std::cout << "added part " << i << std::endl;
                    w[i] = true;
                }
            }
        }
    }
    std::cout << "solution: [";
    for (size_t j=0; j < w.size(); ++j) {
        std::cout << w[j] << ", ";
    }
    construction.setW(w);
    construction.computeTotalMesh();
    return true;
}

int Solver::regularizeDepths() {
    double margin = diameter_ / settings_.voxel_resolution*2;
    return construction.regularizeDepths(margin);
}

void Solver::refineConnectionContacts(bool useCurves) {
    double margin = diameter_ / settings_.voxel_resolution*3;
    std::cout << "refining connection contacts with margin " << margin << std::endl;
    construction.recomputeConnectionContacts(margin, useCurves);
    /*if (connectionsResolved) {
        Construction::edge_iter ei, eb;
        auto &g = construction.g;
        for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
            Construction::Edge e = *ei;
            size_t partIdx1 = g[vertex(g[e].part1, g)].partIdx;
            size_t partIdx2 = g[vertex(g[e].part2, g)].partIdx;
            //decide between corner and slide connection based on stability heuristic
            //partIdx1 is the part that is cut by the connection
            bool sideConnection = sideConnectionValid(g[e]);
            if (sideConnection) {
                std::cout << "changed " << partIdx1 << '-' << partIdx2 << " to side connection" << std::endl;
            }
        }
    }*/
}

double Solver::cornerConnectionPenetration(ConnectionEdge &connection, bool &cutoff) {
    cutoff = false;
    double penetration = 0;
    double voxel_width = diameter_/settings_.voxel_resolution;
    Construction::Vertex v1 = vertex(connection.part1, construction.g);
    Construction::Vertex v2 = vertex(connection.part2, construction.g);
    size_t part1 = construction.g[v1].partIdx;
    size_t part2 = construction.g[v2].partIdx;
    const auto &pd1 = construction.partData[part1];
    const auto &pd2 = construction.partData[part2];
    Vector3d n1 = pd1.normal();
    Vector3d n2 = pd2.normal();
    double offset1 = pd1.offset();
    double offset2 = pd2.offset();
    double depth1 = construction.getStock(part1).thickness;
    double depth2 = construction.getStock(part2).thickness;
    //check if the interface would "cut through" any part of the shape, if so then eliminate
    if (connection.innerEdge.size() > 0) {
        Edge3d innerEdgeFull(connection.innerEdge.getEdge(0).first, connection.innerEdge.getEdge(connection.innerEdge.size()-1).second);
        Edge2d innerEdgeProj(pd1.project(innerEdgeFull.first.transpose()).transpose(),
                             pd1.project(innerEdgeFull.second.transpose()).transpose());
        Vector2d n2proj = pd1.projectDir(n2);
        Vector2d offsetDir = connection.backface2 ? n2proj : -n2proj;
        Vector2d dir = innerEdgeProj.second-innerEdgeProj.first;
        double len = dir.norm();
        dir /= len;
        Ray2d intersector(innerEdgeProj.first + offsetDir * (depth2 + voxel_width * 3), dir, 0, len);
        auto intersections = construction.getShape(part1).cutPath->intersect(intersector);
        std::cout << "part " << part1 << " shape intersects " << part2 << " " << intersections.size() << " times " << std::endl;

        bool entered = false;
        double tLast = 0;
        for (const auto &intersection : intersections) {
            if (intersection.entering) {
                entered = true;
            } else {
                penetration += (intersection.t - tLast);
                if (entered) cutoff = true;
            }
            tLast = intersection.t;
        }
        if (!intersections.empty() && intersections.back().entering) {
            penetration += intersector.end - tLast;
        }
        if (cutoff) {
            std::cout << "part " << part1 << " shape cut off by corner connection with " << part2
                      << " with penetration " << penetration << std::endl;
        }
    }
    return penetration;
}

bool Solver::edgeValues(size_t part1, size_t part2, const Edge3d &edge, const Eigen::Vector3d &faceNormal, const Eigen::Vector3d &connectionNormal, double &outMaxColorDifference, double &outMaxLumDerivative, double &outWeight, int id) {
    double voxel_width = diameter_/settings_.voxel_resolution;
    double edgeLength = (edge.second - edge.first).norm();
    Vector3d edgeDir = (edge.second - edge.first)/edgeLength;
    int visSamples = static_cast<int>(std::round(edgeLength/voxel_width));
    auto visGroup = getVisibility((edge.first + edge.second)/2, faceNormal);
    Matrix<double, 2, 3> visEdgeMat;
    visEdgeMat << edge.first.transpose(), edge.second.transpose();
    //for each image, for each visible subinterval, for each offset
    std::vector<std::vector<std::vector<double>>> luminanceDerivatives;
    // for each view in cluster_visibility[visGroupIndex], all visible sub-intervals
    std::vector<std::vector<std::pair<double, double>>> allIntervals;
    RunningAverage avgMaxLumDerivative;
    RunningAverage avgMaxColorDifference;
    for (auto imageId : visGroup) {
        auto &image = reconstruction_->images[imageId.first];
        auto resolution = reconstruction_->resolution(imageId.first);
        Vector3d cameraOrigin = image.origin();
        std::vector<bool> visibleVector(visSamples, true);
        {
            std::cout << "computing visible sub intervals for view " << imageId.first << std::endl;
            Vector3d samplePoint = edge.first;
            for (size_t visSample = 0; visSample < visSamples; ++visSample) {
                Vector2d projectedSamplePoint = reconstruction_->project(samplePoint.transpose(),
                                                                         imageId.first).transpose();
                if (projectedSamplePoint.x() < 0 || projectedSamplePoint.y() < 0 || projectedSamplePoint.x() >= resolution.x() || projectedSamplePoint.y() >= resolution.y()) {
                    visibleVector[visSample] = false;
                } else {
                    Vector3d rayDir = reconstruction_->initRay(projectedSamplePoint.y(), projectedSamplePoint.x(),
                                                               imageId.first);
                    double tPoint = (samplePoint - cameraOrigin).norm();
                    igl::Hit hit;
                    bool intersected = construction.aabb.intersect_ray(std::get<0>(construction.mesh),
                                                                       std::get<1>(construction.mesh),
                                                                       cameraOrigin.transpose(),
                                                                       rayDir.transpose(), hit);
                    if (intersected) {
                        int intersectedPart = std::get<2>(construction.mesh)(hit.id);
                        if (intersectedPart != part1 && intersectedPart != part2 && hit.t - voxel_width < tPoint) {
                            visibleVector[visSample] = false;
                        }
                    }
                    samplePoint += edgeDir * voxel_width;
                }
            }
        }
        std::cout << "computing intervals" << std::endl;
        std::vector<std::pair<double, double>> intervals;
        {
            std::vector<size_t> crossings;
            bool entering = true;
            for (size_t visSample = 0; visSample < visSamples; ++visSample) {
                if (entering && visibleVector[visSample]) {
                    entering = false;
                    crossings.push_back(visSample);
                } else if (!entering && !visibleVector[visSample]) {
                    entering = true;
                    crossings.push_back(visSample);
                }
            }
            if (!entering) {
                crossings.push_back(visSamples);
            }
            for (size_t crossing = 0; crossing < crossings.size(); crossing += 2) {
                if (crossings[crossing] < crossings[crossing + 1] - MIN_VISIBLE_LENGTH) {
                    intervals.emplace_back(crossings[crossing] * voxel_width,
                                           crossings[crossing + 1] * voxel_width);
                }
            }
        }
        allIntervals.push_back(std::move(intervals));
    }
    std::cout << "populating view indices" << std::endl;
    /** indices of views in cluster_visibility[visGroupIndex] to use */
    std::vector<size_t> viewIndices;
    for (size_t imageInd=0; imageInd < visGroup.size() && viewIndices.size() < settings_.max_views_per_cluster; ++imageInd) {
        auto imageId = visGroup[imageInd];
        auto &image = reconstruction_->images[imageId.first];
        if (allIntervals[imageInd].empty()) continue;
        viewIndices.push_back(imageInd);
    }
    //if unseen, give it a small non-zero score that has to be "beaten" by an existing observation (TODO: Make parameter)
    if (viewIndices.empty()) {
        return false;
    }
    std::cout << "views used: ";
    for (auto index : viewIndices) {
        std::cout << visGroup[index].first << ", ";
    }
    std::cout << std::endl;
    std::vector<double> speeds;
    speeds.reserve(viewIndices.size());
    //std::cout << "precomputing speed" << std::endl;
    for (auto viewInd : viewIndices) {
        size_t image_id = visGroup[viewInd].first;
        auto &image = reconstruction_->images[image_id];
        Vector2d dadt = reconstruction_->directionalDerivative(visEdgeMat.row(0).transpose(), connectionNormal, image_id);
        Vector2d dbdt = reconstruction_->directionalDerivative(visEdgeMat.row(1).transpose(), connectionNormal, image_id);
        speeds.push_back(1.0/std::max(dadt.norm(), dbdt.norm()));
    }
    /** step size in world coordinates to ensure no pixels are missed in the images */
    double worldSpeed = *std::min_element(speeds.begin(), speeds.end());
    double offset_tolerance=diameter_/settings_.voxel_resolution;
    int pixelDistance = static_cast<int>(std::floor(offset_tolerance / worldSpeed));
    /** total offsets to consider */
    int N = pixelDistance * 2 + 1;

    outWeight = 0.0;
    for (auto viewInd : viewIndices) {
        size_t imageId = visGroup[viewInd].first;
        std::cout << "computing image features for " << imageId << std::endl;
        auto &image = reconstruction_->images[imageId];
        luminanceDerivatives.emplace_back();
        for (const auto &interval : allIntervals[viewInd]) {
            double maxLumDerivative = 0.0;
            double maxColorDifference = 0.0;
            Eigen::Matrix<double, 2, 3> edgeMat;
            edgeMat << (edge.first + interval.first * edgeDir).transpose(),
                    (edge.first + interval.second * edgeDir).transpose();
            //std::cout << "computing derivatives for image " << imageId << std::endl;
            luminanceDerivatives.back().emplace_back();
            double totalWeight = 0.0;
            RunningAverage leftColor;
            RunningAverage rightColor;
            for (int i = 0; i < N; i++) {
                double magnitude = static_cast<double>(i - pixelDistance) * worldSpeed;
                /** endpoints of the edge in camera space */
                Matrix2d edgeMatCam = reconstruction_->project(edgeMat.array().rowwise() + magnitude * connectionNormal.transpose().array(), imageId);
                //compute normalized image coordinate derivatives
                //since corners contain infinitely high frequencies, derivative magnitudes are always bounded by sampling rate
                //so we compute image derivatives according to a speed that is uniform across the images
                /** derivatives of pixel coordinates w.r.t. world position at cut endpoints */
                Vector2d dadt = reconstruction_->directionalDerivative(edgeMat.row(0).transpose(), connectionNormal, imageId).normalized();
                Vector2d dbdt = reconstruction_->directionalDerivative(edgeMat.row(1).transpose(), connectionNormal, imageId).normalized();
                /** analytical luminance derivative */
                std::cout << "computing lum derivative " << i << ": ";
                double lumDeriv = integrate_image(image.getDerivativeX(),
                                                  edgeMatCam.row(0).transpose(),
                                                  edgeMatCam.row(1).transpose(),
                                                  true,
                                                  dadt,
                                                  dbdt,
                                                  image.getDerivativeY())(0) / 16.0;
                std::cout << lumDeriv << std::endl;
                std::cout << "computing lum " << i << ": ";
                VectorXd lum = integrate_image(image.getImage(),
                                               edgeMatCam.row(0).transpose(),
                                               edgeMatCam.row(1).transpose());
                std::cout << lum << std::endl;

                double weight = (edgeMatCam.row(1) - edgeMatCam.row(0)).norm();
                if (i < pixelDistance) {
                    leftColor.add(lum, weight);
                } else if (i > pixelDistance) {
                    rightColor.add(lum, weight);
                }
                lumDeriv = std::fabs(lumDeriv);
                luminanceDerivatives.back().back().push_back(lumDeriv);
                maxLumDerivative = std::max(maxLumDerivative, lumDeriv);
                totalWeight += weight;
            }
            outWeight += totalWeight;
            maxColorDifference = std::max(maxColorDifference, (leftColor.get() - rightColor.get()).norm());
            avgMaxLumDerivative.add(maxLumDerivative, totalWeight);
            avgMaxColorDifference.add(maxColorDifference, totalWeight);
        }
    }
    outMaxColorDifference = avgMaxColorDifference.getScalar();
    outMaxLumDerivative = avgMaxLumDerivative.getScalar();
    //std::cout << "debug visualization-saving images..." << std::endl;
    //DEBUG_VISUALIZATION
#ifdef DEBUG_VIS
    std::cout << "visualizing" << std::endl;
        for (size_t viewIndInd=0; viewIndInd < viewIndices.size(); ++viewIndInd) {
            auto viewInd = viewIndices[viewIndInd];
            auto image_id = visGroup[viewInd].first;
            auto &image = reconstruction_->images[image_id];
            for (size_t intervalId=0; intervalId < allIntervals[viewInd].size(); ++intervalId) {
                const auto &interval = allIntervals[viewInd][intervalId];
                Eigen::Matrix<double, 2, 3> edgeMat;
                edgeMat << (edge.first + interval.first * edgeDir).transpose(),
                        (edge.first + interval.second * edgeDir).transpose();
                cv::Mat visImageA = image.getImage(true).clone();
                cv::Mat visImageB = image.getImage(true).clone();
                for (int i = 0; i < N; i += 4) {
                    double magnitude = static_cast<double>(i - pixelDistance) * worldSpeed;
                    Matrix<double, 2, 2> edgeMatCam = reconstruction_->project(
                            edgeMat.array().rowwise() + magnitude * connectionNormal.transpose().array(), image_id);
                    Vector2d dadt =
                            reconstruction_->directionalDerivative(edgeMat.row(0).transpose(), connectionNormal,
                                                                   image_id) * offset_tolerance;
                    Vector2d dbdt =
                            reconstruction_->directionalDerivative(edgeMat.row(1).transpose(), connectionNormal,
                                                                   image_id) * offset_tolerance;

                    cv::Point2d ptA(edgeMatCam(0, 0), edgeMatCam(0, 1));
                    cv::Point2d ptB(edgeMatCam(1, 0), edgeMatCam(1, 1));
                    cv::line(visImageA, ptA,
                             ptB,
                             static_cast<uchar>(std::min(255.0, std::abs(luminanceDerivatives[viewIndInd][intervalId][i] * 10))),
                             2);
                    for (auto &im : std::array<cv::Mat *, 2>{&visImageA, &visImageB}) {
                        cv::Point2d offseta(dadt.x(), dadt.y());
                        cv::Point2d offsetb(dbdt.x(), dbdt.y());
                        auto color = static_cast<uchar>((i * 255) / N);
                        cv::line(*im, ptA, ptA + offseta, color, 4);
                        cv::line(*im, ptB, ptB + offsetb, color, 4);
                    }
                }
                cv::imwrite("cut_deriv_" + std::to_string(part1) + "-" + std::to_string(part2) + "_view" + std::to_string(image_id) + "_edge_" + std::to_string(id) + "_subinterval_" + std::to_string(intervalId) + "_analytic.png", visImageA);
            }
        }
#endif
        return true;
}

double Solver::connectionScore(const ConnectionEdge &connection) {
    double voxel_width = diameter_/settings_.voxel_resolution;
    Construction::Vertex v1 = vertex(connection.part1, construction.g);
    Construction::Vertex v2 = vertex(connection.part2, construction.g);
    size_t part1 = construction.g[v1].partIdx;
    size_t part2 = construction.g[v2].partIdx;
    std::cout << "computing score for connection [" << part1 << '-' << part2 << ']' << std::endl;
    if (!construction.hasTotalMeshInfo) {
        std::cout << "no mesh info!!" << std::endl;
    }
    const auto &pd1 = construction.partData[part1];
    const auto &pd2 = construction.partData[part2];
    Vector3d n1 = pd1.normal();
    Vector3d n2 = pd2.normal();
    double offset1 = pd1.offset();
    double offset2 = pd2.offset();
    double depth1 = construction.getStock(part1).thickness;
    double depth2 = construction.getStock(part2).thickness;
    size_t numParts = construction.partData.size();

    //if backface of part1 aligns with the cut, inspect images that see that side
    size_t visGroupIndex = connection.backface1 ? part1 + numParts : part1;
    Vector3d faceNormal = connection.backface1 ? -n1 : n1;
    Vector3d connectionNormal = connection.backface2 ? -n2 : n2; //normal of face part at connection
    RunningAverage avgMaxLumDerivative;
    RunningAverage avgMaxColorDifference;
    for (size_t edgeIndex=0; edgeIndex < connection.innerEdge.size(); ++edgeIndex) {
        Edge3d visEdge = connection.innerEdge.getEdge(edgeIndex);
        for (int x = 0; x < 2; ++x) {
            auto &pt = x == 0 ? visEdge.first : visEdge.second;
            pt += depth1 * faceNormal;
        }

        double outMaxColorDifference, outMaxLumDerivative, outWeight;
        bool success = edgeValues(part1, part2, visEdge, faceNormal, connectionNormal, outMaxColorDifference, outMaxLumDerivative, outWeight, edgeIndex);
        if (success) {
            avgMaxLumDerivative.add(outMaxLumDerivative, outWeight);
            avgMaxColorDifference.add(outMaxColorDifference, outWeight);
        }
    }
    //add cut surface edges
//    {
//        Edge3d visEdge;
//        visEdge.first = connection.innerEdge.getEdge(0).first;
//        visEdge.second = visEdge.first + depth1 * faceNormal;
//        double outMaxColorDifference, outMaxLumDerivative, outWeight;
//        bool success = edgeValues(part1, part2, visEdge, -connection.innerEdge.d, connectionNormal, outMaxColorDifference, outMaxLumDerivative, outWeight, 1001);
//
//        if (success) {
//            avgMaxLumDerivative.add(outMaxLumDerivative, outWeight);
//            avgMaxColorDifference.add(outMaxColorDifference, outWeight);
//        }
//    }
//    {
//        Edge3d visEdge;
//        visEdge.first = connection.innerEdge.getEdge(connection.innerEdge.size()).second;
//        visEdge.second = visEdge.first + depth1 * faceNormal;
//        double outMaxColorDifference, outMaxLumDerivative, outWeight;
//        bool success = edgeValues(part1, part2, visEdge, connection.innerEdge.d, connectionNormal, outMaxColorDifference, outMaxLumDerivative, outWeight, 1002);
//
//        if (success) {
//            avgMaxLumDerivative.add(outMaxLumDerivative, outWeight);
//            avgMaxColorDifference.add(outMaxColorDifference, outWeight);
//        }
//    }

    //std::cout << "saved" << std::endl;
    std::cout << "max lum derivative: " << avgMaxLumDerivative.getScalar() << std::endl;
    std::cout << "max color difference: " << avgMaxColorDifference.getScalar() << std::endl;
    return avgMaxLumDerivative.getScalar() + avgMaxColorDifference.getScalar();
}

void Solver::initializeConnectionTypes() {
    double margin = diameter_/settings_.voxel_resolution * 3;
    Construction::edge_iter ei, eb;
    auto &g = construction.g;
//    Construction::IndexMap index = get(vertex_index, g);
    for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
        Construction::Edge e = *ei;
        if (g[e].optimizeStage > 0) continue;
        bool valid12, valid21;
        //search for convex shape edges shared by parts; these must be corners
        //they also enable a special method for narrowing down connection orientation
        //if this fails to eliminate either case, fall back on images
        size_t partIdx1 = g[vertex(g[e].part1, g)].partIdx;
        size_t partIdx2 = g[vertex(g[e].part2, g)].partIdx;
        const auto &pd1 = construction.partData[partIdx1];
        const auto &pd2 = construction.partData[partIdx2];
        auto &sd = construction.getShape(partIdx1);
        double dotprod = std::fabs(pd1.normal().dot(pd2.normal()));
//        if (dotprod > settings_.norm_parallel_threshold) {
////            g[e].type = ConnectionEdge::CET_FACE_TO_FACE;
//            //TODO: logic for cut to cut from opposite sides
//        } else if (dotprod < settings_.norm_adjacency_threshold) {
            g[e].type = ConnectionEdge::CET_CUT_TO_FACE;
            ConnectionEdge c1 = g[e];
            ConnectionEdge c2 = g[e];
            std::swap(c2.part1, c2.part2);

            valid12 = construction.connectionValid(c1, margin);
            valid21 = construction.connectionValid(c2, margin);
            std::cout << "connection " << partIdx1 << "-" << partIdx2 << (valid12 ? "valid" : "not valid") << ", backface: " << c1.backface2 << std::endl;
            std::cout << "connection " << partIdx2 << "-" << partIdx1 << (valid21 ? "valid" : "not valid") << ", backface: " << c2.backface2 << std::endl;

            if (!valid12 || !valid21) {
                const auto &constraints1 = construction.getShape(partIdx1).shapeConstraints;
                //const auto &constraints2 = construction.getShape(partIdx2).shapeConstraints;
                auto it1 = constraints1.find(partIdx2);
//                auto it2 = constraints2.find(partIdx1);
                bool corner1 = false;//, corner2 = false;
//                int constraintInd1;//, constraintInd2;
                double maxCornerLength2 = 0.0;
                bool backface12, backface21;
                if (it1 != constraints1.end()) {
                    for (int ci = 0; ci < it1->second.size(); ++ci) {
                        const auto &cc = it1->second[ci];
                        if (cc.convex) {
                            std::cout << "convex corner from " << partIdx2;
                            if (cc.otherOpposing) std::cout << " (otherOpposing)";
                            std::cout << std::endl;
                            //if 1-2 or 2-1 is invalid, no need to check consistency for that face
                            //otherwise, corner must come from the side opposite the connection
                            if ((!valid12 || (c1.backface2 ^ cc.otherOpposing)) ||
                                    (!valid21 || (c2.backface2 ^ cc.opposing))) {
                                double edgeLen2 = (cc.edge.second - cc.edge.first).squaredNorm();
                                if (edgeLen2 > maxCornerLength2) {
                                    corner1 = true;
                                    maxCornerLength2 = edgeLen2;
                                    backface12 = !cc.otherOpposing;
                                    backface21 = !cc.opposing;
                                }
                            }
                        }
                    }
                }
                /*if (it2 != constraints2.end()) {
                    for (int ci = 0; ci < it2->second.size(); ++ci) {
                        const auto &cc = it2->second[ci];
                        if (cc.convex) {
                            std::cout << "convex corner from " << partIdx1;
                            if (cc.otherOpposing) std::cout << " (otherOpposing)";
                            std::cout << std::endl;
                            if (!valid21 || (c2.backface2 ^ cc.otherOpposing)) {
                                corner2 = true;
                                constraintInd2 = ci;
                                break;
                            }
                        }
                    }
                }*/

                if (corner1) {
                    std::cout << "convex corner detected for connection " << partIdx1 << '-' << partIdx2
                              << "; assuming corner connection" << std::endl;
                    //if corner from part2 is from its backface, the connection meets its front face
                    if (!valid12)
                        c1.backface2 = backface12;
                    if (!valid21)
                        c2.backface2 = backface21;
                    valid12 = true;
                    valid21 = true;
//            hasConvexCorner = true;
                } else {
                    std::cout << "convex corner " << partIdx1 << '-' << partIdx2 << " inconsistent with geometry"
                              << std::endl;
                }
            }
            g[e] = c1;
            if (valid12 && valid21) {
                g[e].type = ConnectionEdge::CET_CORNER;
                g[e].backface1 = !c2.backface2;
                g[e].backface2 = c1.backface2;
            } else if (valid21 && !valid12) {
                std::swap(g[e].part1, g[e].part2);
                g[e] = c2;
            }
//        } else {
//            g[e].type = ConnectionEdge::CET_UNKNOWN;
//            //TODO: validate contacts in non-corner case
//        }
    }
}

void Solver::optimizeConnections(bool useImages) {
    double voxel_width = diameter_/settings_.voxel_resolution;
    if (useImages) reconstruction_->setImageScale(settings_.image_scale);
    Construction::edge_iter ei, eb;
    auto &g = construction.g;
//    Construction::IndexMap index = get(vertex_index, g);
    for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
        Construction::Edge e = *ei;
        size_t partIdx1 = g[vertex(g[e].part1, g)].partIdx;
        size_t partIdx2 = g[vertex(g[e].part2, g)].partIdx;
        if (g[e].optimizeStage > 0) {
            std::cout << "skipping connection " << partIdx1 <<'-'<<partIdx2 << " at stage " << g[e].optimizeStage <<std::endl;
            continue;
        }
        if (g[e].type == ConnectionEdge::CET_CORNER) {
            auto &sd = construction.getShape(partIdx1);
            const auto &constraints1 = construction.getShape(partIdx1).shapeConstraints;
            ConnectionEdge c1 = g[e];
            ConnectionEdge c2 = g[e];
            std::swap(c2.part1, c2.part2);
            std::swap(c2.backface1, c2.backface2);
            c2.backface1 = !c2.backface1;
            c2.backface2 = !c2.backface2;

            std::cout << "validating corner connection [" << partIdx1 << '-' << partIdx2 << ']' << std::endl;
            bool c1cutoff;
            bool c2cutoff;
            double c1penetration = cornerConnectionPenetration(c1, c1cutoff);
            double c2penetration = cornerConnectionPenetration(c2, c2cutoff);
            double score12= c1cutoff ? 0 : 1, score21= c2cutoff ? 0 : 1;

            //TODO: discard connection if both penetrate
            if (useImages && !(c1cutoff ^ c2cutoff)) {
                std::cout << "ambiguous connection; computing scores using images" << std::endl;
                score12 = connectionScore(c1);
                score21 = connectionScore(c2);
            }
            std::cout << "score12: " << score12 << "; score21: " << score21 << std::endl;
            if (score21 > score12) {
                std::swap(g[e].part1, g[e].part2);
                std::swap(c1, c2);
                std::swap(partIdx1, partIdx2);
            }
            std::cout << "chose connection [" << partIdx1 <<
                      '-' << partIdx2 << ']' << std::endl;
            g[e].backface1 = !c2.backface2;
            g[e].backface2 = c1.backface2;
        }
        if (g[e].type == ConnectionEdge::CET_CUT_TO_FACE || g[e].type == ConnectionEdge::CET_CORNER) {
            construction.getShape(partIdx1).dirty = true;
        }
        g[e].optimizeStage = 1;
    }
}

bool Solver::sideConnectionValid(ConnectionEdge &connection) {
    double voxel_width = diameter_/settings_.voxel_resolution;
    Construction::Vertex v1 = vertex(connection.part1, construction.g);
    Construction::Vertex v2 = vertex(connection.part2, construction.g);
    size_t partIdx1 = construction.g[v1].partIdx;
    size_t partIdx2 = construction.g[v2].partIdx;
    std::cout << "validating cut-cut connection [" << partIdx1 << '-' << partIdx2 << ']' << std::endl;
    const auto &pd1 = construction.partData[partIdx1];
    const auto &pd2 = construction.partData[partIdx2];
    Vector3d n1 = pd1.normal();
    Vector3d n2 = pd2.normal();
    double offset1 = pd1.offset();
    double offset2 = pd2.offset();
    double depth1 = construction.getStock(partIdx1).thickness;
    double depth2 = construction.getStock(partIdx2).thickness;

    if (connection.type == ConnectionEdge::CET_CORNER) {
        std::cout << "analyzing part " << partIdx1 << " as cut by part " << partIdx2 << std::endl;
        bool stableConnectionExists = false;
        for (size_t edgeIndex=0; edgeIndex < connection.innerEdge.size(); ++edgeIndex) {
            Edge3d edge3d = connection.innerEdge.getEdge(edgeIndex);
            Edge2d edge2d(construction.partData[partIdx1].project(edge3d.first.transpose()).transpose(),
                          construction.partData[partIdx1].project(edge3d.second.transpose()).transpose());
            Vector2d dir = edge2d.second - edge2d.first;
            Vector2d edgeNormal(-dir.y(), dir.x());
            Vector2d n = construction.partData[partIdx1].projectDir(
                    construction.partData[partIdx2].normal());
            if (connection.backface2) {
                n = -n;
            }
            if (edgeNormal.dot(n) < 0) {
                std::swap(edge2d.first, edge2d.second);
            }
            double gap;
            double cornerConnectionThickness = primitive_thickness(*construction.getShape(partIdx1).cutPath, edge2d,
                                                                   voxel_width, gap);
            std::cout << "corner thickness: " << cornerConnectionThickness << " vs thickness " << construction.getStock(partIdx2).thickness << std::endl;

            if (cornerConnectionThickness >= construction.getStock(partIdx2).thickness) {
                stableConnectionExists = true;
                break;
            }
        }
        if (stableConnectionExists) {
            return false;
        } else {
            std::cout << "no stable connection" << std::endl;
            connection.type = ConnectionEdge::CET_CUT_TO_CUT;
        }
    }

    if (connection.type == ConnectionEdge::CET_CUT_TO_CUT) {
        for (size_t edgeIndex = 0; edgeIndex < connection.innerEdge.size(); ++edgeIndex) {
            Edge3d edge3d = connection.innerEdge.getEdge(edgeIndex);
            Edge2d edge2d(construction.partData[partIdx1].project(edge3d.first.transpose()).transpose(),
                          construction.partData[partIdx1].project(edge3d.second.transpose()).transpose());
            Vector2d dir = edge2d.second - edge2d.first;
            Vector2d edgeNormal(-dir.y(), dir.x());
            Vector2d n = -construction.partData[partIdx1].projectDir(
                    construction.partData[partIdx2].normal());
            if (edgeNormal.dot(n) < 0) {
                std::swap(edge2d.first, edge2d.second);
            }
            double gap;
            double cornerConnectionThickness = primitive_thickness(*construction.getShape(partIdx1).cutPath, edge2d,
                                                                   voxel_width, gap);
            Edge2d slideEdge1(edge2d.first, edge2d.first + n * construction.getStock(partIdx2).thickness);
            Edge2d slideEdge2(edge2d.second + n * construction.getStock(partIdx2).thickness, edge2d.second);
//                std::cout << "slideEdge1: " << slideEdge1.first.transpose() << "; " << slideEdge1.second.transpose() << std::endl;
//                std::cout << "slideEdge2: " << slideEdge2.first.transpose() << "; " << slideEdge2.second.transpose() << std::endl;
            double sideGap1, sideGap2;
            double slideConnectionThickness1 = primitive_thickness(*construction.getShape(partIdx1).cutPath, slideEdge1,
                                                                   voxel_width, sideGap1);
            double slideConnectionThickness2 = primitive_thickness(*construction.getShape(partIdx1).cutPath, slideEdge2,
                                                                   voxel_width, sideGap2);
            std::cout << "corner thickness: " << cornerConnectionThickness << " ("
                      << cornerConnectionThickness - construction.getStock(partIdx2).thickness << "); slide1: "
                      << slideConnectionThickness1 << "; slide2: " << slideConnectionThickness2 << std::endl;

            cornerConnectionThickness -= construction.getStock(partIdx2).thickness;
            if (cornerConnectionThickness < construction.getStock(partIdx2).thickness) {
                Vector3d interfaceDir = construction.partData[partIdx1].normal().cross(
                        construction.partData[partIdx2].normal());
                Vector3d edgeDir3d = construction.partData[partIdx1].unproject(dir.transpose()).transpose();
                if (edgeDir3d.dot(interfaceDir) < 0) {
                    interfaceDir = -interfaceDir;
                }
                if (slideConnectionThickness1 > voxel_width) {
                    connection.interfaces.push_back({edge3d.first, -interfaceDir});
                }
                if (slideConnectionThickness2 > voxel_width) {
                    connection.interfaces.push_back({edge3d.second, interfaceDir});
                }
            }
        }
    }
    if (connection.interfaces.empty()) {
        connection.type = ConnectionEdge::CET_UNKNOWN;
        return false;
    }
    return true;
}

bool Solver::hasNormals() const {
    return cloud_->N.rows() == cloud_->P.rows();
}

void Solver::setDataPoints(PointCloud3::Handle cloud) {
    cloud_ = std::move(cloud);
}

void Solver::computeBounds() {
    RowVector3d bboxMin = cloud_->P.colwise().minCoeff();
    RowVector3d bboxMax = cloud_->P.colwise().maxCoeff();
    RowVector3d bbox = bboxMax - bboxMin;
    diameter_ = bbox.maxCoeff();
    floorHeight = bboxMin.z();// + diameter_/settings_.voxel_resolution;
}

void Solver::setReconstructionData(ReconstructionData::Handle rec) {
    reconstruction_ = std::move(rec);
}

void Solver::visualize() const {
    construction.visualize(diameter_/settings_.voxel_resolution/100, settings_.connector_mesh, settings_.connector_spacing, settings_.connector_scale);
}

/*bool Solver::exportMesh(const std::string &filename) const {
    return construction.exportMesh(filename);
}

bool Solver::exportModel(const std::string &filename) const {
    return construction.exportModel(filename);
}*/


int Solver::buildGraph() {
    hasGraph = true;
    return construction.buildGraph(diameter_/settings_.voxel_resolution * 4);
}

int Solver::findNewConnections() {
    return construction.findNewConnections(diameter_/settings_.voxel_resolution * 4);
}


struct RelationEdge {
    enum Type {
        REL_PARALLEL=0,
        REL_ORTHOGONAL=1
    };
    Type type;
};
struct RelationNode {
    /** list of (partIdx, flip) */
    std::vector<std::pair<size_t, bool>> allIds;
};

void Solver::realign() {
    //detect parallel


    typedef adjacency_list<vecS, vecS, undirectedS, RelationNode, RelationEdge> RelationGraph;
    typedef graph_traits<RelationGraph>::vertex_iterator vertex_iter;
    typedef graph_traits<RelationGraph>::edge_iterator edge_iter;
    typedef graph_traits<RelationGraph>::out_edge_iterator out_edge_iter;
    typedef graph_traits<RelationGraph>::vertex_descriptor Vertex;
    typedef graph_traits<RelationGraph>::edge_descriptor Edge;
    typedef graph_traits<RelationGraph>::adjacency_iterator adjacency_iter;
    typedef property_map<RelationGraph, vertex_index_t>::type IndexMap;

    RelationGraph g(num_vertices(construction.g));

    {
        Construction::edge_iter ei, eb;
        Construction::IndexMap index = get(vertex_index, construction.g);
        Construction::vertex_iter vi, vend;
        for (boost::tie(vi, vend) = vertices(construction.g); vi != vend; ++vi) {
            Construction::Vertex v1 = *vi;
            g[vertex(index[v1], g)].allIds.emplace_back(construction.g[v1].partIdx, false);
        }

        //add orthogonal constraints from connections
        for (boost::tie(ei, eb) = edges(construction.g); ei != eb; ++ei) {
            Construction::Edge e = *ei;
            if (construction.g[e].type == ConnectionEdge::CET_CUT_TO_FACE || construction.g[e].type == ConnectionEdge::CET_CORNER) {
                size_t partIdx1 = construction.g[source(e, construction.g)].partIdx;
                size_t partIdx2 = construction.g[target(e, construction.g)].partIdx;
                Vector3d n1 = construction.partData[partIdx1].normal();
                Vector3d n2 = construction.partData[partIdx2].normal();
                if (std::fabs(n1.dot(n2)) < settings_.norm_adjacency_threshold) {
                    if (!construction.g[e].innerEdge.ranges.empty()) {
                        bool inserted;
                        Edge enew;
                        boost::tie(enew, inserted) = add_edge(index[source(e, construction.g)],
                                                              index[target(e, construction.g)], g);
                        g[enew].type = RelationEdge::REL_ORTHOGONAL;
                    }
                }
            }
        }
    }
    //add parallel constraints
    vertex_iter vi, vend;
    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
        Vertex v1 = *vi;
        for (vertex_iter vi2=vi+1; vi2 != vend; ++vi2) {
            Vertex v2 = *vi2;
            Vector3d normal1 = construction.partData[g[v1].allIds[0].first].normal();
            Vector3d normal2 = construction.partData[g[v2].allIds[0].first].normal();
            if (std::abs(normal1.dot(normal2)) > settings_.align_parallel_threshold) {
                //ADD TO RELATION GRAPH
                bool inserted;
                Edge enew;
                boost::tie(enew, inserted) = add_edge(v1, v2, g);
                g[enew].type = RelationEdge::REL_PARALLEL;
            }
        }
    }
    std::cout << "reducing graph..." << std::endl;
    bool has_parallel_edge = true;
    size_t iter = 0;
    while (has_parallel_edge) {
        IndexMap index = get(vertex_index, g);
//        adjacency_iter ai, ab;
        out_edge_iter out_i, out_end;
        //DEBUG
        for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
            Vertex v1 = *vi;
            std::cout << "vertex " << index[v1] << " (";
            for (auto part : g[v1].allIds) {
                if (part.second) std::cout << '!';
                std::cout << part.first << ", ";
            }
            std::cout << "): ";
            for (tie(out_i, out_end) = out_edges(v1, g); out_i != out_end; ++out_i) {
                Edge e = *out_i;
                Vertex adj = target(e, g) == v1 ? source(e, g) : target(e, g);
                if (g[e].type == RelationEdge::REL_PARALLEL) std::cout << "p ";
                std::cout << index[adj] << ", ";
            }
            std::cout << std::endl;
        }
        //
        has_parallel_edge = false;
        edge_iter ei, eb;
        for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
            Edge e = *ei;
            if (g[e].type == RelationEdge::REL_PARALLEL) {
                has_parallel_edge = true;
                Vertex v1 = source(e, g);
                Vertex v2 = target(e, g);
                //delete v2, merge all connections into v1
                for (tie(out_i, out_end) = out_edges(v2, g); out_i != out_end; ++out_i) {
                    Edge ae = *out_i;
                    Vertex adj = target(ae, g);
                    if (adj != v1 && !edge(adj, v1, g).second) {
                        bool inserted;
                        Edge enew;
                        boost::tie(enew, inserted) = add_edge(v1, adj, g);
                        g[enew] = g[ae];
                        std::cout << "connected " << index[adj] << " to " << index[v1] << " (type " << g[enew].type << " from " << g[ae].type << ')' << std::endl;
                    }
                }
                //g[v1].allIds.insert(g[v1].allIds.end(), g[v2].allIds.begin(), g[v2].allIds.end());
                //insert all nodes, flipping if the "base" part has opposite normal
                Vector3d n = construction.partData[g[v1].allIds[0].first].normal();
                Vector3d n2 = construction.partData[g[v2].allIds[0].first].normal();
                bool flip = n.dot(n2) < 0.0;
                for (const auto &pair : g[v2].allIds) {
                    g[v1].allIds.emplace_back(pair.first, flip == !pair.second);
                }

                std::cout << "removing vertex " << index[v2] << "... ";
                clear_vertex(v2, g);
                remove_vertex(v2, g);
                std::cout << "removed" << std::endl;
                break;
            }
        }
        ++iter;
    }
    std::cout << "reduced from " << num_vertices(construction.g) << " to " << num_vertices(g) << " vertices" << std::endl;
    // build optimization problem from reduced relation graph
    std::vector<size_t> dimToCluster;
    std::vector<std::pair<size_t, size_t>> orthogonalities;
    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
        Vertex v1 = *vi;
        dimToCluster.emplace_back(g[v1].allIds[0].first);
    }

    edge_iter ei, eb;
    IndexMap index = get(vertex_index, g);
    for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
        Edge e = *ei;
        size_t i1 = index[source(e, g)];
        size_t i2 = index[target(e, g)];
        if (i1 != i2)
            orthogonalities.emplace_back(i1, i2);
    }

    auto itEnd = std::unique(orthogonalities.begin(), orthogonalities.end());
    std::vector<std::pair<size_t, size_t>> newOrthogonalities(orthogonalities.begin(), itEnd);

    std::cout << "produce initial condition" << std::endl;
    //convert current configuration into initial population
    std::cout << "num vertices: " << num_vertices(g) << std::endl;
    vector_double x(num_vertices(g) * 3, 0);
    std::cout << "pop params: " << x.size() << std::endl;
    size_t i=0;
    std::vector<double> baseOffsets(num_vertices(g));
    std::vector<Quaterniond> baseRotations(num_vertices(g));
    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi,++i) {
        Vertex v1 = *vi;
        Vector3d n = construction.partData[g[v1].allIds[0].first].normal();
        baseOffsets[i] = construction.partData[g[v1].allIds[0].first].offset();
        baseRotations[i] = Quaterniond::FromTwoVectors(Vector3d(0, 0, 1), n);
    }

    std::cout << "initialize problem" << std::endl;
    double offsetTol = diameter_/settings_.master_resolution;
    //offsetTol * offsetScale = alignment_tol
    double offsetScale = settings_.alignment_tol/offsetTol;
    double totalScale = settings_.alignment_stride/(diameter_ * diameter_ * cloud_->P.rows());
    //TODO: get rid of dimtocluster
    std::vector<std::vector<int>> clusters(construction.partData.size());
    for (size_t c=0; c < clusters.size(); ++c) {
        clusters[c] = construction.partData[c].pointIndices;
    }
    AlignmentProblem prob(cloud_, std::move(baseOffsets), baseRotations, std::move(newOrthogonalities), std::move(dimToCluster), clusters, diameter_/5, offsetScale, totalScale, settings_.alignment_stride);
    problem p{std::move(prob)};
    //NLOPT algorithms that take equality constraints: slsqp, auglag
    //otherwise IPOPT
    /* nlopt alg("auglag"); //needs subsidiary local optimizer (use gradients)
    nlopt subsid("tnewton");
    subsid.set_xtol_abs(settings_.alignment_tol);
    alg.set_local_optimizer(std::move(subsid));*/

    nlopt alg("slsqp");
    alg.set_xtol_abs(settings_.alignment_tol);

    std::cout << "initialize pop" << std::endl;
    population pop(p);
    std::cout << "insert" << std::endl;
    pop.push_back(std::move(x));
    // 4 - Evolve the population
    std::cout << "initialized population" << std::endl;
    //std::cout << pop << std::endl;
    std::cout << "nf: " << pop.get_f()[pop.best_idx()].size() << std::endl;
#ifdef TESTGRAD
    double error = testGradient(p, pop.get_x()[pop.best_idx()], settings_.alignment_tol);
    std::cout << "max gradient error: " << error << std::endl;
    error = testHessian(p, pop.get_x()[pop.best_idx()], settings_.alignment_tol);
    std::cout << "max hessian error: " << error << std::endl;
#endif
    auto start_t = clock();
    pop = alg.evolve(pop);
    auto total_t = clock() - start_t;
    float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
    std::cout << "optimization finished in " << time_sec << " seconds" << std::endl;
    const auto &vec = pop.get_x()[pop.best_idx()];
    const auto &f = pop.get_f()[pop.best_idx()];
    //std::cout << pop << std::endl;
    //set model parameters
    i=0;
    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi,++i) {
        //TODO: rotate about centroid
        Vertex v1 = *vi;
        for (const auto & pair : g[v1].allIds) {
            size_t partIdx = pair.first;
            PartData &pd = construction.partData[partIdx];
            Vector3d n = pd.normal();
            double offset = pd.offset();
            Vector3d newN = baseRotations[i] * AlignmentProblem::unitvec(vec[i*3], vec[i*3+1]);
            double nNorm = newN.norm();
            newN /= nNorm;
            //TODO: re-shift offsets based on normalization error
            if (pair.second) {
                newN = -newN;
            }
            Quaterniond newR = Quaterniond::FromTwoVectors(n, newN);
            pd.rot = newR * pd.rot;
            //TODO: handle offsets of equivalent planes better
            double offsetShift = offset + offsetScale * vec[i * 3 + 2] - pd.offset();
            // e3^T . (R^T . (x-T)) = 0
            // n^T . (x-T) = 0
            // n . x - n . T = 0
            // n . x + offset = 0
            // offset = - n . T
            // offset + shift = - n . (T - shift * n)
            pd.pos -= offsetShift * newN;
        }
    }
//    construction.recomputeConnectionContacts();
}

void Solver::recomputeConnectionConstraints() {
    double margin = diameter_/settings_.voxel_resolution * 3;
    construction.recomputeConnectionConstraints(margin, settings_.norm_parallel_threshold);
    construction.pruneShapeConstraints();
}

void Solver::optimizeShapes(bool useConstraints) {
    Construction::IndexMap index = get(vertex_index, construction.g);
    Construction::vertex_iter vi, vend;
    double voxel_width = diameter_/settings_.voxel_resolution;
    double margin = voxel_width * settings_.constraint_factor;
    double error_scale = diameter_/settings_.master_resolution;
    double bezier_cost = settings_.curve_cost * error_scale * error_scale;
    double line_cost = settings_.line_cost * error_scale * error_scale;
    {
        //const size_t nthreads = omp_get_max_threads();
        size_t N = num_vertices(construction.g);
        auto t_start = clock();
        auto t_start_w = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for default(none) shared(N,index,margin,construction, bezier_cost, line_cost, voxel_width)
        for (size_t i=0; i<N; ++i) {
            Construction::Vertex v = vertex(i, construction.g);
            size_t partIdx1 = construction.g[v].partIdx;
            std::cout << "processing part " << partIdx1 << std::endl;
            PartData &pd1 = construction.partData[partIdx1];
            if (pd1.groundPlane) continue;
            Vector3d n1 = pd1.normal();
            const auto &oldShape = construction.getShape(partIdx1).cutPath;
            MatrixX2d points = oldShape->points();
            //collect shape constraints from neighboring parts
            std::vector<LineConstraint> constraints;
            if (useConstraints) {
                for (const auto &pair : construction.getShape(partIdx1).connectionConstraints) {
                    for (const auto &constraint : pair.second) {
                        if (constraint.useInCurveFit) {
                            LineConstraint lc;
                            lc.edge = constraint.edge;
                            lc.alignable = false;
                            lc.threshold = margin;
                            Vector3d n2 = construction.partData[pair.first].normal();
                            double angCos = -n2.dot(n1);
                            auto v2 = construction.partIdxToVertex(pair.first);
                            if (!v2.second) continue;
                            auto edgeQuery = edge(v, v2.first, construction.g);
                            if (!edgeQuery.second) continue;
                            if (construction.g[edgeQuery.first].backface2) {
                                angCos = -angCos;
                            }
                            if (std::fabs(angCos) >= settings_.norm_adjacency_threshold) {
                                //tilted constraint
                                lc.tiltAngCos = angCos;
                            }
                            auto vo = construction.partIdxToVertex(pair.first);
                            if (vo.second) {
                                lc.uniqueId = index[vo.first];
                            } else { lc.uniqueId = -((int) pair.first); }
                            std::cout << "connection constraint id " << lc.uniqueId << std::endl;
                            constraints.push_back(std::move(lc));
                        }
                    }
                }
                for (const auto &pair : construction.getShape(partIdx1).shapeConstraints) {
                    for (const auto &constraint : pair.second) {
                        LineConstraint lc;
                        lc.edge = constraint.edge;
                        lc.alignable = true;
                        lc.threshold = voxel_width;
                        lc.uniqueId = -((int) pair.first);
                        //TODO: handle tilts here as well
                        std::cout << "shape constraint id " << lc.uniqueId << std::endl;
                        constraints.push_back(std::move(lc));
                    }
                }
            }
            CombinedCurve newCurve;
            double gridSpacing = construction.getShape(partIdx1).gridSpacing;
            int ksize = std::max(3, static_cast<int>(std::ceil((5 * voxel_width) / gridSpacing)));
            std::cout << "fitting shape with ksize " << ksize << ", spacing " << gridSpacing << ", " << constraints.size() << " constraints" << std::endl;
//            DECLARE_TIMING(curve);
//            START_TIMING(curve);
            double L2 = newCurve.fit(points, settings_.min_knot_angle, settings_.max_knots,
                                     bezier_cost, line_cost, settings_.curve_weight, 0, -1, ksize,
                                     Vector2d(0, 0), Vector2d(0, 0), constraints);
            //debug
            std::string name = "part" + std::to_string(partIdx1) + "_curveFit_" + std::to_string(constraints.size()) + "_constraints";
            if (settings_.debug_visualization) {
                display_fit(name + ".png", newCurve, points, ksize, true, constraints);
                std::ofstream of(name + ".txt");
                newCurve.exportPlaintext(of);
            }
            std::cout << "curve size: " << newCurve.size() << std::endl;
            //find best direction to align curves
            Vector2d upDir(0, 1);
            if (constraints.empty()) {
                int numAligned = newCurve.align(voxel_width, ANGLE_THRESHOLD, upDir);
                std::cout << "aligment with predefined direction: " << numAligned << " alignments" << std::endl;
            } else {
                CombinedCurve alignedCurve;
                int maxAligned = 0;
                int alignConstraint=-1;
                for (size_t constraintInd=0; constraintInd<constraints.size(); ++constraintInd) {
                    const auto &constraint = constraints[constraintInd];
                    CombinedCurve curveCopy(newCurve);
//                    std::cout << "copy size: " << curveCopy.size() << std::endl;
                    Vector2d dir = (constraint.edge.second - constraint.edge.first).transpose();
                    int numAligned = curveCopy.align(voxel_width, ANGLE_THRESHOLD, dir);
//                    std::cout << "copy size after alignment: " << curveCopy.size() << std::endl;
                    std::cout << "alignment with " << constraintInd << " (" << constraint.uniqueId << ") constraint direction " << dir.transpose() << ": " << numAligned << " aligned" << std::endl;
                    if (numAligned > maxAligned) {
                        maxAligned = numAligned;
                        alignedCurve = std::move(curveCopy);
//                        std::cout << "aligned curve size: " << alignedCurve.size() << std::endl;
                        alignConstraint = constraintInd;
                        upDir = dir;
                    }
                }
                if (alignConstraint >= 0) {
                    std::cout << "used constraint " << alignConstraint << std::endl;
                    newCurve = std::move(alignedCurve);
                }
            }
            if (settings_.debug_visualization) {
                display_fit(name + "_aligned.png", newCurve, points, ksize, true, constraints);
                std::ofstream of(name + "_aligned.txt");
                newCurve.exportPlaintext(of);
            }
//            std::cout << "curve size after alignment: " << newCurve.size() << std::endl;
//            newCurve.fixKnots(60/180.0*M_PI, points);
//            if (settings_.debug_visualization) display_fit(name + "_aligned_fixedKnots.png", newCurve, points, ksize, true, constraints, thresholds);
            std::cout << "curve size after fixing knots: " << newCurve.size() << std::endl;
//            newCurve.ransac(points, 0.1, M_PI/12, diameter_/settings_.master_resolution, random_);
//            STOP_TIMING(curve);
//            PRINT_TIMING(curve);

            //handle holes
            std::vector<std::shared_ptr<Primitive>> holes;
            size_t childIndex = 0;
            for (const auto &child : oldShape->children()) {
                std::cout << "processing child " << childIndex << std::endl;
                MatrixX2d childPoints = child->points().colwise().reverse();
                CombinedCurve childCurve;
//                DECLARE_TIMING(hole);
//                START_TIMING(hole);
                double L2c = childCurve.fit(childPoints, settings_.min_knot_angle, settings_.max_knots,
                                            bezier_cost, line_cost, settings_.curve_weight, 0, -1, ksize, Vector2d(0, 0), Vector2d(0, 0), constraints);
                if (settings_.debug_visualization) {
                    display_fit(name + "_child" + std::to_string(childIndex) + ".png", childCurve, childPoints, ksize,
                                true, constraints);
                    std::ofstream of(name + "_child" + std::to_string(childIndex) + ".txt");
                    childCurve.exportPlaintext(of);
                }
                childCurve.align(voxel_width, ANGLE_THRESHOLD, upDir);

                if (settings_.debug_visualization) {
                    display_fit(name + "_child"+std::to_string(childIndex)+"_aligned.png", childCurve, childPoints, ksize, true, constraints);
                    std::ofstream of(name + "_child"+std::to_string(childIndex)+"_aligned.txt");
                    childCurve.exportPlaintext(of);
                }

//                childCurve.fixKnots(60/180.0*M_PI, childPoints);
//                if (settings_.debug_visualization) display_fit(name + "_child"+std::to_string(childIndex)+"_aligned_fixedKnots.png", childCurve, childPoints, ksize, true, constraints, thresholds);

//                childCurve.ransac(points, 0.1, M_PI/12, diameter_/settings_.master_resolution, random_);
//                STOP_TIMING(hole);
//                PRINT_TIMING(hole);
                holes.emplace_back(new PolyCurveWithHoles(std::move(childCurve)));
                ++childIndex;
            }
            if (newCurve.size() != 0) {
                construction.getShape(partIdx1).cutPath = std::make_shared<PolyCurveWithHoles>(std::move(newCurve),
                                                                                               std::move(holes));
            }
        }
        auto total_t = clock() - t_start;
        auto total_t_w = std::chrono::high_resolution_clock::now() - t_start_w;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        std::cout << "shape optimization finished in " << time_sec << " CPU seconds (" << std::chrono::duration<double>(total_t_w).count() << "s wall clock time" << std::endl;
    }
}

void Solver::globalAlignShapes() {
    double voxel_width = diameter_/settings_.voxel_resolution;
    double margin = voxel_width * 2;

    for (int dim=2; dim<3; ++dim) { //ONLY ALIGN VERTICAL NOW
        Vector3d up(0, 0, 0);
        up(dim) = 1;
        size_t N = num_vertices(construction.g);
        bool found=false;
        for (size_t i = 0; i < N; ++i) {
            Construction::Vertex v = vertex(i, construction.g);
            size_t partIdx1 = construction.g[v].partIdx;
            Vector3d n = construction.partData[partIdx1].normal();
            double dotprod = n.dot(up);
            if (std::abs(dotprod) > settings_.norm_parallel_threshold) {
                up = dotprod < 0 ? -n : n;
                found = true;
                break;
            }
        }
        if (!found) continue;
        std::cout << "up direction: " << up.transpose() << std::endl;
        std::vector<double> groups;
        for (size_t i = 0; i < N; ++i) {
            Construction::Vertex v = vertex(i, construction.g);
            size_t partIdx1 = construction.g[v].partIdx;
            Vector3d n = construction.partData[partIdx1].normal();
            if (std::abs(n.dot(up)) < settings_.norm_adjacency_threshold) {
                double partOffset = construction.partData[partIdx1].pos.dot(up);
                //temporarily remove offset from other groups
                for (auto &group : groups) {
                    group -= partOffset;
                }
                std::cout << "aligning part " << partIdx1 << std::endl;
                Vector2d up2d = construction.partData[partIdx1].projectDir(up);
                auto &shape = construction.getShape(partIdx1);
                CombinedCurve curve = shape.cutPath->curves();
                int numAligned = curve.align(margin, ANGLE_THRESHOLD, up2d, groups);
                std::cout << "aligned " << numAligned << std::endl;

                std::vector<std::shared_ptr<Primitive>> holes;
                for (const auto &child : shape.cutPath->children()) {
                    std::cout << "aligning part " << partIdx1 << " child" << std::endl;
                    CombinedCurve childCurve = child->curves();
                    numAligned = childCurve.align(margin, ANGLE_THRESHOLD, up2d, groups);
                    std::cout << "aligned " << numAligned << std::endl;
                    holes.emplace_back(new PolyCurveWithHoles(std::move(childCurve)));
                }
                shape.cutPath = std::make_shared<PolyCurveWithHoles>(std::move(curve), std::move(holes));
                //add back global offset to all groups
                for (auto &group : groups) {
                    group += partOffset;
                }
                std::cout << "alignment heights: " << groups << std::endl;
            }
        }
        std::cout << "final alignment heights: " << groups << std::endl;

    }
}

int Solver::removeDisconnectedParts() {
    int numRemoved = 0;
    bool found = true;
    while (found) {
        found = false;
        size_t N = num_vertices(construction.g);
        for (size_t i = 0; i < N; ++i) {
            Construction::Vertex v = vertex(i, construction.g);
            auto pair = out_edges(v, construction.g);
            if (pair.first == pair.second) {
                std::cout << "removing vertex " << i << " with no connections" << std::endl;
                remove_vertex(v, construction.g);
                ++numRemoved;
                found = true;
                break;
            }
        }
    }
    return numRemoved;
}



void Solver::regularizeKnots() {
    double margin = diameter_/settings_.voxel_resolution * 3;
    size_t N = num_vertices(construction.g);

    for (size_t i=0; i<N; ++i) {
        std::cout << "processing node " << i << std::endl;
        Construction::Vertex v = vertex(i, construction.g);
        size_t partIdx1 = construction.g[v].partIdx;
        auto &shape = construction.getShape(partIdx1);

        CombinedCurve curve = shape.cutPath->curves();

        std::cout << "removing coplanar" << std::endl;
        int numRemoved = curve.removeCoplanar(MAX_PARALLEL_ANGlE);
        std::cout << "removed " << numRemoved << " parallel lines from part " << partIdx1 << std::endl;

        std::cout << "fixing knots for part " << partIdx1 << std::endl;
        curve.fixKnots(settings_.min_knot_angle, margin);
        std::cout << "fixed knots" << std::endl;
        std::vector<std::shared_ptr<Primitive>> holes;
        for (const auto &child : shape.cutPath->children()) {
            CombinedCurve childCurve = child->curves();

            numRemoved = childCurve.removeCoplanar(MAX_PARALLEL_ANGlE);
            std::cout << "removed " << numRemoved << " parallel lines from part " << partIdx1 << std::endl;

            std::cout << "fixing child knots" << std::endl;
            childCurve.fixKnots(settings_.min_knot_angle, margin);
            std::cout << "fixed knots" << std::endl;
            holes.emplace_back(new PolyCurveWithHoles(std::move(childCurve)));
        }
        std::cout << "setting new shape" << std::endl;
        shape.cutPath = std::make_shared<PolyCurveWithHoles>(std::move(curve), std::move(holes));
        std::cout << "set new shape" << std::endl;
    }
}

const Construction &Solver::getConstruction() const {
    return construction;
}

Construction &Solver::getConstruction() {
    return construction;
}