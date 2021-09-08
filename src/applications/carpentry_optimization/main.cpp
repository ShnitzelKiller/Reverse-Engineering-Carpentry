//
// Created by James Noeckel on 7/8/20.
//

#include "utils/settings.h"
#include <iostream>
#include "reconstruction/point_cloud_io.h"
#include "reconstruction/ReconstructionData.h"

#include "construction/Solver.h"
#include "utils/timingMacros.h"

using namespace Eigen;

void evaluate(Solver &solver, const Settings &settings) {
    std::cout << "computing mean distance from model" << std::endl;
    auto allDistances = solver.getConstruction().allDistances(solver.cloud_, 1);
    VectorXd minDistances = allDistances.rowwise().minCoeff();
    double meanDistance = minDistances.mean();
    std::cout << "mean squared error: " << meanDistance / (solver.getDiameter() * solver.getDiameter()) << std::endl;
    std::cout << solver.cloud_->P.rows() << " points" << std::endl;
}

bool selectionStage(Solver &solver, const Settings &settings);
bool assemblyStage(Solver &solver, const Settings &settings);
bool segmentationStage(Solver &solver, const Settings &settings);
bool alignmentAndCurveFittingStage(Solver &solver, const Settings &settings);
bool postProcessingStage(Solver &solver, const Settings &settings);

bool selectionStage(Solver &solver, const Settings &settings) {
    DECLARE_TIMING(selection);
    START_TIMING(selection);
    std::cout << "INITIAL SELECTION STAGE" << std::endl;
    solver.initialize();
    if (!solver.getConstruction().exportMesh(settings.result_path + "_allparts.obj")) {
        std::cout << "failed to save decomposition" << std::endl;
    }
//    std::vector<bool> testW(solver.construction.partData.size(), true);
//    solver.construction.setW(testW);
    if (settings.visualize) {
        std::cout << "visualizing all parts: " << std::endl;
        solver.visualize();
    }
    std::cout << "pruning opposing parts:" << std::endl;
    solver.pruneOpposing();
    if (settings.visualize) {
        solver.visualize();
    }
    std::cout << "optimizing part decomposition: " << std::endl;
    if (!solver.optimizeW(2)) {
        std::cout << "optimization failed" << std::endl;
        return false;
    }
    STOP_TIMING(selection);
    std::cout << "SELECTION TIME: " << std::endl;
    PRINT_TIMING(selection);
    if (!solver.getConstruction().exportPlaintext(settings.result_path + "_selection_checkpoint.txt", SELECTION)) {
        std::cout << "failed to export plaintext " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportPlaintext("selection_checkpoint.txt", SELECTION)) {
            std::cout << "failed backup save" << std::endl;
        }
    }
    if (!solver.getConstruction().exportMesh(settings.result_path + "_selection.obj")) {
        std::cout << "failed to save decomposition" << std::endl;
    }
    if (settings.visualize) {
        solver.visualize();
    }

    solver.computeVisibility();
    std::cout << "visibility groups: " << std::endl;
    for (size_t c=0; c<solver.cluster_visibility.size(); c++) {
        if (c < solver.construction.partData.size())
            std::cout << "views of cluster " << c << ": ";
        else
            std::cout << "views of cluster " << c - solver.construction.partData.size() << " backface: ";
        for (auto v : solver.cluster_visibility[c]) {
            std::cout << "im " << v.first << " (" << v.second << "), ";
        }
        std::cout << std::endl;
    }

    return assemblyStage(solver, settings);
}

bool assemblyStage(Solver &solver, const Settings &settings) {
    DECLARE_TIMING(assembly);
    START_TIMING(assembly);
    std::cout << "ASSEMBLY STAGE" << std::endl;
    std::cout << "building graph: ";
    int edgeCount = solver.buildGraph();
    std::cout << edgeCount << " initial connections" << std::endl;
    std::cout << "initializing connections" << std::endl;
    solver.initializeConnectionTypes();
    std::cout << "initializing connection contacts" << std::endl;
    solver.refineConnectionContacts();
    if (settings.visualize) {
        solver.visualize();
    }
    std::cout << "optimizing connections" << std::endl;
    solver.optimizeConnections(true);
    STOP_TIMING(assembly);
    std::cout << "ASSEMBLY TIME: " << std::endl;
    PRINT_TIMING(assembly);
    if (settings.visualize) {
        solver.visualize();
    }

    std::cout << "refining connection contacts" << std::endl;
    solver.refineConnectionContacts();
    std::cout << "finding initial connection constraints" << std::endl;
    solver.recomputeConnectionConstraints();

    if (!solver.getConstruction().exportPlaintext(settings.result_path + "_connection_checkpoint.txt", ASSEMBLY)) {
        std::cout << "failed to export plaintext " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportPlaintext("connection_checkpoint.txt", ASSEMBLY)) {
            std::cout << "failed backup save" << std::endl;
        }
    }

    if (settings.visualize) {
        solver.visualize();
    }
    return segmentationStage(solver, settings);
}


bool segmentationStage(Solver &solver, const Settings &settings) {
    std::cout << "IMAGE ANALYSIS STAGE" << std::endl;

    DECLARE_TIMING(segmentation);
    START_TIMING(segmentation);
//    solver.getConstruction().rotateParts();
//    if (settings.visualize) {
//        solver.visualize();
//    }

    int newParts = solver.shapeFromImages(true);
    if (newParts > 0) {
        std::cout << "recomputing visibility to account for " << newParts << " new parts" << std::endl;
        solver.computeVisibility();
    }
    std::cout << "recomputing meshes" << std::endl;
    solver.recomputeMeshes();
    int numIters = 1;
    int numNewConnections;
    std::cout << "checking for new connections" << std::endl;
    while ((numNewConnections = solver.findNewConnections()) > 0) {
        std::cout << "found " << numNewConnections << " new connections; re-running refinement; iteration " << numIters;
        std::cout << "optimizing connections" << std::endl;
        solver.initializeConnectionTypes();
        solver.refineConnectionContacts();
        solver.optimizeConnections(true);
        solver.recomputeConnectionConstraints();
        solver.refineConnectionContacts();
        if (settings.visualize) {
            solver.visualize();
        }
        std::cout << "analyzing images (iteration " << numIters << ')' << std::endl;
        newParts = solver.shapeFromImages(true);
        if (newParts > 0) {
            std::cout << "recomputing visibility (iteration " << numIters << ')' << std::endl;
            solver.computeVisibility();
        }
        std::cout << "recomputing meshes (iteration " << numIters << ')' << std::endl;
        solver.recomputeMeshes();
        if (settings.visualize) {
            solver.visualize();
        }
        ++numIters;
    }
    STOP_TIMING(segmentation);
    std::cout << "SEGMENTATION TIME: " << std::endl;
    PRINT_TIMING(segmentation);
    std::cout << "refining connection contacts after potential part splits" << std::endl;
    solver.refineConnectionContacts();

    if (!solver.getConstruction().exportMesh(settings.result_path + "_dense.obj")) {
        std::cout << "failed to save mesh " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportMesh("denseMesh.obj")) {
            std::cout << "failed backup save" << std::endl;
        }
    }
    if (!solver.getConstruction().exportModel(settings.result_path + "_dense.xml")) {
        std::cout << "failed to save solution " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportModel("denseSolution.xml")) {
            std::cout << "failed backup save" << std::endl;
        }
    }
    if (!solver.getConstruction().exportPlaintext(settings.result_path + "_segmentation_checkpoint.txt", SEGMENTATION)) {
        std::cout << "failed to export plaintext " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportPlaintext("segmentation_checkpoint.txt", SEGMENTATION)) {
            std::cout << "failed backup save" << std::endl;
        }
    }

    if (settings.visualize) {
        solver.visualize();
    }

    return alignmentAndCurveFittingStage(solver, settings);
}

bool alignmentAndCurveFittingStage(Solver &solver, const Settings &settings) {
    std::cout << "CURVE FITTING STAGE" << std::endl;
    DECLARE_TIMING(curvefit);
    START_TIMING(curvefit);
    std::cout << "realigning parts: " << std::endl;
    solver.realign();
//    solver.recomputeMeshes();
//    if (settings.visualize) {
//        solver.visualize();
//    }
    std::cout << "regularizing thickness" << std::endl;
    int numClusters = solver.regularizeDepths();
    std::cout << "reduced to " << numClusters << " thicknesses" << std::endl;
    solver.recomputeMeshes(false);
    std::cout << "refining connection constraints" << std::endl;
    //recompute connection contacts since we realigned parts
    solver.refineConnectionContacts();
    solver.recomputeConnectionConstraints();


    if (settings.visualize) {
        solver.visualize();
    }

    solver.optimizeShapes(true);
    STOP_TIMING(curvefit);
    std::cout << "CURVE FIT TIME: " << std::endl;
    PRINT_TIMING(curvefit);
//    construction.recenter();
//    construction.scale(settings.final_scale / construction.stockData.back().thickness);
    solver.getConstruction().computeMeshes(false);

    if (!solver.getConstruction().exportMesh(settings.result_path + ".obj")) {
        std::cout << "failed to save mesh " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportMesh("finalMesh.obj")) {
            std::cout << "failed backup save" << std::endl;
        }
    }
    if (!solver.getConstruction().exportModel(settings.result_path + ".xml")) {
        std::cout << "failed to save solution " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportModel("finalSolution.xml")) {
            std::cout << "failed backup save" << std::endl;
        }
    }
    if (!solver.getConstruction().exportPlaintext(settings.result_path + ".txt", SEGMENTATION)) {
        std::cout << "failed to export plaintext " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportPlaintext("finalSolution.txt", SEGMENTATION)) {
            std::cout << "failed backup save" << std::endl;
        }
    }

    if (settings.visualize) {
        solver.visualize();
    }
    return postProcessingStage(solver, settings);
}

bool postProcessingStage(Solver &solver, const Settings &settings) {

    DECLARE_TIMING(post);
    START_TIMING(post);
    int numRemoved = solver.removeDisconnectedParts();

    if (numRemoved > 0) {
        std::cout << "removed " << numRemoved << " parts with no connections" << std::endl;
        if (settings.visualize) {
            solver.recomputeMeshes();
            solver.visualize();
        }
    }

    solver.globalAlignShapes();

    if (settings.visualize) {
        solver.recomputeMeshes();
        solver.visualize();
    }

    solver.regularizeKnots();
    std::cout << "saving shapes" << std::endl;
    solver.getConstruction().saveShapes();
    std::cout << "recomputing meshes" << std::endl;
    STOP_TIMING(post);
    std::cout << "POSTPROCESSING TIME: " << std::endl;
    PRINT_TIMING(post);
    solver.recomputeMeshes();


    if (!solver.getConstruction().exportPlaintext(settings.result_path + "_postProcessed.txt", SEGMENTATION)) {
        std::cout << "failed to export plaintext " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportPlaintext("postProcessedSolution.txt", SEGMENTATION)) {
            std::cout << "failed backup save" << std::endl;
        }
    }

    if (!solver.getConstruction().exportMesh(settings.result_path + "_postProcessed.obj", settings.connector_mesh, settings.connector_spacing, settings.connector_scale)) {
        std::cout << "failed to save mesh " << settings.result_path << std::endl;
        if (!solver.getConstruction().exportMesh("postProcessed.obj")) {
            std::cout << "failed backup save" << std::endl;
        }
    }


    if (settings.visualize) {
        solver.visualize();
    }
    return true;
}

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

    std::srand(settings.random_seed);

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

    reconstruction->export_rhino_camera("customstool_cameras.txt");

    PointCloud3::Handle cloud;
    std::cout << "loading point cloud..." << std::endl;
    auto start_t = clock();
    if (!load_pointcloud(settings.points_filename, cloud)) {
        return 1;
    }

    auto total_t = clock() - start_t;
    float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
    std::cout << "loaded " << cloud->P.rows() << " points in " << time_sec << " seconds" << std::endl;
    //optimization
    std::mt19937 randomEngine(settings.random_seed);
    Solver solver(settings, randomEngine);
    solver.setDataPoints(cloud);
    solver.computeBounds();
    solver.setReconstructionData(reconstruction);

    std::cout << "solver diameter: " << solver.getDiameter() << std::endl;

    if (!settings.curvefit_checkpoint.empty() && solver.getConstruction().loadPlaintext(settings.curvefit_checkpoint, SEGMENTATION)) {
        solver.recomputeMeshes();
        solver.computeVisibility();
        solver.getConstruction().exportModel(settings.result_path + "_reloaded.xml");
//        evaluate(solver, settings);
        solver.refineConnectionContacts(true);
        solver.recomputeConnectionConstraints();
        if (settings.visualize) {
            solver.visualize();
        }
        postProcessingStage(solver, settings);
    } else if (!settings.oldresult_checkpoint.empty() && solver.getConstruction().loadPlaintext(settings.oldresult_checkpoint, BASICPLUS)) {
        solver.recomputeMeshes();
        evaluate(solver, settings);
        solver.computeVisibility();
        if (settings.visualize) {
            solver.visualize();
        }
        postProcessingStage(solver, settings);
    } else if (!settings.segmentation_checkpoint.empty() && solver.getConstruction().loadPlaintext(settings.segmentation_checkpoint, SEGMENTATION)) {
        solver.recomputeMeshes();
        solver.computeVisibility();
        solver.refineConnectionContacts();
        solver.recomputeConnectionConstraints();
        if (settings.visualize) {
            solver.visualize();
        }
        alignmentAndCurveFittingStage(solver, settings);
    } else if (!settings.connection_checkpoint.empty() && solver.getConstruction().loadPlaintext(settings.connection_checkpoint, ASSEMBLY)) {
        solver.recomputeMeshes();
//        if (settings.visualize) {
//            solver.visualize();
//        }
        solver.computeVisibility();
        solver.refineConnectionContacts();
        solver.recomputeConnectionConstraints();
        if (settings.visualize) {
            solver.visualize();
        }
        segmentationStage(solver, settings);
    } else if (!settings.selection_checkpoint.empty() && solver.getConstruction().loadPlaintext(settings.selection_checkpoint, SELECTION)) {
        solver.recomputeMeshes();
        if (settings.visualize) {
            solver.visualize();
        }
        solver.computeVisibility();
        assemblyStage(solver, settings);
    } else {
        selectionStage(solver, settings);
    }
    return 0;
}