//
// Created by James Noeckel on 11/2/20.
//

#include "reconstruction/ReconstructionData.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " input outpath" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outPath = argv[2];
    ReconstructionData reconstruction;
    std::cout << "loading reconstruction " << inputPath << ":" << std::endl;
    if (inputPath.rfind(".out") != std::string::npos) {
        if (!reconstruction.load_bundler_file(inputPath, "")) {
            std::cerr << "failed to load bundler file " << inputPath << std::endl;
            return 1;
        }
    } else {
        if (!reconstruction.load_colmap_reconstruction(inputPath, "",
                                                        "")) {
            std::cerr << "failed to load reconstruction in path " << inputPath << std::endl;
            return 1;
        }
    }
    std::cout << "exporting" << std::endl;
    reconstruction.export_colmap_reconstruction(outPath);
    return 0;
}