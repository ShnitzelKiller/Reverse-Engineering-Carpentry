//
// Created by James Noeckel on 3/26/20.
//

#include <iostream>
#include "geometry/mesh_interior_point_labeling.h"
#include "reconstruction/mesh_io.h"

int main(int argc, char **argv) {
    Eigen::MatrixX3d V;
    Eigen::MatrixX3i F;
    load_mesh("../test_data/cube.obj", V, F);
    std::cout << "vertices: " << std::endl << V << std::endl;
    std::cout << "faces: " << std::endl << F << std::endl;
    if (V.rows() != 8 || F.rows() != 12) {
        std::cerr << "expected cube" << std::endl;
        return 1;
    }
    Eigen::MatrixX3d samples(1000, 3);
    for (int i=0; i<10; i++) {
        for (int j=0; j<10; j++) {
            for (int k=0; k<10; k++) {
                Eigen::RowVector3d pt(-1.5 + i/3.0, -1.5 + j/3.0, -1.5 + k/3.0);
                pt.array() += 0.00000143;
                samples.row(i*100 + j*10 + k) = pt;
            }
        }
    }
    std::vector<bool> labels = mesh_interior_point_labeling(samples, V, F);
    if (labels.size() != 1000) {
        std::cerr << "labeling has the wrong size: " << labels.size() << std::endl;
        return 1;
    }
    int correct = 0;
    for (int i=0; i<1000; i++) {
        bool inside = labels[i];
        Eigen::RowVector3d pt = samples.row(i);
        bool really_inside = pt.x() > -1.0 && pt.x() < 1.0 && pt.y() > -1.0 && pt.y() < 1.0 && pt.z() > -1.0 && pt.z() < 1.0;
        if (inside != really_inside) {
            std::cerr << "wrong assignment for point " << i << '(' << pt << "): " << "computed " << inside << " but is really " << really_inside << " with num hits=" << labels[i] << std::endl;
        } else {
            correct++;
        }
    }
    std::cout << "correct: " << correct << "/1000" << std::endl;
    return 0;
}