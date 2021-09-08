//
// Created by James Noeckel on 1/16/21.
//

#include "Constraints.h"

Edge2d ConnectionConstraint::getGuide() const {
    Eigen::Vector2d dir = (edge.second - edge.first).normalized();
    return {edge.first - dir * startGuideExtention, edge.second + dir * endGuideExtension};
}
