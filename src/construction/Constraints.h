//
// Created by James Noeckel on 1/16/21.
//

#pragma once
#include "utils/typedefs.hpp"
struct ShapeConstraint {
    Edge2d edge;
    bool convex;
    /** whether this constraint comes from the backside of the other part */
    bool otherOpposing = false;
    /** whether this constraint comes from the backside of this part */
    bool opposing = false;
    bool inside=false;
    bool outside=true;
};
struct ConnectionConstraint {
    Edge2d edge;
//    double gap;
    double margin = 0.0;
    bool inside=true;
    bool outside=true;
    bool useInCurveFit=true;
//    double startGuideExtention=0.0;
//    double endGuideExtension=0.0;
//    Edge2d getGuide() const;
//    Construction::Edge connection;
//    int contactID;
};

struct Guide {
    Edge2d edge;
};