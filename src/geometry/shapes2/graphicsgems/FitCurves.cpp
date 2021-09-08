/*
An Algorithm for Automatically Fitting Digitized Curves
by Philip J. Schneider
from "Graphics Gems", Academic Press, 1990

 Adapted to C++ by James Noeckel on 4/15/2020
*/

/*	Piecewise cubic fitting code	*/

#include "FitCurves.h"
#include <vector>

static void FitCubic(const Eigen::Ref<Eigen::MatrixX2d> &d, int first, int last, const Eigen::Vector2d &tHat1,
                     const Eigen::Vector2d &tHat2, double error, std::vector<Eigen::Matrix<double, 4, 2>> &curves);

static double
NewtonRaphsonRootFind(const Eigen::Matrix<double, -1, 2> &Q, const Eigen::Ref<const Eigen::Vector2d> &P, double u);

static double B0(double u), B1(double u), B2(double u), B3(double u);

static Eigen::Vector2d ComputeLeftTangent(const Eigen::Ref<const Eigen::MatrixX2d> &d, int end);

static Eigen::Vector2d ComputeRightTangent(const Eigen::Ref<const Eigen::MatrixX2d> &d, int end);

static Eigen::Vector2d ComputeCenterTangent(const Eigen::Ref<const Eigen::MatrixX2d> &d, int center);

void FitCurve(const Eigen::Ref<Eigen::MatrixX2d> &d, double error, std::vector<Eigen::Matrix<double, 4, 2>> &curves) {
    Eigen::Vector2d tHat1, tHat2;    /*  Unit tangent vectors at endpoints */

    tHat1 = ComputeLeftTangent(d, 0);
    tHat2 = ComputeRightTangent(d, d.rows() - 1);
    FitCubic(d, 0, d.rows() - 1, tHat1, tHat2, error, curves);
}

Eigen::MatrixX2d sampleCurve(const Eigen::Matrix<double, 4, 2> &bezierCurve, int numPts) {
    Eigen::MatrixX2d pts(numPts, 2);
    for (int i=0; i<numPts; i++) {
        pts.row(i) = BezierII(3, bezierCurve, static_cast<double>(i)/numPts).transpose();
    }
    return pts;
}

Eigen::MatrixX2d sampleCurves(const std::vector<Eigen::Matrix<double, 4, 2>> &bezierCurve, int subdivisions) {
    Eigen::MatrixX2d pts(subdivisions * bezierCurve.size(), 2);
    for (int i=0; i<bezierCurve.size(); i++) {
        pts.block(subdivisions * i, 0, subdivisions, 2) = sampleCurve(bezierCurve[i], subdivisions);
    }
    return pts;
}


/*
 *  FitCubic :
 *  	Fit a Bezier curve to a (sub)set of digitized points
 */
static void FitCubic(const Eigen::Ref<Eigen::MatrixX2d> &d, int first, int last, const Eigen::Vector2d &tHat1,
                     const Eigen::Vector2d &tHat2, double error, std::vector<Eigen::Matrix<double, 4, 2>> &curves) {
    int N = d.rows();
    Eigen::Matrix<double, 4, 2> bezCurve; /*Control points of fitted Bezier curve*/
    std::vector<double> u;        /*  Parameter values for point  */
    double maxError;    /*  Maximum fitting error	 */
    int splitPoint;    /*  Point to split point set at	 */
    int nPts;        /*  Number of points in subset  */
    double iterationError; /*Error below which you try iterating  */
    int maxIterations = 4; /*  Max times to try iterating  */
    Eigen::Vector2d tHatCenter;    /* Unit tangent vector at splitPoint */
    int i;

    iterationError = error * 4.0;    /* fixed issue 23 */
    nPts = last - first + 1;
    int first_w = first % N;
    int last_w = last % N;

    /*  Use heuristic if region only has two points in it */
    if (nPts == 2) {
        double dist = (d.row(last_w) - d.row(first_w)).norm() / 3.0;
        bezCurve.row(0) = d.row(first_w);
        bezCurve.row(3) = d.row(last_w);
        bezCurve.row(1) = bezCurve.row(0) + tHat1.normalized().transpose() * dist;
        bezCurve.row(2) = bezCurve.row(3) + tHat2.normalized().transpose() * dist;
        curves.push_back(bezCurve);
        return;
    }

    /*  Parameterize points, and attempt to fit curve */
    u = ChordLengthParameterize(d, first, last);
    bezCurve = GenerateBezier(d, first, last, u, tHat1, tHat2);

    /*  Find max deviation of points to fitted curve */
    maxError = ComputeMaxError(d, first, last, bezCurve, u, splitPoint);
    if (maxError < error) {
        curves.push_back(bezCurve);
        return;
    }


    /*  If error not too large, try some reparameterization  */
    /*  and iteration */
    if (maxError < iterationError) {
        for (i = 0; i < maxIterations; i++) {
            std::vector<double> uPrime = Reparameterize(d, first, last, u, bezCurve); /*  Improved parameter values */
            bezCurve = GenerateBezier(d, first, last, uPrime, tHat1, tHat2);
            maxError = ComputeMaxError(d, first, last,
                                       bezCurve, uPrime, splitPoint);
            if (maxError < error) {
                curves.push_back(bezCurve);
                return;
            }
            u = std::move(uPrime);
        }
    }

    /* Fitting failed -- split at max error point and fit recursively */
    tHatCenter = ComputeCenterTangent(d, splitPoint);
    FitCubic(d, first, splitPoint, tHat1, tHatCenter, error, curves);
    FitCubic(d, splitPoint, last, -tHatCenter, tHat2, error, curves);
}

Eigen::Matrix<double, -1, 2> FitBezier(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last,
                                       const std::vector<double> &uPrime, int degree, const Eigen::Vector2d &leftTangent, const Eigen::Vector2d &rightTangent) {
    int N = d.rows();
    int nPts = last - first + 1;

    if (degree==3) {
        const Eigen::Matrix4d bezierMatrix = (Eigen::Matrix4d()
                << -1, 3, -3, 1,
                3, -6, 3, 0,
                -3, 3, 0, 0,
                1, 0, 0, 0).finished();
        Eigen::MatrixX4d T(nPts - 2, 4);
        for (int i = 0; i < nPts - 2; i++) {
            double t = uPrime[i + 1];
            double t2 = t * t;
            double t3 = t2 * t;
            T.row(i) = Eigen::RowVector4d(t3, t2, t, 1);
        }
        Eigen::MatrixX2d b(nPts - 2, 2);
        for (int i = 0; i < nPts - 2; i++) {
            b.row(i) = d.row((i + first + 1) % N);
        }
        bool leftConstrained = leftTangent.squaredNorm() > 0;
        bool rightConstrained = rightTangent.squaredNorm() > 0;
        Eigen::Matrix<double, 4, 2> bezierPoints;
        bezierPoints.row(0) = d.row(first % N);
        bezierPoints.row(3) = d.row(last % N);
        Eigen::MatrixX4d B = T * bezierMatrix;
        if (!leftConstrained && !rightConstrained) {
            b -= (B.col(0) * d.row(first % N) + B.col(3) * d.row(last % N));
            Eigen::MatrixX2d A = B.block(0, 1, nPts-2, 2);
            Eigen::Matrix2d ATA = A.transpose() * A;
            bezierPoints.block<2, 2>(1, 0) = ATA.colPivHouseholderQr().solve(A.transpose() * b);
            return bezierPoints;
        } else if ((leftConstrained && !rightConstrained) || (!leftConstrained && rightConstrained)) {
            //Eigen::Vector4d basis0 = bezierMatrix.col(0);
            //Eigen::Vector4d basis3 = bezierMatrix.col(3);
            int alphaInd = leftConstrained ? 1 : 2;
            int freeInd = leftConstrained ? 2 : 1;
            Eigen::VectorXd B0 = B.col(0);
            Eigen::VectorXd B3 = B.col(3);
            if (leftConstrained) B0 += B.col(alphaInd);
            else if (rightConstrained) B3 += B.col(alphaInd);
            b -= (B0 * d.row(first % N) + B3 * d.row(last % N));
            const Eigen::Vector2d &tangent = leftConstrained ? leftTangent : rightTangent;
            Eigen::MatrixX3d A((nPts-2) * 2, 3);
            A.block(0, 0, nPts-2, 1) = tangent.x() * B.col(alphaInd);
            A.block(nPts-2, 0, nPts-2, 1) = tangent.y() * B.col(alphaInd);
            A.block(0, 1, nPts-2, 1) = B.col(freeInd);
            A.block(nPts-2, 1, nPts-2, 1) = Eigen::VectorXd::Zero(nPts-2);
            A.block(0, 2, nPts-2, 1) = Eigen::VectorXd::Zero(nPts-2);
            A.block(nPts-2, 2, nPts-2, 1) = B.col(freeInd);
            Eigen::VectorXd b2((nPts-2)*2);
            b2.head(nPts-2) = b.col(0);
            b2.tail(nPts-2) = b.col(1);
            Eigen::Matrix3d ATA = A.transpose() * A;
            Eigen::Vector3d sol = ATA.colPivHouseholderQr().solve(A.transpose() * b2);
            if (leftConstrained) {
                bezierPoints.row(alphaInd) = tangent.transpose() * sol(0) + bezierPoints.row(0);
                bezierPoints.row(freeInd) = sol.tail(2).transpose();
            } else {
                bezierPoints.row(alphaInd) = tangent.transpose() * sol(0) + bezierPoints.row(3);
                bezierPoints.row(freeInd) = sol.tail(2).transpose();
            }
            return bezierPoints;
        } else {
            return GenerateBezier(d, first, last, uPrime, leftTangent, rightTangent);
        }

    } else if (degree==2) {
        Eigen::Matrix3d bezierMatrix;
        bezierMatrix << 1, -2, 1,
                        -2, 2, 0,
                        1, 0, 0;
        Eigen::RowVector3d T(uPrime[1] * uPrime[1], uPrime[1], 1);
        Eigen::RowVector2d b = d.row((first + 1) % N);
        b -= (T * (bezierMatrix.col(0) * d.row(first % N) + bezierMatrix.col(2) * d.row(last % N)));
        double A = T * bezierMatrix.col(1);
        Eigen::Matrix<double, 3, 2> bezierPoints;
        bezierPoints.row(0) = d.row(first % N);
        bezierPoints.row(2) = d.row(last % N);
        bezierPoints.row(1) = b.array()/A;
        return bezierPoints;
    } else {
        return Eigen::MatrixX2d(0, 2);
    }
}

Eigen::Matrix<double, 4, 2>
GenerateBezier(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last, const std::vector<double> &uPrime,
               const Eigen::Vector2d &tHat1, const Eigen::Vector2d &tHat2) {
    int N = d.rows();
    int i;

    //Vector2 	A[MAXPOINTS][2];
    int nPts = last - first + 1; /* Number of pts in sub-curve */

    std::vector<std::vector<Eigen::Vector2d>> A(nPts, std::vector<Eigen::Vector2d>(2)); /* Precomputed rhs for eqn	*/
    double C[2][2];            /* Matrix C		*/
    double X[2];            /* Matrix X			*/
    double det_C0_C1,        /* Determinants of matrices	*/
    det_C0_X,
            det_X_C1;
    double alpha_l,        /* Alpha values, left and right	*/
    alpha_r;
    Eigen::Vector2d tmp;            /* Utility variable		*/
    Eigen::Matrix<double, 4, 2> bezCurve;    /* RETURN bezier curve ctl pts	*/
    double segLength;
    double epsilon;

    int first_w = first % N;
    int last_w = last % N;

    /* Compute the A's	*/
    for (i = 0; i < nPts; i++) {
        Eigen::Vector2d v1, v2;
        v1 = tHat1;
        v2 = tHat2;
        v1 = v1.normalized() * B1(uPrime[i]);
        v2 = v2.normalized() * B2(uPrime[i]);
        A[i][0] = v1;
        A[i][1] = v2;
    }

    /* Create the C and X matrices	*/
    C[0][0] = 0.0;
    C[0][1] = 0.0;
    C[1][0] = 0.0;
    C[1][1] = 0.0;
    X[0] = 0.0;
    X[1] = 0.0;

    for (i = 0; i < nPts; i++) {
        C[0][0] += A[i][0].dot(A[i][0]);
        C[0][1] += A[i][0].dot(A[i][1]);
/*					C[1][0] += V2Dot(&A[i][0], &A[i][1]);*/
        C[1][0] = C[0][1];
        C[1][1] += A[i][1].dot(A[i][1]);

        tmp = (d.row((first + i) % N) -

               (d.row(first_w) * B0(uPrime[i])
                + d.row(first_w) * B1(uPrime[i])
                + d.row(last_w) * B2(uPrime[i])
                + d.row(last_w) * B3(uPrime[i]))).transpose();

        X[0] += A[i][0].dot(tmp);
        X[1] += A[i][1].dot(tmp);
    }

    /* Compute the determinants of C and X	*/
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1];
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0];
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1];

    /* Finally, derive alpha values	*/
    alpha_l = (det_C0_C1 == 0) ? 0.0 : det_X_C1 / det_C0_C1;
    alpha_r = (det_C0_C1 == 0) ? 0.0 : det_C0_X / det_C0_C1;

    /* If alpha negative, use the Wu/Barsky heuristic (see text) */
    /* (if alpha is 0, you get coincident control points that lead to
     * divide by zero in any subsequent NewtonRaphsonRootFind() call. */
    segLength = (d.row(last_w) - d.row(first_w)).norm();
    epsilon = 1.0e-6 * segLength;
    if (alpha_l < epsilon || alpha_r < epsilon) {
        // fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        double dist = segLength / 3.0;
        bezCurve.row(0) = d.row(first_w);
        bezCurve.row(3) = d.row(last_w);
        bezCurve.row(1) = tHat1.normalized().transpose() * dist + bezCurve.row(0);
        bezCurve.row(2) = tHat2.normalized().transpose() * dist + bezCurve.row(3);
        return (bezCurve);
    }

    /*  First and last control points of the Bezier curve are */
    /*  positioned exactly at the first and last data points */
    /*  Control points 1 and 2 are positioned an alpha distance out */
    /*  on the tangent vectors, left and right, respectively */
    bezCurve.row(0) = d.row(first_w);
    bezCurve.row(3) = d.row(last_w);
    bezCurve.row(1) = tHat1.normalized().transpose() * alpha_l + bezCurve.row(0);
    bezCurve.row(2) = tHat2.normalized().transpose() * alpha_r + bezCurve.row(3);
    return (bezCurve);
}

std::vector<double>
Reparameterize(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last, const std::vector<double> &u,
               const Eigen::Matrix<double, -1, 2> &bezCurve) {
    int N = d.rows();
    int nPts = last - first + 1;
    int i;
    std::vector<double> uPrime(nPts);        /*  New parameter values	*/

    for (i = first; i <= last; i++) {
        uPrime[i - first] = NewtonRaphsonRootFind(bezCurve, d.row(i % N).transpose(), u[i - first]);
    }
    return (uPrime);
}

/*
 *  NewtonRaphsonRootFind :
 *	Use Newton-Raphson iteration to find better root.
 */
static double
NewtonRaphsonRootFind(const Eigen::Matrix<double, -1, 2> &Q, const Eigen::Ref<const Eigen::Vector2d> &P, double u) {
    Eigen::Vector2d Q_u, Q1_u, Q2_u; /*u evaluated at Q, Q', & Q''	*/
    double uPrime;        /*  Improved u			*/
    int i;
    if (Q.rows() == 4) {
        Eigen::Matrix<double, 3, 2> Q1;
        Eigen::Matrix<double, 2, 2> Q2;
        /* Compute Q(u)	*/
        Q_u = BezierII(3, Q, u);

        /* Generate control vertices for Q'	*/
        for (i = 0; i <= 2; i++) {
            Q1.row(i).x() = (Q.row(i + 1).x() - Q.row(i).x()) * 3.0;
            Q1.row(i).y() = (Q.row(i + 1).y() - Q.row(i).y()) * 3.0;
        }

        /* Generate control vertices for Q'' */
        for (i = 0; i <= 1; i++) {
            Q2.row(i).x() = (Q1.row(i + 1).x() - Q1.row(i).x()) * 2.0;
            Q2.row(i).y() = (Q1.row(i + 1).y() - Q1.row(i).y()) * 2.0;
        }

        /* Compute Q'(u) and Q''(u)	*/
        Q1_u = BezierII(2, Q1, u);
        Q2_u = BezierII(1, Q2, u);
    } else if (Q.rows() == 3) {
        Q_u = BezierII(2, Q, u);

        /* Compute Q'(u) and Q''(u)	*/
        Q1_u = ((2*u-2) * Q.row(0) + (2-4*u) * Q.row(1) + 2 * u * Q.row(2)).transpose();
        Q2_u = (2*Q.row(0) - 4*Q.row(1) + 2*Q.row(2)).transpose();
    } else {
        assert(false);
    }

    /* Compute f(u)/f'(u) */
    double numerator = (Q_u.x() - P.x()) * (Q1_u.x()) + (Q_u.y() - P.y()) * (Q1_u.y());
    double denominator = (Q1_u.x()) * (Q1_u.x()) + (Q1_u.y()) * (Q1_u.y()) +
                  (Q_u.x() - P.x()) * (Q2_u.x()) + (Q_u.y() - P.y()) * (Q2_u.y());
    if (denominator == 0.0f) return u;

    /* u = u - f(u)/f'(u) */
    uPrime = u - (numerator / denominator);
    return (uPrime);
}

Eigen::VectorXd BezierII(int degree, const Eigen::Ref<const Eigen::MatrixXd> &V, double t) {
    int i, j;
    Eigen::MatrixXd Vtemp = V;

    /* Triangle computation	*/
    for (i = 1; i <= degree; i++) {
        for (j = 0; j <= degree - i; j++) {
            Vtemp.row(j) = (1.0 - t) * Vtemp.row(j) + t * Vtemp.row(j + 1);
        }
    }

    return Vtemp.row(0).transpose();
}


/*
 *  B0, B1, B2, B3 :
 *	Bezier multipliers
 */
static double B0(double u) {
    double tmp = 1.0 - u;
    return (tmp * tmp * tmp);
}


static double B1(double u) {
    double tmp = 1.0 - u;
    return (3 * u * (tmp * tmp));
}

static double B2(double u) {
    double tmp = 1.0 - u;
    return (3 * u * u * tmp);
}

static double B3(double u) {
    return (u * u * u);
}


/*
 * ComputeLeftTangent, ComputeRightTangent, ComputeCenterTangent :
 *Approximate unit tangents at endpoints and "center" of digitized curve
 */
static Eigen::Vector2d ComputeLeftTangent(const Eigen::Ref<const Eigen::MatrixX2d> &d, int end) {
    int N = d.rows();
    Eigen::Vector2d tHat1 = (d.row((end + 1) % N) - d.row(end % N)).transpose();
    return tHat1.normalized();
}

static Eigen::Vector2d ComputeRightTangent(const Eigen::Ref<const Eigen::MatrixX2d> &d, int end) {
    int N = d.rows();
    Eigen::Vector2d tHat2 = (d.row((end - 1) % N) - d.row(end % N)).transpose();
    return tHat2.normalized();
}


static Eigen::Vector2d ComputeCenterTangent(const Eigen::Ref<const Eigen::MatrixX2d> &d, int center) {
    int N = d.rows();
    Eigen::Vector2d V1 = (d.row(center > 0 ? (center - 1) % N : center - 1 + N) - d.row(center % N)).transpose();
    Eigen::Vector2d V2 = (d.row(center % N) - d.row((center + 1) % N)).transpose();
    Eigen::Vector2d tHatCenter;
    tHatCenter.x() = (V1.x() + V2.x()) / 2.0;
    tHatCenter.y() = (V1.y() + V2.y()) / 2.0;
    return tHatCenter.normalized();
}

std::vector<double> ChordLengthParameterize(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last) {
    int N = d.rows();
    int i;
    std::vector<double> u(last - first + 1);

    u[0] = 0.0;
    for (i = first + 1; i <= last; i++) {
        u[i - first] = u[i - first - 1] + (d.row(i % N) - d.row((i - 1) % N)).norm();
    }

    for (i = first + 1; i <= last; i++) {
        u[i - first] = u[i - first] / u[last - first];
    }

    return (u);
}

double ComputeMaxError(const Eigen::Ref<const Eigen::MatrixX2d> &d, int first, int last,
                              const Eigen::Matrix<double, 4, 2> &bezCurve, const std::vector<double> &u,
                              int &splitPoint) {
    int N = d.rows();
    int i;
    double maxDist;        /*  Maximum error		*/
    double dist;        /*  Current error		*/
    Eigen::Vector2d P;            /*  Point on curve		*/
    Eigen::Vector2d v;            /*  Vector from point to curve	*/

    splitPoint = (last - first + 1) / 2;
    maxDist = 0.0;
    for (i = first + 1; i < last; i++) {
        P = BezierII(3, bezCurve, u[i - first]);
        v = P - d.row(i % N).transpose();
        dist = v.squaredNorm();
        if (dist >= maxDist) {
            maxDist = dist;
            splitPoint = i;
        }
    }
    return (maxDist);
}
