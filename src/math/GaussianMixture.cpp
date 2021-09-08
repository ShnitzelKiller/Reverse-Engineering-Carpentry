//
// Created by James Noeckel on 1/22/20.
//

#include <numeric>
#include "GaussianMixture.h"
#include <random>

GaussianMixture::GaussianMixture(int k, int d, double min_eigenvalue) :
    d_(d), k_(k), mu_(k, d), sigmas_(k, Eigen::MatrixXd(d, d)),
    log_pi_(k), normalizations_(k), cholesky_decompositions_(k), min_eigenvalue_(min_eigenvalue) {
}

GaussianMixture::GaussianMixture() : d_(0), k_(0) {
}

void GaussianMixture::setNumComponents(int k) {
    k_ = k;
    d_ = 0;
    clear();
}

int GaussianMixture::getNumComponents() const {
    return k_;
}

int GaussianMixture::getNumDims() const {
    return d_;
}

const Eigen::MatrixXd &GaussianMixture::means() const {
    return mu_;
}

const std::vector<Eigen::MatrixXd> &GaussianMixture::sigmas() const {
    return sigmas_;
}

const Eigen::VectorXd &GaussianMixture::log_probs() const {
    return log_pi_;
}

void GaussianMixture::allocate(int k, int d) {
    k_ = k;
    d_ = d;
    sigmas_ = std::vector<Eigen::MatrixXd>(k_, Eigen::MatrixXd(d_, d_));
    cholesky_decompositions_.resize(k_);
    normalizations_.resize(k_);
    mu_.resize(k_, d_);
    log_pi_.resize(k_);
    clear();
}

bool GaussianMixture::initialize_random_means(const Eigen::Ref<const Eigen::MatrixXd> &data, int num_components) {
    int new_k = k_;
    if (num_components > 0) new_k = num_components;
    int n = data.rows();
    if (n < new_k) return false;
    if (new_k != k_ || data.cols() != d_) allocate(new_k, data.cols());

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    int mean_index = 0;
    for (int pick=0; mean_index < k_ && pick < n; pick++) {
        int index = indices[pick];
        bool retry = false;
        for (int j=0; j<mean_index; j++) {
            if (mu_.row(j).isApprox(data.row(index))) {
                retry = true;
                break;
            }
        }
        if (!retry) {
            mu_.row(mean_index) = data.row(index);
            mean_index++;
        }
    }
    complete_ = false;

    if (mean_index < k_) {
        initialized_means_ = false;
        return false;
    }

    initialized_means_ = true;
    return true;
}

bool GaussianMixture::initialize(const Eigen::Ref<const Eigen::MatrixXd> &means,
                                 const std::vector<Eigen::MatrixXd> &sigmas,
                                 const Eigen::Ref<const Eigen::VectorXd> &probabilities) {
    if (means.rows() != k_ || means.cols() != d_) {
        allocate(means.rows(), means.cols());
    }
    mu_ = means;
    complete_ = false;
    if (sigmas.size() != k_ || probabilities.size() != k_) {
        return false;
    }
    for (int k=0; k<k_; k++) {
        if (sigmas[k].rows() != d_ || sigmas[k].cols() != d_) {
            initialized_sigmas_ = false;
            return false;
        }
        sigmas_[k] = sigmas[k];
    }

    log_pi_ = probabilities.array().log();
    initialized_means_ = true;
    initialized_sigmas_ = true;
    initialized_pis_ = true;
    return true;
}

bool GaussianMixture::initialize(const Eigen::Ref<const Eigen::MatrixXd> &means) {
    if (means.rows() != k_ || means.cols() != d_) allocate(means.rows(), means.cols());
    mu_ = means;
    initialized_means_ = true;
    complete_ = false;
    return true;
}

bool GaussianMixture::initialize_volume(const Eigen::Ref<const Eigen::VectorXd> &lower_bound, const Eigen::Ref<const Eigen::VectorXd> &upper_bound, bool initialize_sigmas, int components) {
    int new_k = k_;
    if (components > 0) new_k = components;
    if (new_k <= 0 || lower_bound.size() != upper_bound.size()) return false;
    if (new_k != k_ || lower_bound.size() != d_) allocate(new_k, lower_bound.size());

    Eigen::ArrayXd range = upper_bound - lower_bound;
    double volume = range.prod();
    // (r^d) * n = V => r = (V/n)^(1/d)
    double r = std::pow(0.5*volume/k_, 1.0/d_);
    if (initialize_sigmas) {
        for (int i = 0; i < k_; i++) {
            sigmas_[i] = Eigen::MatrixXd::Identity(d_, d_) * (r * r);
        }
        initialized_sigmas_ = true;
    }
    for (int i=0; i<k_; i++) {
        bool retry = true;
        int iters = 0;
        for (; retry && iters<50; iters++) {
            retry = false;
            mu_.row(i) = (Eigen::RowVectorXd::Random(d_).array() + 1) * 0.5 * range;
            for (int j = 0; j < i; j++) {
                if ((mu_.row(j) - mu_.row(i)).norm() < r) {
                    retry = true;
                    break;
                }
            }
        }
    }
    initialized_means_ = true;
    complete_ = false;
    return true;
}

int GaussianMixture::initialize_k_means(const Eigen::Ref<const Eigen::MatrixXd> &data, int num_components, int max_iters) {
    int new_k = k_;
    if (num_components > 0) new_k = num_components;
    int n = data.rows();
    if (new_k != k_ || data.cols() != d_) allocate(new_k, data.cols());

    if (!initialized_means_) {
        if (!initialize_random_means(data)) {
            return 0;
        }
    }

    std::vector<int> assignments(n, -1);
    int iter=0;
    while (true) {
        bool changed = false;
        //compute assignments
        std::vector<std::vector<int>> cluster_indices(k_);
        for (int j = 0; j < k_; j++) {
            cluster_indices[j].reserve(n / k_);
        }
        for (int i = 0; i < n; i++) {
            double mindist = std::numeric_limits<double>::max();
            int cluster = -1;
            for (int j = 0; j < k_; j++) {
                double dist = (data.row(i) - mu_.row(j)).squaredNorm();
                if (dist < mindist) {
                    cluster = j;
                    mindist = dist;
                }
            }
            cluster_indices[cluster].push_back(i);
            if (assignments[i] != cluster) {
                changed = true;
            }
            assignments[i] = cluster;
        }

        /* Stop and save */
        if (!changed || iter > max_iters) {
            for (int j=0; j<k_; j++) {
                const auto &cluster = cluster_indices[j];
                Eigen::MatrixXd subdata(cluster.size(), data.cols());
                for (int i=0; i<cluster.size(); i++) {
                    subdata.row(i) = data.row(cluster[i]);
                }
                Eigen::RowVectorXd mean = subdata.colwise().mean();
                subdata.rowwise() -= mean;
                sigmas_[j] = subdata.transpose() * subdata / static_cast<double>(std::max(1, static_cast<int>(cluster.size()-1)));
                log_pi_[j] = log(static_cast<double>(cluster.size())/static_cast<double>(n));
            }
            break;
        }

        //compute means
        mu_.setZero();
        for (int j = 0; j < k_; j++) {
            const auto &cluster = cluster_indices[j];
            if (cluster.empty()) {
                initialized_means_ = false;
                return 0;
            }
            for (int i : cluster) {
                mu_.row(j) += data.row(i);
            }
            mu_.row(j) /= static_cast<double>(cluster.size());
        }
        iter++;
    }

    initialized_means_ = true;
    initialized_sigmas_ = true;
    initialized_pis_ = true;
    complete_ = false;
    return iter;
}

void GaussianMixture::clear() {
    initialized_means_ = false;
    initialized_sigmas_ = false;
    initialized_pis_ = false;
    complete_ = false;
}

bool GaussianMixture::success() const {
    return complete_;
}

bool GaussianMixture::useCurrentModel() {
    if (complete_) return true;
    if (k_ <= 0 || d_ <= 0) return false;
    bool success = recompute_normalizations();
    if (success) {
        complete_ = true;
        return true;
    } else {
        return false;
    }
}

bool GaussianMixture::recompute_normalizations() {
    for (int k = 0; k < k_; k++) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(sigmas_[k]);
        Eigen::ArrayXd eigenvalues = eigs.eigenvalues();
        for (int d=0; d<d_; d++) {
            if (eigenvalues[d] < min_eigenvalue_) {
                eigenvalues[d] = min_eigenvalue_;
            } else {
                break;
            }
        }
        sigmas_[k] = (eigs.eigenvectors().array().rowwise() * eigenvalues.transpose()).matrix() * eigs.eigenvectors().transpose();
        cholesky_decompositions_[k] = sigmas_[k].ldlt();
        if (cholesky_decompositions_[k].info() == Eigen::NumericalIssue) {
            return false;
        }
        double logdet = cholesky_decompositions_[k].vectorD().array().log().sum();
        normalizations_[k] = -0.5 * (d_ * log(2.0 * M_PI) + logdet);
    }
    return true;
}

double GaussianMixture::step(const Eigen::Ref<const Eigen::MatrixXd> &data, bool &success) {
    int n = data.rows();
    //E step
    log_p_z_.resize(n, k_);
    log_likelihood_all_.resize(n, k_);
    success = recompute_normalizations();
    if (!success) return std::numeric_limits<double>::lowest();

    //compute log(P( X_i, Z_i=j; θ ))
    for (int k = 0; k < k_; k++) {
        centered_ = data.rowwise() - mu_.row(k);
        log_likelihood_all_.col(k) =
                (log_pi_[k] + normalizations_[k]) - 0.5 * (centered_.array() * cholesky_decompositions_[k].solve(centered_.transpose()).transpose().array()).rowwise().sum();
        log_p_z_.col(k) = log_likelihood_all_.col(k); //joint
    }

    // normalize to get log(P( Z_i=j | X_i; θ ))
    log_p_z_.colwise() -= log_p_z_.rowwise().maxCoeff();
    log_p_z_.colwise() -= log_p_z_.array().exp().rowwise().sum().log().matrix();

    // take expectation of Σ_i[log(P( X_i, Z_i=j; θ ))] w.r.t. Z ~ P( Z | X; θ )
    double log_likelihood = (log_p_z_.array().exp() * log_likelihood_all_.array()).sum();

    Eigen::RowVectorXd log_N = log_p_z_.colwise().maxCoeff();
    log_N = (log_p_z_.rowwise() - log_N).array().exp().colwise().sum().log() + log_N.array();

    //M step pi
    log_pi_ = log_N.array().transpose() - log(static_cast<double>(n));

    //M step mu
    mu_ = (log_p_z_.array().exp().transpose().matrix() * data).array().colwise() * (-log_N).array().exp().transpose();

    //M step sigma
    for (int k = 0; k < k_; k++) {
        centered_ = (data.rowwise() - mu_.row(k));
        centered_.array().colwise() *= (log_p_z_.col(k).array()*0.5).exp();
        sigmas_[k] = centered_.transpose() * centered_ * exp(-log_N[k]);
    }
    return log_likelihood;
}


int GaussianMixture::learn(const Eigen::Ref<const Eigen::MatrixXd> &data, int maxiters, double eps) {
    int n = data.rows();
    if (k_ <= 0 || n <= data.cols()) return 0;
    if (data.cols() != d_) {
        allocate(k_, data.cols());
    }
    if (!initialized_means_) {
        if (initialize_k_means(data) == 0) {
            return 0;
        }
    }
    if (!initialized_sigmas_) {
        centered_ = data.rowwise() - data.colwise().mean();
        Eigen::MatrixXd cov = centered_.transpose() * centered_ / (n-1);
        for (int k=0; k<k_; k++) {
            sigmas_[k] = cov;
        }
    }
    if (!initialized_pis_) {
        log_pi_ = Eigen::VectorXd::Constant(k_, -log(static_cast<double>(k_)));
    }
    double log_likelihood = std::numeric_limits<double>::lowest();
    int iter=0;
    for (; iter<maxiters; iter++) {
        bool success;
        double new_log_likelihood = step(data, success);
        if (!success) {
            complete_ = false;
            initialized_means_ = false;
            initialized_pis_ = false;
            initialized_sigmas_ = false;
            return iter+1;
        }
        if (new_log_likelihood - log_likelihood < eps) {
            iter++;
            break;
        }
        log_likelihood = new_log_likelihood;
    }

    recompute_normalizations();

    centered_.resize(0,0);
    log_p_z_.resize(0,0);
    log_likelihood_all_.resize(0, 0);

    initialized_means_ = true;
    initialized_sigmas_ = true;
    initialized_pis_ = true;
    complete_ = true;
    return iter;
}

Eigen::MatrixXd GaussianMixture::logp_data_given_z(const Eigen::Ref<const Eigen::MatrixXd> &data) const {
    int n = data.rows();
    if (!complete_ || data.cols() != d_) {
        return Eigen::MatrixXd::Constant(n, k_, std::numeric_limits<double>::lowest());
    }
    Eigen::MatrixXd log_likelihoods(n, k_);
    for (int k=0; k<k_; k++) {
        Eigen::MatrixXd centered = data.rowwise() - mu_.row(k);
        log_likelihoods.col(k) = (log_pi_[k] + normalizations_[k]) - 0.5 * (centered.array() * cholesky_decompositions_[k].solve(centered.transpose()).transpose().array()).rowwise().sum();
    }
    return log_likelihoods;
}

Eigen::MatrixXd GaussianMixture::logp_z_given_data(const Eigen::Ref<const Eigen::MatrixXd> &data) const {
    int n = data.rows();
    if (!complete_ || data.cols() != d_) {
        return Eigen::MatrixXd::Constant(n, k_, std::numeric_limits<double>::lowest());
    }
    Eigen::MatrixXd log_likelihoods = logp_data_given_z(data);
    log_likelihoods.colwise() -= log_likelihoods.rowwise().maxCoeff();
    log_likelihoods.colwise() -= log_likelihoods.array().exp().rowwise().sum().log().matrix();
    return log_likelihoods;
}

Eigen::VectorXd GaussianMixture::logp_data(const Eigen::Ref<const Eigen::MatrixXd> &data) const {
    int n = data.rows();
    if (!complete_ || data.cols() != d_) {
        return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::lowest());
    }
    Eigen::MatrixXd log_likelihoods = logp_data_given_z(data);
    Eigen::VectorXd max_log_likelihoods = log_likelihoods.rowwise().maxCoeff();
    log_likelihoods.colwise() -= max_log_likelihoods;
    return log_likelihoods.array().exp().rowwise().sum().log() + max_log_likelihoods.array();
}

std::ostream &operator<<(std::ostream &o, const GaussianMixture &gmm) {
    o << "means: " << std::endl << gmm.mu_ << std::endl;
    o << "sigmas: " << std::endl;
    for (int k=0; k<gmm.k_; k++) {
        o << '[' << gmm.sigmas_[k] << ']' << std::endl;
    }
    o << "weights: " << std::endl << gmm.log_pi_.transpose().array().exp();
    return o;
}

double GaussianMixture::getMinEigenvalue() const {
    return min_eigenvalue_;
}

void GaussianMixture::setMinEigenvalue(double min_eigenvalue) {
    min_eigenvalue_ = min_eigenvalue;
}



