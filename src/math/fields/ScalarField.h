//
// Created by James Noeckel on 9/2/20.
//
#pragma once
#include <memory>
#include <Eigen/Dense>

template <int dim>
struct ScalarField {
    /**
     * Get the value of the field at each query point
     * @param Q n x dim matrix of query points
     * @return n dimensional results vector for each point
     */
    virtual Eigen::VectorXd operator()(const Eigen::Ref<const Eigen::Matrix<double, -1, dim>> &Q) const = 0;

    virtual ~ScalarField() = default;

    typedef std::shared_ptr<ScalarField<dim>> Handle;
};

/*template <int dim>
class BinaryOpField : ScalarField<dim> {
public:
    enum class Type {
        ADDITION,
        SUBTRACTION,
        MULTIPLICATION,
        DIVISION
    };

    BinaryOpField(typename ScalarField<dim>::Handle left, typename ScalarField<dim>::Handle right, Type type)
    : left_(std::move(left)), right_(std::move(right)), type_(type) {}

    Eigen::VectorXd operator()(const Eigen::Ref<const Eigen::Matrix<double, -1, dim>> &Q) const override {
        switch (type_) {
            case Type::ADDITION:
                return (*left_)(Q) + (*right_)(Q);
                break;
            case Type::SUBTRACTION:
                return (*left_)(Q) - (*right_)(Q);
                break;
            case Type::MULTIPLICATION:
                return (*left_)(Q).array() * (*right_)(Q).array();
                break;
            case Type::DIVISION:
                return (*left_)(Q).array() / (*right_)(Q).array();
                break;
        }
    }

private:
    Type type_;
    typename ScalarField<dim>::Handle left_, right_;
};

template <int dim>
typename ScalarField<dim>::Handle operator+(const typename ScalarField<dim>::Handle &left, const typename ScalarField<dim>::Handle &right) {
    return std::make_shared<BinaryOpField<dim>>(left, right, BinaryOpField<dim>::Type::ADDITION);
}*/
