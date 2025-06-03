#ifndef CRITERION_HPP
#define CRITERION_HPP

#include <memory>
#include <vector>

//------------------------------------------------------------------------------------------------
// Abstract interface for "leaf value" + "impurity gain"
// YType = int (for multiclass) or double (for regression)
template <typename YType> struct SplitCriterion {
  // Given y_ptr and indices[0..n_samples-1], return the number of classes.
  virtual std::size_t n_classes(const YType *y_ptr,
                                std::size_t n_samples) const = 0;

  // Given y_ptr and indices[0..count-1], compute the leaf's "value."
  //   - multiclass: a vector of class probabilities
  //   - regression: a length-1 vector containing the mean
  virtual std::vector<double>
  leaf_value(const YType *y_ptr, const std::unique_ptr<std::size_t[]> &indices,
             std::size_t count, std::size_t n_classes) const = 0;

  // Given X, y, and a candidate split (feature_col <= threshold), return
  // the "impurity gain" (to be maximized).
  virtual double impurity_gain(const double *X_ptr, std::size_t n_features,
                               const YType *y_ptr, const std::size_t *indices,
                               std::size_t count, int feature_col,
                               double threshold) const = 0;

  virtual ~SplitCriterion() = default;
};
//------------------------------------------------------------------------------------------------
// Multiclass criterion (Gini + class frequencies)
struct MulticlassCriterion : SplitCriterion<int> {
  std::size_t n_classes(const int *y_ptr, std::size_t n_samples) const override;
  std::vector<double> leaf_value(const int *y_ptr,
                                 const std::unique_ptr<std::size_t[]> &indices,
                                 std::size_t count,
                                 std::size_t n_classes) const override;

  double impurity_gain(const double *X_ptr, std::size_t n_features,
                       const int *y_ptr, const std::size_t *indices,
                       std::size_t count, int feature_col,
                       double threshold) const override;
};

//------------------------------------------------------------------------------------------------
// Regression criterion (variance reduction + leaf mean)
struct RegressionCriterion : SplitCriterion<double> {
  std::size_t n_classes(const double *y_ptr,
                        std::size_t n_samples) const override;
  std::vector<double> leaf_value(const double *y_ptr,
                                 const std::unique_ptr<std::size_t[]> &indices,
                                 std::size_t count,
                                 std::size_t n_classes) const override;

  double impurity_gain(const double *X_ptr, std::size_t n_features,
                       const double *y_ptr, const std::size_t *indices,
                       std::size_t count, int feature_col,
                       double threshold) const override;
};

#endif // CRITERION_HPP
