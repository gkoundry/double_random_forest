#include "criterion.hpp"
#include "utils.hpp"

//------------------------------------------------------------------------------------------------
// Multiclass impurity: Gini impurity reduction for a candidate split.
// Exactly as in the original code.
double get_impurity_score(const double *X_ptr, std::size_t n_features,
                          const int *y_ptr, const std::size_t *indices,
                          std::size_t index_count, int col, double threshold) {
  if (index_count == 0)
    return 0.0;

  // Determine the number of classes (max_y+1)
  int maxClass = 0;
  for (std::size_t i = 0; i < index_count; ++i) {
    maxClass = std::max(maxClass, y_ptr[indices[i]]);
  }
  int K = maxClass + 1;

  // Count parent, left, right class counts
  std::vector<std::size_t> cntP(K), cntL(K), cntR(K);
  std::size_t nL = 0, nR = 0;
  for (std::size_t ii = 0; ii < index_count; ++ii) {
    std::size_t i = indices[ii];
    int c = y_ptr[i];
    cntP[c]++;
    double v = X_ptr[i * n_features + col];
    if (v <= threshold) {
      cntL[c]++;
      ++nL;
    } else {
      cntR[c]++;
      ++nR;
    }
  }
  if (nL == 0 || nR == 0)
    return 0.0;

  // Compute Gini impurity for a given count vector
  auto gini = [&](const std::vector<std::size_t> &cnt, std::size_t n) {
    double g = 1.0;
    for (std::size_t c = 0; c < cnt.size(); ++c) {
      double p = double(cnt[c]) / double(n);
      g -= p * p;
    }
    return g;
  };

  double Gp = gini(cntP, index_count);
  double Gl = gini(cntL, nL);
  double Gr = gini(cntR, nR);
  double n = double(index_count);
  return Gp - (double(nL) / n) * Gl - (double(nR) / n) * Gr;
}

//------------------------------------------------------------------------------------------------
// Multiclass class count
std::size_t MulticlassCriterion::n_classes(const int *y_ptr,
                                           std::size_t n_samples) const {
  std::size_t max_class = 0;
  for (std::size_t i = 0; i < n_samples; ++i) {
    max_class = std::max(max_class, static_cast<std::size_t>(y_ptr[i]));
  }
  return max_class + 1; // classes are 0-indexed
}

//------------------------------------------------------------------------------------------------
// Multiclass criterion (Gini + class frequencies)
std::vector<double> MulticlassCriterion::leaf_value(
    const int *y_ptr, const std::unique_ptr<std::size_t[]> &indices,
    std::size_t count, std::size_t n_classes) const {
  return compute_mean_bincount(y_ptr, indices, count, n_classes);
}

double MulticlassCriterion::impurity_gain(const double *X_ptr,
                                          std::size_t n_features,
                                          const int *y_ptr,
                                          const std::size_t *indices,
                                          std::size_t count, int feature_col,
                                          double threshold) const {
  return get_impurity_score(X_ptr, n_features, y_ptr, indices, count,
                            feature_col, threshold);
}

//------------------------------------------------------------------------------------------------
// Regression class count
std::size_t RegressionCriterion::n_classes(const double * /* y_ptr */,
                                           std::size_t /* n_samples */) const {
  return 1; // not used in regression, but required by interface
}

//------------------------------------------------------------------------------------------------
// Regression criterion (variance reduction + leaf mean)
std::vector<double> RegressionCriterion::leaf_value(
    const double *y_ptr, const std::unique_ptr<std::size_t[]> &indices,
    std::size_t count, std::size_t /* n_classes */) const {
  double sum = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    sum += y_ptr[indices[i]];
  }
  double mu = sum / double(count);
  return std::vector<double>{mu};
}

double RegressionCriterion::impurity_gain(const double *X_ptr,
                                          std::size_t n_features,
                                          const double *y_ptr,
                                          const std::size_t *indices,
                                          std::size_t count, int feature_col,
                                          double threshold) const {
  if (count == 0)
    return 0.0;

  // Parent mean & variance
  double sumP = 0.0, sumSqP = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    double yi = y_ptr[indices[i]];
    sumP += yi;
    sumSqP += yi * yi;
  }
  double muP = sumP / double(count);
  double varP = (sumSqP / double(count)) - (muP * muP);

  // Split into left/right sums & squared sums
  double sumL = 0.0, sumSqL = 0.0;
  double sumR = 0.0, sumSqR = 0.0;
  std::size_t nL = 0, nR = 0;
  for (std::size_t ii = 0; ii < count; ++ii) {
    std::size_t idx = indices[ii];
    double xv = X_ptr[idx * n_features + feature_col];
    double yi = y_ptr[idx];
    if (xv <= threshold) {
      sumL += yi;
      sumSqL += yi * yi;
      ++nL;
    } else {
      sumR += yi;
      sumSqR += yi * yi;
      ++nR;
    }
  }
  if (nL == 0 || nR == 0)
    return 0.0;

  double muL = sumL / double(nL);
  double varL = (sumSqL / double(nL)) - (muL * muL);

  double muR = sumR / double(nR);
  double varR = (sumSqR / double(nR)) - (muR * muR);

  double wL = double(nL) / double(count);
  double wR = double(nR) / double(count);
  double varWeighted = wL * varL + wR * varR;

  return varP - varWeighted;
}
