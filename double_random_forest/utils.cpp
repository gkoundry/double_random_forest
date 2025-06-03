#include "utils.hpp"
#include <iostream>
#include <random>

//------------------------------------------------------------------------------------------------
// Utility: compute bin edges for splitting on a numeric feature.
// Identical for both classification and regression.
std::vector<double> compute_bin_edges(const double *X_ptr,
                                      std::size_t n_features,
                                      const std::size_t *indices,
                                      std::size_t index_count, int column_index,
                                      int max_bin_count) {
  if (index_count == 0 || max_bin_count <= 1) {
    return {};
  }

  // Extract the feature values for the specified column and indices
  std::vector<double> feature_values;
  feature_values.reserve(index_count);

  for (std::size_t i = 0; i < index_count; ++i) {
    std::size_t row_idx = indices[i];
    feature_values.push_back(X_ptr[row_idx * n_features + column_index]);
  }

  // Sort the feature values
  std::sort(feature_values.begin(), feature_values.end());

  // Remove duplicates while preserving order
  auto last = std::unique(feature_values.begin(), feature_values.end());
  feature_values.erase(last, feature_values.end());

  // If we have fewer unique values than max_bin_count,
  // we can't create that many bins
  if (feature_values.size() < static_cast<std::size_t>(max_bin_count)) {
    // Return thresholds between each unique value
    std::vector<double> thresholds;
    for (std::size_t i = 0; i < feature_values.size() - 1; ++i) {
      thresholds.push_back((feature_values[i] + feature_values[i + 1]) / 2.0);
    }
    return thresholds;
  }

  // Calculate bin size for roughly equal-sized bins
  double bin_size = static_cast<double>(feature_values.size()) / max_bin_count;

  std::vector<double> thresholds;
  thresholds.reserve(max_bin_count - 1);

  // Generate thresholds at roughly equal intervals
  for (int i = 1; i < max_bin_count; ++i) {
    double target_position = i * bin_size;
    std::size_t idx = static_cast<std::size_t>(std::round(target_position));

    // Ensure we don't go out of bounds
    if (idx >= feature_values.size()) {
      idx = feature_values.size() - 1;
    }

    // Find a threshold that creates a meaningful split
    // Use the midpoint between consecutive unique values
    if (idx > 0 && idx < feature_values.size()) {
      double threshold = (feature_values[idx - 1] + feature_values[idx]) / 2.0;

      // Avoid duplicate thresholds
      if (thresholds.empty() || threshold > thresholds.back()) {
        thresholds.push_back(threshold);
      }
    }
  }

  return thresholds;
}

//------------------------------------------------------------------------------------------------
// Multiclass helper: for a given node, compute class frequencies over the
// subset of y.
//    Returns a length-(max_class+1) vector of relative frequencies.
std::vector<double>
compute_mean_bincount(const int *y_ptr,
                      const std::unique_ptr<std::size_t[]> &indices,
                      std::size_t index_count, std::size_t n_classes) {
  // Allocate counters for classes 0..n-1
  std::vector<int> counts(n_classes, 0);

  // Count occurrences
  for (std::size_t i = 0; i < index_count; ++i) {
    int value = y_ptr[indices[i]];
    ++counts[value];
  }

  // Convert to relative frequencies
  std::vector<double> freqs(counts.size());
  for (std::size_t c = 0; c < counts.size(); ++c) {
    freqs[c] = double(counts[c]) / double(index_count);
  }

  return freqs;
}

//------------------------------------------------------------------------------------------------
// Utility to convert a vector-of-vectors<double> into a 2D numpy array
py::array_t<double>
vector_to_numpy_efficient(const std::vector<std::vector<double>> &values) {
  std::size_t rows = values.size();
  std::size_t cols = values[0].size();
  auto result = py::array_t<double>({rows, cols});
  auto buf = result.mutable_unchecked<2>();

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      buf(i, j) = values[i][j];
    }
  }
  return result;
}

std::vector<std::size_t>
get_bootstrap_indices(const std::unique_ptr<std::size_t[]> &indices,
                      std::size_t index_size, std::size_t n_samples,
                      std::mt19937 &rng) {

  std::vector<std::size_t> bootstrap_indices(index_size);

  if (index_size < 0.1 * n_samples) {
    for (std::size_t i = 0; i < index_size; ++i)
      bootstrap_indices[i] = indices[i];
  } else {
    std::uniform_int_distribution<std::size_t> dist(0, index_size - 1);
    for (std::size_t i = 0; i < index_size; ++i)
      bootstrap_indices[i] = indices[dist(rng)];
  }

  return bootstrap_indices;
}
