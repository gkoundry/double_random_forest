#include "tree.hpp"
#include "criterion.hpp"
#include "utils.hpp"
#include <algorithm>
#include <memory>
#include <queue>
#include <random>
#include <vector>

//------------------------------------------------------------------------------------------------
// Core tree data structures
struct Split {
  int feature_index;
  double threshold;
  double score;

  Split(int fi, double t, double s)
      : feature_index(fi), threshold(t), score(s) {}
};

struct CandidateSplit {
  int node_index;
  int depth;
  struct Split split;

  CandidateSplit(int ni, int d, const struct Split &s)
      : node_index(ni), depth(d), split(s) {}
};

// custom comparator for max-heap on split.score (descending)
struct CompareSplit {
  bool operator()(const CandidateSplit &a, const CandidateSplit &b) const {
    if (a.split.score == b.split.score) {
      return a.node_index > b.node_index; // tie-break by node_index
    }
    return a.split.score < b.split.score; // max-heap
  }
};

//------------------------------------------------------------------------------------------------
// Generic "get the best split" using any SplitCriterion<YType>
template <typename YType>
Split get_best_split_generic(const double *X_ptr, std::size_t n_features,
                             const YType *y_ptr, const std::size_t *indices,
                             std::size_t count,
                             const SplitCriterion<YType> &criterion,
                             int max_features, std::mt19937 &rng) {
  double best_score = -1.0;
  int best_feature = -1;
  double best_threshold = -1.0;

  std::vector<int> pool(n_features);
  std::iota(pool.begin(), pool.end(), 0);
  std::shuffle(pool.begin(), pool.end(), rng);

  for (int f = 0; f < max_features; ++f) {
    std::size_t feature_index = pool[f];
    auto edges = compute_bin_edges(X_ptr, n_features, indices, count,
                                   feature_index, MAX_BIN_COUNT);
    for (double thr : edges) {
      double sc = criterion.impurity_gain(X_ptr, n_features, y_ptr, indices,
                                          count, feature_index, thr);
      if (sc > best_score) {
        best_score = sc;
        best_feature = feature_index;
        best_threshold = thr;
      }
    }
  }
  return Split(best_feature, best_threshold, best_score);
}

//------------------------------------------------------------------------------------------------
// Generic "fit tree" routine, shared by multiclass and regression
template <typename YType, typename CRIT>
void fit_tree_generic(const double *X_ptr, std::size_t n_samples,
                      std::size_t n_features, const YType *y_ptr,
                      const CRIT &criterion, std::vector<int> &left_children,
                      std::vector<int> &right_children,
                      std::vector<int> &feature_indices,
                      std::vector<double> &thresholds,
                      std::vector<std::vector<double>> &values, int max_depth,
                      std::size_t min_samples_leaf, int max_features,
                      int random_state) {
  if (max_depth < 0) {
    max_depth = std::numeric_limits<int>::max();
  }
  std::mt19937 rng(random_state);

  // Initialize root
  left_children.push_back(-1);
  right_children.push_back(-1);
  feature_indices.push_back(-1);
  thresholds.push_back(-1);

  std::vector<std::unique_ptr<std::size_t[]>> row_indices;
  auto ix0 = std::make_unique<std::size_t[]>(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    ix0[i] = i;
  }
  row_indices.push_back(std::move(ix0));
  std::vector<std::size_t> row_counts{n_samples};

  // Compute root's leaf value and store
  std::size_t n_classes = criterion.n_classes(y_ptr, n_samples);
  values.push_back(
      criterion.leaf_value(y_ptr, row_indices[0], row_counts[0], n_classes));

  std::vector<std::size_t> bootstrap_indices =
      get_bootstrap_indices(row_indices[0], row_counts[0], n_samples, rng);

  // Compute best split for root
  Split root_split = get_best_split_generic<YType>(
      X_ptr, n_features, y_ptr, bootstrap_indices.data(),
      bootstrap_indices.size(), criterion, max_features, rng);

  std::priority_queue<CandidateSplit, std::vector<CandidateSplit>, CompareSplit>
      pq;
  if (root_split.score > 0) {
    pq.push(CandidateSplit(0, 0, root_split));
  }

  // Main splitting loop
  while (!pq.empty()) {
    CandidateSplit top = pq.top();
    pq.pop();
    int node = top.node_index;

    // Assign children indices
    int left_idx = left_children.size();
    int right_idx = left_idx + 1;
    left_children[node] = left_idx;
    right_children[node] = right_idx;
    feature_indices[node] = top.split.feature_index;
    thresholds[node] = top.split.threshold;

    // Allocate new child slots
    left_children.push_back(-1); // for left_idx
    right_children.push_back(-1);
    left_children.push_back(-1); // for right_idx
    right_children.push_back(-1);
    feature_indices.push_back(-1);
    feature_indices.push_back(-1);
    thresholds.push_back(-1);
    thresholds.push_back(-1);

    // Partition this node's indices into left/right
    std::size_t parent_count = row_counts[node];
    auto left_ix = std::make_unique<std::size_t[]>(parent_count);
    auto right_ix = std::make_unique<std::size_t[]>(parent_count);
    std::size_t nL = 0, nR = 0;
    int fcol = top.split.feature_index;
    double thr = top.split.threshold;

    for (std::size_t i = 0; i < parent_count; ++i) {
      std::size_t idx = row_indices[node][i];
      double xv = X_ptr[idx * n_features + fcol];
      if (xv <= thr) {
        left_ix[nL++] = idx;
      } else {
        right_ix[nR++] = idx;
      }
    }

    row_indices[node].reset();                  // free parent's index array
    row_indices.push_back(std::move(left_ix));  // at index = left_idx
    row_indices.push_back(std::move(right_ix)); // at index = right_idx
    row_counts.push_back(nL);
    row_counts.push_back(nR);

    // Compute and store leaf values for left & right children
    values.push_back(
        criterion.leaf_value(y_ptr, row_indices[left_idx], nL, n_classes));
    values.push_back(
        criterion.leaf_value(y_ptr, row_indices[right_idx], nR, n_classes));

    if (top.depth + 1 >= max_depth || nL < min_samples_leaf ||
        nR < min_samples_leaf) {
      // If we reached max depth, we don't need to split further
      row_indices[left_idx].reset();
      row_indices[right_idx].reset();
      continue;
    }

    // Attempt to split left child further
    bootstrap_indices = get_bootstrap_indices(
        row_indices[left_idx], row_counts[left_idx], n_samples, rng);
    Split splitL = get_best_split_generic<YType>(X_ptr, n_features, y_ptr,
                                                 bootstrap_indices.data(), nL,
                                                 criterion, max_features, rng);
    if (splitL.score > 0) {
      pq.push(CandidateSplit(left_idx, top.depth + 1, splitL));
    } else {
      row_indices[left_idx].reset();
    }

    // Attempt to split right child further
    bootstrap_indices = get_bootstrap_indices(
        row_indices[right_idx], row_counts[right_idx], n_samples, rng);
    Split splitR = get_best_split_generic<YType>(X_ptr, n_features, y_ptr,
                                                 bootstrap_indices.data(), nR,
                                                 criterion, max_features, rng);
    if (splitR.score > 0) {
      pq.push(CandidateSplit(right_idx, top.depth + 1, splitR));
    } else {
      row_indices[right_idx].reset();
    }
  }
}

// Instantiate fit_tree_generic<int, MulticlassCriterion>
template void fit_tree_generic<int, MulticlassCriterion>(
    const double *,                     // X_ptr
    std::size_t,                        // n_samples
    std::size_t,                        // n_features
    const int *,                        // y_ptr
    const MulticlassCriterion &,        // criterion (must be 'const CRIT&')
    std::vector<int> &,                 // left_children
    std::vector<int> &,                 // right_children
    std::vector<int> &,                 // feature_indices
    std::vector<double> &,              // thresholds
    std::vector<std::vector<double>> &, // values
    int,                                // max_depth
    std::size_t,                        // min_samples_leaf
    int,                                // max_features
    int                                 // random_state
);

// Instantiate fit_tree_generic<double, RegressionCriterion>
template void fit_tree_generic<double, RegressionCriterion>(
    const double *,                     // X_ptr
    std::size_t,                        // n_samples
    std::size_t,                        // n_features
    const double *,                     // y_ptr
    const RegressionCriterion &,        // criterion (must be 'const CRIT&')
    std::vector<int> &,                 // left_children
    std::vector<int> &,                 // right_children
    std::vector<int> &,                 // feature_indices
    std::vector<double> &,              // thresholds
    std::vector<std::vector<double>> &, // values
    int,                                // max_depth
    std::size_t,                        // min_samples_leaf
    int,                                // max_features
    int                                 // random_state
);
