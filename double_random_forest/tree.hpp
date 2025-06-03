#ifndef TREE_HPP
#define TREE_HPP

#include <vector>

#define MAX_BIN_COUNT 64
#define MIN_ITEMS_PER_BIN 5

template <typename YType, typename CRIT>
void fit_tree_generic(const double *X_ptr, std::size_t n_samples,
                      std::size_t n_features, const YType *y_ptr,
                      const CRIT &criterion, std::vector<int> &left_children,
                      std::vector<int> &right_children,
                      std::vector<int> &feature_indices,
                      std::vector<double> &thresholds,
                      std::vector<std::vector<double>> &values, int max_depth,
                      std::size_t min_samples_leaf, int max_features,
                      int random_state);

#endif // TREE_HPP
