#ifndef UTILS_HPP
#define UTILS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>

namespace py = pybind11;

std::vector<double> compute_bin_edges(const double *X_ptr,
                                      std::size_t n_features,
                                      const std::size_t *indices,
                                      std::size_t index_count, int column_index,
                                      int max_bin_count);

std::vector<double>
compute_mean_bincount(const int *y_ptr,
                      const std::unique_ptr<std::size_t[]> &indices,
                      std::size_t index_count, std::size_t n_classes);

py::array_t<double>
vector_to_numpy_efficient(const std::vector<std::vector<double>> &values);

std::vector<std::size_t>
get_bootstrap_indices(const std::unique_ptr<std::size_t[]> &indices,
                      std::size_t index_size, std::size_t n_samples,
                      std::mt19937 &rng);
#endif
