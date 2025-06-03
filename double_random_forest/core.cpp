#include "criterion.hpp"
#include "tree.hpp"
#include "utils.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

//------------------------------------------------------------------------------------------------
// PyBind11 wrapper: build_multiclass_tree
py::tuple build_multiclass_tree(py::array_t<double> X, py::array_t<int> y,
                                int max_depth, std::size_t min_samples_leaf,
                                int max_features, int random_state) {
  // Unwrap X
  auto Xc = py::array::ensure(X, py::array::c_style);
  auto X_buf = Xc.request();
  std::size_t n_samples = X_buf.shape[0];
  std::size_t n_features = X_buf.shape[1];
  const double *X_ptr = static_cast<const double *>(X_buf.ptr);

  // Unwrap y
  auto y_buf = y.request();
  const int *y_ptr = static_cast<const int *>(y_buf.ptr);

  // Prepare containers
  std::vector<int> left_children;
  std::vector<int> right_children;
  std::vector<int> feature_indices;
  std::vector<double> thresholds;
  std::vector<std::vector<double>> values;

  // Build tree with MulticlassCriterion
  MulticlassCriterion crit;
  fit_tree_generic<int, MulticlassCriterion>(
      X_ptr, n_samples, n_features, y_ptr, crit, left_children, right_children,
      feature_indices, thresholds, values, max_depth, min_samples_leaf,
      max_features, random_state);

  // Convert to numpy arrays
  py::array_t<int> L_arr(left_children.size(), left_children.data());
  py::array_t<int> R_arr(right_children.size(), right_children.data());
  py::array_t<int> F_arr(feature_indices.size(), feature_indices.data());
  py::array_t<double> T_arr(thresholds.size(), thresholds.data());
  py::array_t<double> V_arr = vector_to_numpy_efficient(values);

  return py::make_tuple(L_arr, R_arr, F_arr, T_arr, V_arr);
}

//------------------------------------------------------------------------------------------------
// 12) PyBind11 wrapper: build_regression_tree
py::tuple build_regression_tree(py::array_t<double> X, py::array_t<double> y,
                                int max_depth, std::size_t min_samples_leaf,
                                int max_features, int random_state) {
  // Unwrap X
  auto Xc = py::array::ensure(X, py::array::c_style);
  auto X_buf = Xc.request();
  std::size_t n_samples = X_buf.shape[0];
  std::size_t n_features = X_buf.shape[1];
  const double *X_ptr = static_cast<const double *>(X_buf.ptr);

  // Unwrap y (double)
  auto y_buf = y.request();
  const double *y_ptr = static_cast<const double *>(y_buf.ptr);

  // Prepare containers
  std::vector<int> left_children;
  std::vector<int> right_children;
  std::vector<int> feature_indices;
  std::vector<double> thresholds;
  std::vector<std::vector<double>> values;

  // Build tree with RegressionCriterion
  RegressionCriterion crit;
  fit_tree_generic<double, RegressionCriterion>(
      X_ptr, n_samples, n_features, y_ptr, crit, left_children, right_children,
      feature_indices, thresholds, values, max_depth, min_samples_leaf,
      max_features, random_state);

  // Convert to numpy arrays
  py::array_t<int> L_arr(left_children.size(), left_children.data());
  py::array_t<int> R_arr(right_children.size(), right_children.data());
  py::array_t<int> F_arr(feature_indices.size(), feature_indices.data());
  py::array_t<double> T_arr(thresholds.size(), thresholds.data());
  py::array_t<double> V_arr = vector_to_numpy_efficient(values);

  return py::make_tuple(L_arr, R_arr, F_arr, T_arr, V_arr);
}

//------------------------------------------------------------------------------------------------
// PyBind11 wrapper: predict_multiclass_tree (unchanged)
py::array_t<double> predict_multiclass_tree(py::array_t<double> X,
                                            py::array_t<int> left_children,
                                            py::array_t<int> right_children,
                                            py::array_t<int> feature_indices,
                                            py::array_t<double> thresholds,
                                            py::array_t<double> values) {
  auto Xc = py::array::ensure(X, py::array::c_style);
  auto X_buf = Xc.request();
  std::size_t n_samples = X_buf.shape[0];
  std::size_t n_features = X_buf.shape[1];
  const double *X_ptr = static_cast<const double *>(X_buf.ptr);

  auto Lb = left_children.request();
  const int *L_ptr = static_cast<const int *>(Lb.ptr);
  auto Rb = right_children.request();
  const int *R_ptr = static_cast<const int *>(Rb.ptr);
  auto Fb = feature_indices.request();
  const int *F_ptr = static_cast<const int *>(Fb.ptr);
  auto Tb = thresholds.request();
  const double *T_ptr = static_cast<const double *>(Tb.ptr);
  auto Vb = values.request();
  const double *V_ptr = static_cast<const double *>(Vb.ptr);

  // Number of classes = values.shape(1)
  std::size_t n_classes = values.shape(1);

  // Prepare output array (n_samples x n_classes)
  auto predictions = py::array_t<double>({static_cast<py::ssize_t>(n_samples),
                                          static_cast<py::ssize_t>(n_classes)});
  auto pred_buf = predictions.mutable_unchecked<2>();

  // Zero-out
  for (std::size_t i = 0; i < n_samples; ++i) {
    for (std::size_t j = 0; j < n_classes; ++j) {
      pred_buf(i, j) = 0.0;
    }
  }

  // For each sample, traverse the tree
  for (std::size_t i = 0; i < n_samples; ++i) {
    int node = 0;
    while (L_ptr[node] >= 0) {
      int fcol = F_ptr[node];
      double thr = T_ptr[node];
      if (X_ptr[i * n_features + fcol] <= thr) {
        node = L_ptr[node];
      } else {
        node = R_ptr[node];
      }
    }
    // copy leaf's class-prob vector
    for (std::size_t j = 0; j < n_classes; ++j) {
      pred_buf(i, j) = V_ptr[node * n_classes + j];
    }
  }
  return predictions;
}

//------------------------------------------------------------------------------------------------
// PyBind11 wrapper: predict_regression_tree
py::array_t<double> predict_regression_tree(py::array_t<double> X,
                                            py::array_t<int> left_children,
                                            py::array_t<int> right_children,
                                            py::array_t<int> feature_indices,
                                            py::array_t<double> thresholds,
                                            py::array_t<double> values) {
  auto Xc = py::array::ensure(X, py::array::c_style);
  auto X_buf = Xc.request();
  std::size_t n_samples = X_buf.shape[0];
  std::size_t n_features = X_buf.shape[1];
  const double *X_ptr = static_cast<const double *>(X_buf.ptr);

  auto Lb = left_children.request();
  const int *L_ptr = static_cast<const int *>(Lb.ptr);
  auto Rb = right_children.request();
  const int *R_ptr = static_cast<const int *>(Rb.ptr);
  auto Fb = feature_indices.request();
  const int *F_ptr = static_cast<const int *>(Fb.ptr);
  auto Tb = thresholds.request();
  const double *T_ptr = static_cast<const double *>(Tb.ptr);
  auto Vb = values.request();
  const double *V_ptr = static_cast<const double *>(Vb.ptr);

  // In regression, values is (n_nodes x 1)
  py::array_t<double> preds(n_samples);
  auto pbuf = preds.mutable_unchecked<1>();

  // Traverse each sample to its leaf
  for (std::size_t i = 0; i < n_samples; ++i) {
    int node = 0;
    while (L_ptr[node] >= 0) {
      int fcol = F_ptr[node];
      double thr = T_ptr[node];
      if (X_ptr[i * n_features + fcol] <= thr) {
        node = L_ptr[node];
      } else {
        node = R_ptr[node];
      }
    }
    // leaf's mean is V_ptr[node]
    pbuf(i) = V_ptr[node];
  }
  return preds;
}

//------------------------------------------------------------------------------------------------
// Module definition
PYBIND11_MODULE(core, m) {
  m.doc() = "Decision-tree builders & predictors (multiclass + regression)";

  m.def("build_multiclass_tree", &build_multiclass_tree,
        "Create a decision tree for multiclass target", py::arg("X"),
        py::arg("y"), py::arg("max_depth"), py::arg("min_samples_leaf"),
        py::arg("max_features"), py::arg("random_state"));

  m.def("predict_multiclass_tree", &predict_multiclass_tree,
        "Predict using a multiclass decision tree", py::arg("X"),
        py::arg("left_children"), py::arg("right_children"),
        py::arg("feature_indices"), py::arg("thresholds"), py::arg("values"));

  m.def("build_regression_tree", &build_regression_tree,
        "Create a decision tree for regression target", py::arg("X"),
        py::arg("y"), py::arg("max_depth"), py::arg("min_samples_leaf"),
        py::arg("max_features"), py::arg("random_state"));

  m.def("predict_regression_tree", &predict_regression_tree,
        "Predict using a regression decision tree", py::arg("X"),
        py::arg("left_children"), py::arg("right_children"),
        py::arg("feature_indices"), py::arg("thresholds"), py::arg("values"));
}
