#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using matrix_exp_fn = void(*)(Tensor&, const Tensor&);

DECLARE_DISPATCH(matrix_exp_fn, matrix_exp_stub);

// common functions used for both CPU and CUDA

static inline Tensor operator_1_norm(const Tensor& t) {
  return std::get<0>(t.abs().sum(-2).max(-1));
}

template <typename scalar_t, int ROW, int COL>
using array2d = std::array<std::array<scalar_t, COL>, ROW>;

template <typename scalar_t>
static inline Tensor compute_T18(const Tensor& A) {
  constexpr int num_prods = 5;
  constexpr array2d<scalar_t, num_prods, num_prods> b = {{
    {
      0.,
      -1.00365581030144618291e-01,
      -8.02924648241156932449e-03,
      -8.92138498045729985177e-04,
      0.
    },
    {
      0.,
      3.97849749499645077844e-01,
      1.36783778460411720168e+00,
      4.98289622525382669416e-01,
      -6.37898194594723280150e-04
    },
    {
      -1.09676396052962061844e+01,
      1.68015813878906206114e+00,
      5.71779846478865511061e-02,
      -6.98210122488052056106e-03,
      3.34975017086070470649e-05
    },
    {
      -9.04316832390810593223e-02,
      -6.76404519071381882256e-02,
      6.75961301770459654925e-02,
      2.95552570429315521194e-02,
      -1.39180257516060693404e-05
    },
    {
      0.,
      0.,
      -9.23364619367118555360e-02,
      -1.69364939002081722752e-02,
      -1.40086798182036094347e-05
    }
  }};

  const auto& I = at::eye(A.size(-1), A.options()).expand_as(A);
  const auto& A2 = at::matmul(A, A);
  const auto& A3 = at::matmul(A2, A);
  const auto& A6 = at::matmul(A3, A3);
  std::array<
    std::reference_wrapper<const Tensor>,
    num_prods> As = {I, A, A2, A3, A6};

  std::array<Tensor, num_prods> Bs;
  for (int i = 0; i < num_prods; ++i) {
    Bs[i] = at::zeros(A.sizes(), A.options());
  }

  for (int i = 0; i < num_prods; ++i) {
    for (int j = 0; j < num_prods; ++j) {
      Bs[i] += b[i][j] * As[j];
    }
  }

  const auto& A9 = at::matmul(Bs[0], Bs[4]) + Bs[3];
  const auto& res = Bs[1] + at::matmul(Bs[2] + A9, A9);

  return res;
}


}} // namespace at::native
