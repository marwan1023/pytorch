#include <ATen/native/MatrixExponential.h>

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

namespace {

template <typename T>
constexpr T theta = 3.010066362817634e+00; // deg 18

template<>
constexpr double theta<double> = 1.090863719290036e+00; // deg 18

}

namespace at { namespace native {

void matrix_exp_cuda_kernel(Tensor& res, const Tensor& a) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(),
    "matrix_exp_cuda", [&] {
      using primitive_type = typename scalar_value_type<scalar_t>::type;

      // scale
      const auto norm = at::native::operator_1_norm(a);
      const auto s = at::max(
        at::zeros_like(norm),
        at::ceil(at::log2(norm / theta<primitive_type>))).to(at::kInt);
      const auto pow2s = at::pow(2, s);
      auto a_scaled = a / pow2s.unsqueeze(-1).unsqueeze(-1);

      // square
      auto mexp_scaled = at::native::compute_T18<primitive_type>(a_scaled);
      const auto s_cpu = s.to(at::kCPU);
      for (int64_t i = 0; i < a.size(0); ++i) {
        auto s_val = s_cpu.select(0, i).template item<int>();
        auto mexp = mexp_scaled.select(0, i);
        for (int p = 0; p < s_val; ++p) {
          mexp = at::matmul(mexp, mexp);
        }
        res.select(0, i).copy_(mexp);
      }
    }
  );
}

REGISTER_DISPATCH(matrix_exp_stub, &matrix_exp_cuda_kernel);

}}
