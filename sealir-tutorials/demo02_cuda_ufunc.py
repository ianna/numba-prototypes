# # Demo 2: CUDA backend for Tanh Approximation for GELU activation layer
#
# (Depends on Ch.08)
from numba import cuda
from ch08_gpu_offload import GPUBackend
from demo01_gelu_tanh_approx import *


class GpuUfuncBackend(Backend, GPUBackend):
    # Ufunc + GPU backend
    pass


gpu_compiler = Compiler(
    ExtendEGraphToRVSDG, GpuUfuncBackend(), MyCostModel(), True
)

if cuda.is_available():
    cuda_vectorized_gelu = ufunc_vectorize(
        input_type=Float32,
        ndim=1,
        ufunc_compiler=gpu_compiler,
        extra_ruleset=additional_rules,
    )(gelu_tanh_forward)


if __name__ == "__main__":
    if not cuda.is_available():
        print("SKIPPED. CUDA unavailable")
    else:
        relclose = lambda x, y: np.allclose(x, y, rtol=1e-6)
        input_val = np.random.random(100).astype(np.float32)
        run_test(
            gelu_tanh_forward,
            cuda_vectorized_gelu,
            (input_val,),
            equal=relclose,
            verbose=True,
        )
