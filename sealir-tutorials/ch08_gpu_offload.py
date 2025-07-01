# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Chapter 8: GPU Offloading for MLIR Ufunc Operations
#
# This chapter extends the MLIR ufunc system to support GPU offloading using
# CUDA. We show how to compile and execute vectorized operations (ufunc) on
# NVIDIA GPUs, demonstrating the full pipeline from Python functions to GPU
# kernels.
#
# The chapter covers:
# - How to extend the MLIR backend for GPU compilation
# - How to use CUDA-specific MLIR passes for kernel generation
# - How to handle GPU memory management and data transfer

# ## Imports and Setup

from __future__ import annotations

import ctypes
import os
from collections import namedtuple
from ctypes.util import find_library

import mlir.execution_engine as execution_engine
import mlir.ir as ir
import mlir.passmanager as passmanager
import mlir.runtime as runtime
import numpy as np
from numba import cuda

from ch05_typeinfer_array import NbOp_ArrayType
from ch06_mlir_backend import Backend as _Backend
from ch06_mlir_backend import NbOp_Type
from ch07_mlir_ufunc import Float64, compiler_config, ufunc_vectorize
from utils import IN_NOTEBOOK

# ## CUDA Environment Setup
#
# Configure CUDA environment variables for proper GPU compilation.

# Requires the CUDA toolkit.
# If using `conda install cuda`, set `CUDA_HOME=$CONDA_PREFIX`
if "CUDA_HOME" not in os.environ and "CONDA_PREFIX" in os.environ:
    os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]

# ## GPU Backend Implementation
#
# Extend the MLIR backend to support GPU compilation and execution.

_DEBUG = False


class GPUBackend(_Backend):
    compile_only: bool

    def __init__(self, compile_only: bool = False):
        super().__init__()
        self.compile_only = compile_only

    # Lower symbolic array to respective memref.
    # Note: This is not used within ufunc builder,
    # since it has explicit declaration of the respective
    # MLIR memrefs.
    def lower_type(self, ty: NbOp_Type):
        match ty:
            case NbOp_ArrayType(
                dtype=dtype,
                ndim=int(ndim),
                datalayout=str(datalayout),
                shape=shape,
            ):
                mlir_dtype = self.lower_type(dtype)
                with self.loc:
                    memref_ty = ir.MemRefType.get(shape, mlir_dtype)
                return memref_ty
        return super().lower_type(ty)

    def run_passes(self, module):
        """GPU Pass Pipeline

        Define the MLIR pass pipeline for GPU compilation, including
        parallelization, kernel outlining, and NVVM code generation.
        """
        if _DEBUG:
            module.dump()
        pass_man = passmanager.PassManager(context=module.context)

        if _DEBUG:
            module.context.enable_multithreading(False)
        if _DEBUG and not IN_NOTEBOOK:
            # notebook may hang if ir_printing is enabled and and MLIR failed.
            pass_man.enable_ir_printing()

        pass_man.add("convert-linalg-to-affine-loops")
        pass_man.add("affine-loop-fusion")
        pass_man.add("inline")
        pass_man.add("func.func(affine-parallelize)")
        pass_man.add(
            "builtin.module(func.func(gpu-map-parallel-loops,convert-parallel-loops-to-gpu))"
        )
        pass_man.add("lower-affine")
        pass_man.add("scf-parallel-loop-fusion")
        pass_man.add(
            "func.func(gpu-map-parallel-loops,convert-parallel-loops-to-gpu)"
        )
        pass_man.add("gpu-kernel-outlining")
        if not self.compile_only:
            pass_man.add('gpu-lower-to-nvvm-pipeline{cubin-format="fatbin"}')
        pass_man.add("convert-scf-to-cf")
        pass_man.add("finalize-memref-to-llvm")
        pass_man.add("convert-math-to-libm")
        pass_man.add("convert-func-to-llvm")
        pass_man.add("convert-index-to-llvm")
        pass_man.add("convert-bufferization-to-memref")
        pass_man.add("reconcile-unrealized-casts")
        pass_man.add("func.func(llvm-request-c-wrappers)")
        pass_man.enable_verifier(True)
        pass_man.run(module.operation)
        # Output LLVM-dialect MLIR
        if _DEBUG:
            module.dump()
        return module

    @classmethod
    def get_exec_ptr(cls, mlir_ty, val):
        """Get Execution Pointer

        Convert MLIR types to C-types and allocate memory for the value.
        """
        if isinstance(mlir_ty, ir.IntegerType):
            val = 0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_int64(val))
        elif isinstance(mlir_ty, ir.F32Type):
            val = 0.0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_float(val))
        elif isinstance(mlir_ty, ir.F64Type):
            val = 0.0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_double(val))
        elif isinstance(mlir_ty, ir.MemRefType):
            if isinstance(mlir_ty.element_type, ir.F64Type):
                np_dtype = np.float64
            elif isinstance(mlir_ty.element_type, ir.F32Type):
                np_dtype = np.float32
            else:
                raise TypeError(
                    "The current array element type is not supported"
                )
            val = (
                np.zeros(mlir_ty.shape, dtype=np_dtype) if val is None else val
            )
            val = cls.np_arr_to_np_duck_device_arr(val)
            ptr = ctypes.pointer(
                ctypes.pointer(runtime.get_ranked_memref_descriptor(val))
            )

        return ptr, val

    @classmethod
    def get_out_val(cls, res_ptr, res_val):
        if isinstance(res_val, cuda.cudadrv.devicearray.DeviceNDArray):
            return res_val.copy_to_host()
        else:
            return super().get_out_val(res_ptr, res_val)

    @classmethod
    def np_arr_to_np_duck_device_arr(cls, arr):
        da = cuda.to_device(arr)
        ctlie = namedtuple("ctypes_lie", "data data_as shape")
        da.ctypes = ctlie(
            da.__cuda_array_interface__["data"][0],
            lambda x: ctypes.cast(da.ctypes.data, x),
            da.__cuda_array_interface__["shape"],
        )
        da.itemsize = arr.itemsize
        return da

    def jit_compile_extra(
        self,
        llmod,
        input_types,
        output_types,
        function_name="func",
        exec_engine=None,
        is_ufunc=False,
        **execution_engine_params,
    ):
        """JIT Compilation

        Convert the MLIR module into a JIT-callable function using the MLIR
        execution engine.
        """
        if self.compile_only:
            return None
        cuda_libs = (
            "mlir_cuda_runtime",
            "mlir_c_runner_utils",
            "mlir_runner_utils",
        )
        cuda_shared_libs = [find_library(x) for x in cuda_libs]
        return super().jit_compile_extra(
            llmod,
            input_types,
            output_types,
            function_name,
            is_ufunc=is_ufunc,
            exec_engine=execution_engine.ExecutionEngine(
                llmod, opt_level=3, shared_libs=cuda_shared_libs
            ),
            **execution_engine_params,
        )


# ## GPU Compiler Configuration
#
# Set up the GPU compiler configuration with fallback to CPU if CUDA is
# unavailable.

gpu_compiler_config = {
    **compiler_config,
    "backend": GPUBackend(compile_only=not cuda.is_available()),
}

# ## Example: GPU Ufunc Function
#
# Demonstrate GPU offloading with a simple ufunc function that performs
# element-wise operations on GPU.

if __name__ == "__main__":

    @ufunc_vectorize(
        input_type=Float64, ndim=2, compiler_config=gpu_compiler_config
    )
    def foo(a, b, c):
        x = a + 1.0
        y = b - 2.0
        z = c + 3.0
        return x + y + z

    # Create NumPy arrays
    ary = np.arange(100, dtype=np.float64).reshape(10, 10)
    ary_2 = np.arange(100, dtype=np.float64).reshape(10, 10)
    ary_3 = np.arange(100, dtype=np.float64).reshape(10, 10)

    if not cuda.is_available():
        print("SKIPPED. CUDA unavailable")
    else:
        got = foo(ary, ary_2, ary_3)
        print("Got", got)
