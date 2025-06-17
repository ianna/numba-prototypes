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

# # Ch 7. MLIR ufunc operations
#

# In this chapter, we'll look at type inference for array operations.

from __future__ import annotations

import inspect

import mlir.dialects.func as func
import mlir.dialects.linalg as linalg
import mlir.ir as ir
import numpy as np

from ch04_2_typeinfer_loops import (
    Compiler,
    MyCostModel,
    setup_argtypes,
)
from ch05_typeinfer_array import (
    NbOp_ArrayDimSymbolic,
    NbOp_ArrayType,
    Type,
    base_ruleset,
)
from ch06_mlir_backend import Backend as _Backend
from ch06_mlir_backend import ConditionalExtendGraphtoRVSDG, NbOp_Type

# Type declaration for array elements
Float64 = NbOp_Type("Float64")
TypeFloat64 = Type.simple("Float64")
Float32 = NbOp_Type("Float32")
TypeFloat32 = Type.simple("Float32")


class Backend(_Backend):
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


# Decorator function for vecotrization.
def ufunc_vectorize(input_type, ndim, ufunc_compiler, extra_ruleset=None):
    def to_input_dtypes(ty):
        if ty == Float64:
            return TypeFloat64
        elif ty == Float32:
            return TypeFloat32

    def wrapper(inner_func):
        sig = inspect.signature(inner_func)
        num_inputs = len(sig.parameters)
        ruleset = base_ruleset | setup_argtypes(
            *(to_input_dtypes(input_type),) * num_inputs
        )
        if extra_ruleset is not None:
            ruleset |= extra_ruleset
        # Compile the inner function and get the IR as a module.
        llmod, func_egraph = ufunc_compiler.lower_py_fn(
            inner_func,
            argtypes=(input_type,) * num_inputs,
            ruleset=ruleset,
        )

        # Now within the module declare a seperate function named
        # 'ufunc' which acts as a wrapper around the innner 'func'
        with (
            llmod.context,
            ir.Location.unknown(context=llmod.context),
            ir.InsertionPoint(llmod.body),
        ):
            f32 = ir.F32Type.get()
            f64 = ir.F64Type.get()

            match input_type.name:
                case "Float32":
                    internal_dtype = f32
                case "Float64":
                    internal_dtype = f64
                case _:
                    raise TypeError("The current input type is not supported")

            dynsize = ir.ShapedType.get_dynamic_size()
            memref_ty = ir.MemRefType.get([dynsize] * ndim, internal_dtype)

            # The function 'ufunc' has N + 1 number of arguments
            # (where N is the nuber of arguments for the original function)
            # The extra argument is an explicitly declared resulting array.
            input_typ_outer = (memref_ty,) * (num_inputs + 1)

            fun = func.FuncOp("ufunc", (input_typ_outer, ()))
            fun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            const_block = fun.add_entry_block()
            constant_entry = ir.InsertionPoint(const_block)

            # Within this function we declare the symbolic representation of
            # input and output arrays of appropriate shapes using memrefs.
            with constant_entry:
                arys = fun.arguments[:-1]
                res = fun.arguments[-1]

                # Affine map declaration
                indexing_maps = ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            ir.AffineMap.get(
                                ndim,
                                0,
                                [
                                    ir.AffineExpr.get_dim(i)
                                    for i in range(ndim)
                                ],
                            )
                        ),
                    ]
                    * (num_inputs + 1)
                )
                iterators = ir.ArrayAttr.get(
                    [ir.Attribute.parse(f"#linalg.iterator_type<parallel>")]
                    * (num_inputs)
                )
                matmul = linalg.GenericOp(
                    result_tensors=[],
                    inputs=arys,
                    outputs=[res],
                    indexing_maps=indexing_maps,
                    iterator_types=iterators,
                )
                # Within the affine loop body make calls to the inner function.
                body = matmul.regions[0].blocks.append(
                    *([internal_dtype] * (num_inputs + 1))
                )
                with ir.InsertionPoint(body):
                    m = func.CallOp(
                        [internal_dtype], "func", [*body.arguments[:-1]]
                    )
                    linalg.YieldOp([m])
                func.ReturnOp([])

        ufunc_compiler.run_backend_passes(llmod)

        jit_func = ufunc_compiler.compile_module_(
            llmod,
            [memref_ty] * num_inputs,
            (memref_ty,),
            "ufunc",
            is_ufunc=True,
        )

        def call_wrapper(*args, out=None):
            if isinstance(memref_ty.element_type, ir.F64Type):
                np_dtype = np.float64
            elif isinstance(memref_ty.element_type, ir.F32Type):
                np_dtype = np.float32
            else:
                raise TypeError(
                    "The current array element type is not supported"
                )
            out_shape = np.broadcast(*args).shape
            out = np.zeros(out_shape, dtype=np_dtype) if out is None else out
            return jit_func(*args, out)

        return call_wrapper

    return wrapper


if __name__ == "__main__":
    compiler = Compiler(
        ConditionalExtendGraphtoRVSDG, Backend(), MyCostModel(), True
    )

    @ufunc_vectorize(input_type=Float64, ndim=2, ufunc_compiler=compiler)
    def foo(a, b, c):
        x = a + 1.0
        y = b - 2.0
        z = c + 3.0
        return x + y + z

    # Create NumPy arrays
    ary = np.arange(100, dtype=np.float64).reshape(10, 10)
    ary_2 = np.arange(100, dtype=np.float64).reshape(10, 10)
    ary_3 = np.arange(100, dtype=np.float64).reshape(10, 10)

    got = foo(ary, ary_2, ary_3)
    print("Got", got)
