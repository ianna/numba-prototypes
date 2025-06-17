# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: sealir_basic_compiler
#     language: python
#     name: python3
# ---

# # Ch 3. EGraph Program Rewrites
#
# In this chapter, we’ll walk through implementing our first program rewrite,
# guiding you step-by-step as we transform code using the tools and techniques
# introduced earlier.

from __future__ import annotations

from egglog import EGraph, Unit, function, i64, rewrite, rule, ruleset
from sealir import rvsdg
from sealir.eqsat import rvsdg_eqsat
from sealir.eqsat.rvsdg_eqsat import GraphRoot, Term, TermList

# We'll be extending from chapter 2.
from ch02_egraph_basic import (
    backend,
    frontend,
    jit_compile,
    middle_end,
    run_test,
)
from utils import IN_NOTEBOOK

# Next, we’ll explore a new compiler pipeline designed with customizable rulesets. To enable this flexibility, we’ve introduced a `ruleset` argument, allowing you to tailor the pipeline’s behavior to your specific needs.


def compiler_pipeline(fn, *, verbose=False, ruleset):
    rvsdg_expr, dbginfo = frontend(fn)

    # Middle end
    def define_egraph(egraph: EGraph, func):
        # For now, the middle end is just an identity function that exercise
        # the encoding into and out of egraph.
        root = GraphRoot(func)
        egraph.let("root", root)
        egraph.run(ruleset.saturate())
        if verbose and IN_NOTEBOOK:
            # For inspecting the egraph
            egraph.display(graphviz=True)

    cost, extracted = middle_end(rvsdg_expr, define_egraph)
    print("Extracted from EGraph".center(80, "="))
    print("cost =", cost)
    print(rvsdg.format_rvsdg(extracted))

    llmod = backend(extracted)
    return jit_compile(llmod, extracted)


# ## Rules for defining constants
#
# Now, let’s define a simple rule by specifying what makes a constant boolean.
# We’ll use `egglog.function` to annotate properties on terms (`Term`
# instances). Each term directly corresponds to an RVSDG-IR node, which in turn
# maps to a Python AST node. As a result, a term can represent various
# constructs—such as an expression, a literal constant, an operation, or a
# control-flow element.
#
# An `egglog.function` acts as a symbolic entity, meaning it doesn’t require a
# function body. In our case, we’ll use it to mark specific terms: a term is
# labeled as `IsConstantTrue(Term)` if it represents an expression of a non-zero
# literal int64, indicating a constant `True`. Conversely, we mark a term as
# `IsConstantFalse(Term)` if it’s an expression of a literal zero, signifying a
# constant `False`.
#


# +
@function
def IsConstantTrue(t: Term) -> Unit: ...


@function
def IsConstantFalse(t: Term) -> Unit: ...


# -


# Rules can be organized into groups known as `ruleset`. Below, we’ll define a
# set of rules for recognizing constants, laying the groundwork for our
# optimization process.


@ruleset
def ruleset_const_propagate(a: Term, ival: i64):
    # a Literal Int64 is constant True if it's non-zero
    yield rule(
        # Given a LiteralI64 where the integer-value is non zero
        a == Term.LiteralI64(ival),
        ival != 0,
    ).then(
        # Setup the following fact
        IsConstantTrue(a)
    )
    # a Literal Int64 is constant False if it's zero
    yield rule(
        # Given a LiteralI64 where the integer-value is zero
        a == Term.LiteralI64(ival),
        ival == 0,
    ).then(
        # Setup the following fact
        IsConstantFalse(a)
    )


# Now, we’ll test our newly defined ruleset. This complete ruleset combines a
# few built-in RVSDG rules with our recently crafted simple constant-propagation
# rules.

if __name__ == "__main__":

    def ifelse_fold(a, b):
        c = 0
        if c:
            return a
        else:
            return b

    # Add our const-propagation rule to the basic rvsdg ruleset
    my_ruleset = rvsdg_eqsat.ruleset_rvsdg_basic | ruleset_const_propagate

    jt = compiler_pipeline(ifelse_fold, verbose=True, ruleset=my_ruleset)
    run_test(ifelse_fold, jt, (12, 34))


# In the example above, observe that a new `IsConstantFalse` node appears on the
# `LiteralI64(0)`. This indicates that our ruleset is successfully identifying
# constants as intended.

# ## Rules for folding if-else
#
# Now, let’s create a more complex rule. This time, we’ll fold an if-else
# expression when its condition is a constant, simplifying the code by resolving
# the branch at compile time.


@ruleset
def ruleset_const_fold_if_else(a: Term, b: Term, c: Term, operands: TermList):
    yield rewrite(
        # Define the if-else pattern to match
        Term.IfElse(cond=a, then=b, orelse=c, operands=operands),
        subsume=True,  # subsume to disable extracting the original term
    ).to(
        # Define the target expression
        # This apply region `b` (then) using the `operands`.
        Term.Apply(b, operands),
        # Given that the condition is constant True
        IsConstantTrue(a),
    )
    yield rewrite(
        # Define the if-else pattern to match
        Term.IfElse(cond=a, then=b, orelse=c, operands=operands),
        subsume=True,  # subsume to disable extracting the original term
    ).to(
        # Define the target expression.
        # This apply region `c` (orelse) using the `operands`.
        Term.Apply(c, operands),
        # Given that the condition is constant False
        IsConstantFalse(a),
    )


if __name__ == "__main__":
    my_ruleset = (
        rvsdg_eqsat.ruleset_rvsdg_basic
        | ruleset_const_propagate
        | ruleset_const_fold_if_else  # <-- the new rule for if-else
    )

    jt = compiler_pipeline(ifelse_fold, verbose=True, ruleset=my_ruleset)
    run_test(ifelse_fold, jt, (12, 34))

# After applying the rewrite, the RVSDG simplifies dramatically, leaving a
# nearly empty function body. The `!ret` instruction is now hardcoded to
# `$0[2]`, which represents the variable `b`, aligning with the `return b` from
# the `else` branch.
#
# Meanwhile, the EGraph becomes more intriguing, with numerous nodes merged to
# reflect their equivalence. For example, the `Term.Apply` and `Term.IfElse`
# nodes are now combined, showcasing how the rewrite consolidates equivalent
# expressions.
