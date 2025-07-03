import demo02_cuda_ufunc
from demo02_cuda_ufunc import *

from .autotests import autotest_notebook


def test_demo02_autotest():
    autotest_notebook(demo02_cuda_ufunc)
