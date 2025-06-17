from sealir import ase

import ch05_typeinfer_array
from ch05_typeinfer_array import *

from .autotests import autotest_notebook


def test_ch05_autotest():
    autotest_notebook(ch05_typeinfer_array)
