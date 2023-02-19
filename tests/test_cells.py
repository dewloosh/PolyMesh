import numpy as np
import unittest

from polymesh.cells import H8, TET10


class TestGeneratedExpressions(unittest.TestCase):

    def test_generated_H8(self):
        pcoords = H8.lcoords()
        shpf = H8.shape_function_values
        shpmf = H8.shape_function_matrix
        dshpf = H8.shape_function_derivatives
        _shpf, _shpmf, _dshpf= H8.generate(return_symbolic=False)
        self.assertTrue(np.all(np.isclose(_shpf(pcoords), shpf(pcoords))))
        self.assertTrue(np.all(np.isclose(_dshpf(pcoords), dshpf(pcoords))))
        self.assertTrue(np.all(np.isclose(_shpmf(pcoords), shpmf(pcoords))))
        
    def test_generated_TET10(self):
        pcoords = TET10.lcoords()
        shpf = TET10.shape_function_values
        shpmf = TET10.shape_function_matrix
        dshpf = TET10.shape_function_derivatives
        _shpf, _shpmf, _dshpf= TET10.generate(return_symbolic=False)
        self.assertTrue(np.all(np.isclose(_shpf(pcoords), shpf(pcoords))))
        self.assertTrue(np.all(np.isclose(_dshpf(pcoords), dshpf(pcoords))))
        self.assertTrue(np.all(np.isclose(_shpmf(pcoords), shpmf(pcoords))))
        

if __name__ == "__main__":

    unittest.main()