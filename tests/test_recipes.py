import unittest

from polymesh.recipes import circular_disk, ribbed_plate


class TestMeshing(unittest.TestCase):

    def test_recipes(self):
        circular_disk(120, 60, 5, 25)
        ribbed_plate(lx=5.0, ly=5.0, t=1.0)
        ribbed_plate(lx=5.0, ly=5.0, t=1.0, 
                     wx=1.0, hx=2.0, ex=0.05,
                     wy=1.0, hy=2.0, ey=-0.05)
        

if __name__ == "__main__":

    unittest.main()