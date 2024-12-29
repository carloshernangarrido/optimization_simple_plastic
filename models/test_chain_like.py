import unittest
import numpy as np
from matplotlib import pyplot as plt
from models.chain_like import Element


class TestElement(unittest.TestCase):
    element_penalty_gap = Element('penalty_gap', 0, 1, {'value': 1, 'contact_stiffness': 10, 'penetration': .1})
    element_gap = Element('gap', 0, 1, {'value': 1, 'contact_stiffness': 10})

    def test_force_ij(self):
        x = np.linspace(-2, 2, 1000)
        fig, ax = plt.subplots(1, 1)
        force_vect_gap = np.vectorize(lambda _: self.element_gap.force_ij([_, 0], [0, 0]))
        force_vect_penalty_gap = np.vectorize(lambda _: self.element_penalty_gap.force_ij([_, 0], [0, 0]))
        ax.plot(x, force_vect_gap(x), label='gap')
        ax.plot(x, force_vect_penalty_gap(x), label='penalty gap')
        ax.grid()
        ax.legend()
        plt.show()

        # self.fail()
    #
    # def test_update_props(self):
    #     self.fail()
    #
    # def test_aliases(self):
    #     self.fail()
    #
    # def test_is_same_as(self):
    #     self.fail()


if __name__ == '__main__':
    unittest.main()
