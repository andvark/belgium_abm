import unittest

import numpy as np

from belgium_abm.rng import RngFactory


class RngTests(unittest.TestCase):
    def test_same_seed_and_name_gives_identical_stream(self) -> None:
        a = RngFactory(7).stream("x").random(5)
        b = RngFactory(7).stream("x").random(5)
        np.testing.assert_array_equal(a, b)

    def test_names_and_seeds_give_different_streams(self) -> None:
        f = RngFactory(7)
        self.assertFalse(np.array_equal(f.stream("x").random(5), f.stream("y").random(5)))
        self.assertFalse(np.array_equal(RngFactory(8).stream("x").random(5), f.stream("x").random(5)))

    def test_streams_are_independent_of_creation_order(self) -> None:
        f1, f2 = RngFactory(3), RngFactory(3)
        f1.stream("a").random(100)
        np.testing.assert_array_equal(f1.stream("b").random(3), f2.stream("b").random(3))

    def test_negative_seed_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RngFactory(-1)


if __name__ == "__main__":
    unittest.main()
