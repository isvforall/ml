import unittest

import numpy as np
import numpy.testing as npt

import task1
import task2
import task3
import task4
import task5
import task6


class Task1(unittest.TestCase):
    def test1(self):
        x = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]])
        self.assertEqual(task1.prod_nonzero_ondiagonal(x), 3)

    def test2(self):
        x = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]])
        self.assertEqual(task1.prod_nonzero_ondiagonal_np(x), 3)

    def test3(self):
        x = np.eye(5)
        self.assertEqual(task1.prod_nonzero_ondiagonal(x), 1)

    def test4(self):
        x = np.eye(5)
        self.assertEqual(task1.prod_nonzero_ondiagonal_np(x), 1)

    def test5(self):
        x = np.array([[5, 6], [6, 5]])
        self.assertEqual(task1.prod_nonzero_ondiagonal(x), 25)

    def test6(self):
        x = np.array([[5, 6], [6, 5]])
        self.assertEqual(task1.prod_nonzero_ondiagonal_np(x), 25)


class Task2(unittest.TestCase):
    def test1(self):
        X = np.array(range(4 * 5)).reshape(4, 5) + 1
        i_idx = np.array([1, 3, 0, 2])
        j_idx = np.array([0, 2, 3, 1])
        answer = [6, 18, 4, 12]
        npt.assert_equal(task2.build_vec(X, i_idx, j_idx), answer)

    def test2(self):
        X = np.array(range(4 * 5)).reshape(4, 5) + 1
        i_idx = np.array([1, 3, 0, 2])
        j_idx = np.array([0, 2, 3, 1])
        answer = [6, 18, 4, 12]
        npt.assert_equal(task2.build_vec_np(X, i_idx, j_idx), answer)

    def test3(self):
        n = 100
        X = np.eye(n)
        i_idx = np.array(range(n))
        j_idx = np.array(range(n))
        answer = np.ones(n, dtype='int')
        npt.assert_equal(task2.build_vec(X, i_idx, j_idx), answer)

    def test4(self):
        n = 100
        X = np.eye(n)
        i_idx = np.array(range(n))
        j_idx = np.array(range(n))
        answer = np.ones(n, dtype='int')
        npt.assert_equal(task2.build_vec_np(X, i_idx, j_idx), answer)


class Task3(unittest.TestCase):
    def test1(self):
        x = np.array([1, 2, 2, 4])
        y = np.array([4, 2, 1, 2])
        self.assertTrue(task3.is_multiset(x, y))

    def test2(self):
        x = np.array([1, 2, 2, 4])
        y = np.array([4, 2, 1, 2])
        self.assertTrue(task3.is_multiset_np(x, y))

    def test3(self):
        x = np.array([1, 1, 2])
        y = np.array([1, 2, 2])
        self.assertFalse(task3.is_multiset(x, y))

    def test5(self):
        x = np.array([1, 1, 2])
        y = np.array([1, 2, 2])
        self.assertFalse(task3.is_multiset_np(x, y))


class Task4(unittest.TestCase):
    def test1(self):
        x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
        self.assertEqual(task4.max_before_zero(x), 5)

    def test2(self):
        x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
        self.assertEqual(task4.max_before_zero_np(x), 5)

    def test3(self):
        x = np.array([0, 0, 10, 11])
        self.assertEqual(task4.max_before_zero(x), 10)

    def test4(self):
        x = np.array([0, 0, 10, 11])
        self.assertEqual(task4.max_before_zero_np(x), 10)


class Task5(unittest.TestCase):
    def test1(self):
        img = np.array([[[131, 177, 78, 255], [125, 175, 74, 255]],
                        [[129, 175, 79, 255], [123, 173, 74, 255]]])
        answer = [[151, 148], [150, 146]]
        npt.assert_equal(task5.make_gray(img), answer)

    def test2_np(self):
        img = np.array([[[131, 177, 78, 255], [125, 175, 74, 255]],
                        [[129, 175, 79, 255], [123, 173, 74, 255]]])
        answer = np.array([[151, 148], [150, 146]])
        npt.assert_equal(task5.make_gray_np(img), answer)


class Task6(unittest.TestCase):
    def test1(self):
        x = np.array([2, 2, 2, 3, 3, 3, 5])
        answer = (np.array([2, 3, 5]), np.array([3, 3, 1]))
        npt.assert_equal(task6.run_length(x), answer)

    def test2(self):
        x = np.array([2, 2, 2, 3, 3, 3, 5])
        answer = (np.array([2, 3, 5]), np.array([3, 3, 1]))
        npt.assert_equal(task6.run_length_np(x), answer)

    def test3(self):
        x = np.array([1, 1, 1, 1, 1])
        answer = (np.array([1]), np.array([5]))
        npt.assert_equal(task6.run_length(x), answer)

    def test4(self):
        x = np.array([1, 1, 1, 1, 1])
        answer = (np.array([1]), np.array([5]))
        npt.assert_equal(task6.run_length_np(x), answer)

    def test5(self):
        x = np.array([1, 2, 3, 4, 5])
        answer = (np.array([1, 2, 3, 4, 5]), np.array([1, 1, 1, 1, 1]))
        npt.assert_equal(task6.run_length(x), answer)

    def test6(self):
        x = np.array([1, 2, 3, 4, 5])
        answer = (np.array([1, 2, 3, 4, 5]), np.array([1, 1, 1, 1, 1]))
        npt.assert_equal(task6.run_length_np(x), answer)


if __name__ == '__main__':
    unittest.main()
