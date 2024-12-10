import pytest
import numpy as np

from po_minigrid.core.particles import Particles


def test_initialization():
    particles = Particles()
    assert hasattr(particles, "_pose")
    assert hasattr(particles, "_carrying")
    assert hasattr(particles, "_toggled")
    assert hasattr(particles, "_reset_pose")
    assert hasattr(particles, "_is_belief")
    assert particles.is_belief is True


def test_pose_initialization():
    particles = Particles(pose=None)
    np.testing.assert_array_equal(particles.pose, np.array([[-1, -1, 0]]))

    custom_pose = np.array([[1, 2, 3]])
    particles = Particles(pose=custom_pose)
    np.testing.assert_array_equal(particles.pose, custom_pose)


def test_is_belief_initialization():
    particles = Particles(is_belief=False)
    assert not particles.is_belief


def test_pose_getter():
    custom_pose = np.array([[1, 2, 3]])
    particles = Particles(pose=custom_pose)
    np.testing.assert_array_equal(particles.pose, custom_pose)


def test_pose_setter():
    particles = Particles()
    new_pose = np.array([[4, 5, 6]])
    particles.pose = new_pose
    np.testing.assert_array_equal(particles.pose, new_pose)


def test_pose_setter_ndim():
    particles = Particles(pose=np.arange(9).reshape(3, 3))
    np.testing.assert_array_equal(
        particles.pose,
        np.array(
            [
                [
                    0,
                    1,
                    2,
                ],
                [3, 4, 5],
                [6, 7, 8],
            ]
        ),
    )

    particles.pose = None
    np.testing.assert_array_equal(
        particles.pose, np.array([[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]])
    )


def test_pose_setter_specific_indices():
    particles = Particles(pose=np.arange(9).reshape(3, 3))
    new_pose_index1 = np.array([90, 90, 3])

    particles.pose[1] = new_pose_index1
    np.testing.assert_array_equal(
        particles.pose,
        np.array(
            [
                [
                    0,
                    1,
                    2,
                ],
                [90, 90, 3],
                [6, 7, 8],
            ]
        ),
    )


def test_pose_setter_none():
    particles = Particles()
    particles.pose = np.array([[4, 5, 6]])

    particles.pose = None
    np.testing.assert_array_equal(particles.pose, np.array([[-1, -1, 0]]))


def test_pose_setter_shape_mismatch():
    particles = Particles(pose=np.array([[1, 2, 3]]))
    with pytest.raises(AssertionError):
        particles.pose = np.array([[1, 2]])


def test_pose_setter_dtype_mismatch():
    particles = Particles(pose=np.array([[1, 2, 3]], dtype=np.int64))
    with pytest.raises(AssertionError):
        particles.pose = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)


# def test_carrying_initialization():
#     # Test that the carrying property initializes correctly
#     particles = Particles()
#     assert particles.carrying is not None
#     np.testing.assert_array_equal(particles.carrying, np.array([None]))


# def test_carrying_setter():
#     # Test setting a valid carrying array
#     particles = Particles()
#     new_carrying = np.array([dict()])
#     particles.carrying = new_carrying
#     np.testing.assert_array_equal(particles.carrying, new_carrying)


# def test_carrying_setter_none():
#     # Test that setting carrying to None initializes it correctly
#     particles = Particles()
#     particles.carrying = None
#     np.testing.assert_array_equal(particles.carrying, np.array([None]))


# def test_carrying_setter_shape_mismatch():
#     # Test shape mismatch when setting carrying
#     particles = Particles()
#     with pytest.raises(AssertionError):
#         particles.carrying = np.array([1, 2, 3])  # Initial valid state
