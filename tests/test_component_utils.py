import torch

from spd.utils.component_utils import rand_perm


def test_rand_perm_creates_correct_shape():
    shape = (2, 3)
    perm = rand_perm(shape, dim=1)
    assert perm.shape == shape


def test_rand_perm_creates_permutations():
    # Given a random permutation of shape (100, 200), along dim 1
    shape = (100, 200)
    perm = rand_perm(shape, dim=1)

    # when we sort along dim 1
    sorted_perm = perm.sort(dim=1).values

    # then, we should get a simple arange for each row
    sorted_target = torch.arange(200).repeat(100, 1)
    assert torch.equal(sorted_perm, sorted_target)


def test_rand_perm_creates_permutations_along_correct_indices():
    # Given a random permutation of shape (100, 200), along dim 1
    shape = (100, 200)
    perm = rand_perm(shape, dim=1)

    # when we sort along dim 0 (not dim 1)
    sorted_perm_wrong_axis = perm.sort(dim=0).values

    # then we should NOT get a simple arange for each row (technically it's possible, but
    # the probability is 1/(100!) which is ≈never)
    sorted_target_wrong_axis = torch.arange(100).repeat(200, 1)
    assert not torch.equal(sorted_perm_wrong_axis, sorted_target_wrong_axis)
