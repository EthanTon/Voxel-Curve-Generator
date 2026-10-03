from types import SimpleNamespace
import numpy as np
from sandbox.terraform_test.check import drops


def test_a_changed_cliff_that_is_no_steeper_than_the_save_is_accepted():
    surfaces = SimpleNamespace(finished=np.array([[61., 65.]]),
        natural=np.array([[60., 65.]]), wet=np.zeros((1, 2), dtype=bool),
        track=np.zeros((1, 2), dtype=bool), x0=0, z0=0)
    assert not drops(surfaces, lambda x, z, untouched: False, 2, rock=False, natural=True)
    assert not drops(surfaces, lambda x, z, untouched: False, 2, rock=False, natural=False)


def test_a_new_steeper_face_is_still_reported():
    surfaces = SimpleNamespace(finished=np.array([[60., 66.]]),
        natural=np.array([[60., 65.]]), wet=np.zeros((1, 2), dtype=bool),
        track=np.zeros((1, 2), dtype=bool), x0=0, z0=0)
    assert len(drops(surfaces, lambda x, z, untouched: False, 2, rock=False, natural=True)) == 1
