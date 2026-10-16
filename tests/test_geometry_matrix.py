"""Real swept voxel output across path, section and terrain families."""

import pytest

from tests.geometry_matrix import (
    ENVIRONMENTS, check_build, check_environment, profiles, routes, transformed, write_profiles,
)
from src.Terraform import Look


@pytest.fixture(scope="module")
def profile_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("geometry_profiles")
    write_profiles(directory)
    return directory


@pytest.mark.parametrize("route", routes(), ids=lambda r: r.name)
@pytest.mark.parametrize("shape", profiles(), ids=lambda p: p.name)
def test_path_cross_section_matrix(profile_files, route, shape, request):
    result, _, _, _ = check_build(profile_files, route, (shape,))
    if route.name == "figure_eight" and shape.name in (
        "offset_side_6", "asymmetric_left_L", "asymmetric_right_L"
    ):
        request.node.add_marker(pytest.mark.xfail(
            reason="Nearest-station frame clipping disconnects offset material or the raised wall at a self-crossing",
            strict=True,
        ))
    assert not result["issues"], result


@pytest.mark.parametrize("route", routes(), ids=lambda r: r.name)
@pytest.mark.parametrize("shape", [p for p in profiles() if p.name in (
    "asymmetric_left_L", "asymmetric_right_L")], ids=lambda p: p.name)
def test_asymmetric_walls_with_fitting_joins(profile_files, route, shape, request):
    result, _, _, _ = check_build(profile_files, route, (shape,), ridden=True)
    if route.name == "figure_eight":
        request.node.add_marker(pytest.mark.xfail(
            reason="Nearest-station frame clipping splits the raised wall at a same-height crossing",
            strict=True,
        ))
    assert not result["issues"], result


@pytest.mark.parametrize("route", [r for r in routes() if r.name in (
    "straight_1_0", "arc_R25_cw_90", "S_bend_R40", "arc_R35_ccw_180", "uphill")], ids=lambda r: r.name)
@pytest.mark.parametrize("ridden", (False, True), ids=("corners", "fitting_joins"))
def test_profile_transitions_and_fitting_join_modes(profile_files, route, ridden):
    shapes = tuple(p for p in profiles() if p.name in ("symmetric_flat_5", "asymmetric_left_L", "hollow_tunnel_9x7"))
    result, _, _, _ = check_build(profile_files, route, shapes, transitions=True, ridden=ridden)
    assert not result["issues"], result


@pytest.mark.parametrize("name", ENVIRONMENTS)
@pytest.mark.parametrize("shape", [p for p in profiles() if p.name in (
    "wide_deck_19", "asymmetric_left_L", "asymmetric_right_L", "hollow_tunnel_9x7")], ids=lambda p: p.name)
@pytest.mark.parametrize("route", [r for r in routes() if r.name in ("straight_1_0", "S_bend_R10")], ids=lambda r: r.name)
def test_environment_matrix(profile_files, route, shape, name):
    result, _, _ = check_environment(profile_files, route, shape, name)
    assert not result["issues"], result


@pytest.mark.parametrize("look", (Look.NATURAL, Look.MINIMAL), ids=lambda l: l.value)
@pytest.mark.parametrize("name", ("rolling_hills", "river", "left_cliff", "right_cliff"))
def test_additional_terrain_styles(profile_files, name, look):
    route = next(r for r in routes() if r.name == "S_bend_R10")
    shape = next(p for p in profiles() if p.name == "asymmetric_left_L")
    result, _, _ = check_environment(profile_files, route, shape, name, look)
    assert not result["issues"], result


@pytest.mark.parametrize("shape", profiles(), ids=lambda p: p.name)
@pytest.mark.parametrize("shift", ((-400, -96, -700), (300000, 180, -300000)), ids=("negative", "distant"))
def test_translating_the_route_preserves_every_voxel_and_material(profile_files, shape, shift):
    route = next(r for r in routes() if r.name == "arc_R25_cw_90")
    _, _, _, original = check_build(profile_files, route, (shape,))
    _, _, _, moved = check_build(profile_files, transformed(route, shift=shift), (shape,))
    assert moved == {tuple(p[i]+shift[i] for i in range(3)): state for p, state in original.items()}
