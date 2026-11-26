"""Schema 1.5 forced-dynamics declaration checks."""

import copy
import math

import numpy as np
import pytest

from anf.dynamics import (HarmonicArtifactError, HarmonicResult, HarmonicSolveError,
                          NewmarkSolveError, TransientArtifactError, TransientResult,
                          assemble_mass, harmonic_response, load_harmonic_result,
                          load_transient_result, save_harmonic_result,
                          save_transient_result, transient_response)
from anf.model import (Damping, ModelValidationError, canonical_dynamics_input_hash,
                       validate_dynamics_model)

UNITS = {"length": "m", "force": "N", "stress": "Pa"}
E, RHO, A, I, L = 210e9, 7850.0, 0.01, 1e-6, 2.0


def frame(analysis, loads=None, *, elements=16, nodes=None, supports=None, members=None):
    return {
        "schema_version": "1.5",
        "problem_type": "planar_structure",
        "units": dict(UNITS),
        "model": {
            "nodes": nodes if nodes is not None else {"base": [0.0, 0.0], "tip": [L, 0.0]},
            "members": members if members is not None else {
                "beam": {"kind": "frame2d", "i": "base", "j": "tip", "elements": elements,
                         "young_modulus": E, "area": A, "second_moment": I, "density": RHO}},
            "supports": (supports if supports is not None
                         else {"base": {"ux": 0, "uy": 0, "rz": 0}}),
        },
        "loads": {} if loads is None else loads,
        "analysis": analysis,
    }


def truss(analysis, loads=None):
    return frame(analysis, loads, nodes={"a": [0.0, 0.0], "b": [L, 0.0]},
                 members={"rod": {"kind": "truss2d", "i": "a", "j": "b",
                                  "young_modulus": E, "area": A, "density": RHO}},
                 supports={"a": {"ux": 0, "uy": 0}, "b": {"uy": 0}})


def harmonic(**overrides):
    block = {"type": "harmonic", "frequencies_hz": [0.0, 7.2358, 20.0],
             "damping": {"kind": "rayleigh", "alpha": 1.720507, "beta": 4.743437e-5}}
    block.update(overrides)
    return block


def transient(**overrides):
    block = {"type": "transient",
             "time": {"start_s": 0.0, "end_s": 0.05, "steps": 500},
             "damping": {"kind": "rayleigh", "alpha": 0.0, "beta": 0.0}}
    block.update(overrides)
    return block


def code_of(doc):
    with pytest.raises(ModelValidationError) as info:
        validate_dynamics_model(doc)
    return info.value.code, info.value.path


def test_harmonic_normalizes_options_and_complex_loads():
    spec = validate_dynamics_model(frame(
        harmonic(), {"nodal": {"tip": {"fy": {"amplitude": 1000.0, "phase_deg": 90.0},
                                       "mz": 5.0}}}))
    assert spec.analysis_type == "harmonic"
    assert spec.modes is None
    assert spec.dynamics.kind == "harmonic"
    assert spec.dynamics.frequencies_hz == (0.0, 7.2358, 20.0)
    assert spec.dynamics.damping == Damping(1.720507, 4.743437e-5)
    nodal = spec.loads["nodal"]
    assert dict(nodal["tip"]["fy"]) == {"amplitude": 1000.0, "phase_deg": 90.0}
    assert dict(nodal["tip"]["mz"]) == {"amplitude": 5.0, "phase_deg": 0.0}
    assert spec.to_dict()["analysis"]["damping"] == {
        "kind": "rayleigh", "alpha": 1.720507, "beta": 4.743437e-5}


def test_zero_amplitude_loads_are_dropped():
    spec = validate_dynamics_model(frame(harmonic(), {"nodal": {"tip": {"fy": 0.0}}}))
    assert dict(spec.loads["nodal"]) == {}


def test_damping_defaults_to_zero():
    spec = validate_dynamics_model(frame({"type": "harmonic", "frequencies_hz": [1.0]}))
    assert spec.dynamics.damping == Damping(0.0, 0.0)


def test_transient_grid_materializes_uniform_times():
    spec = validate_dynamics_model(frame(transient()))
    grid = spec.dynamics.time
    assert grid.steps == 500
    assert grid.dt_s == pytest.approx(1e-4)
    assert len(grid.times_s) == 501
    assert grid.times_s[0] == 0.0
    assert grid.times_s[-1] == pytest.approx(0.05)
    assert dict(spec.dynamics.initial_displacement) == {}
    assert dict(spec.dynamics.initial_velocity) == {}


def test_times_array_form_round_trips():
    spec = validate_dynamics_model(frame(transient(time={"times_s": [0.0, 0.001, 0.002]})))
    grid = spec.dynamics.time
    assert grid.times_s == (0.0, 0.001, 0.002)
    assert grid.steps == 2
    assert spec.to_dict()["analysis"]["time"] == {
        "start_s": 0.0, "end_s": 0.002, "steps": 2}


def test_initial_state_is_normalized():
    spec = validate_dynamics_model(frame(transient(initial_state={
        "displacement": {"tip": {"uy": 0.001, "ux": 0.0}},
        "velocity": {"tip": {"rz": 0.2}},
    })))
    assert dict(spec.dynamics.initial_displacement["tip"]) == {"uy": 0.001}
    assert dict(spec.dynamics.initial_velocity["tip"]) == {"rz": 0.2}
    assert spec.to_dict()["analysis"]["initial_state"]["displacement"] == {
        "tip": {"uy": 0.001}}


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_negative_damping_rejected(name):
    code, path = code_of(frame(harmonic(damping={"kind": "rayleigh", name: -1.0})))
    assert (code, path) == ("negative_damping", f"/analysis/damping/{name}")


def test_unsupported_damping_kind_rejected():
    assert code_of(frame(harmonic(damping={"kind": "modal"})))[0] == "unsupported_damping"


@pytest.mark.parametrize("frequencies, code", [
    ([1.0, 0.5], "frequency_not_increasing"),
    ([1.0, 1.0], "frequency_not_increasing"),
    ([-1.0], "negative_frequency"),
    ([], "invalid_value"),
    (list(range(201)), "frequency_budget_exceeded"),
])
def test_frequency_contract(frequencies, code):
    assert code_of(frame(harmonic(frequencies_hz=frequencies)))[0] == code


@pytest.mark.parametrize("block", [
    {"start_s": 0.0, "end_s": 0.0, "steps": 10},
    {"start_s": 1.0, "end_s": 0.5, "steps": 10},
    {"start_s": -0.1, "end_s": 1.0, "steps": 10},
])
def test_invalid_time_grid_rejected(block):
    assert code_of(frame(transient(time=block)))[0] == "invalid_time_grid"


def test_time_budgets_rejected():
    assert code_of(frame(transient(time={"start_s": 0, "end_s": 1, "steps": 5001})))[0] == \
        "time_budget_exceeded"
    many = [0.001 * index for index in range(5002)]
    assert code_of(frame(transient(time={"times_s": many})))[0] == "time_budget_exceeded"


def test_non_monotonic_and_non_uniform_times_rejected():
    assert code_of(frame(transient(time={"times_s": [0.0, 0.002, 0.001]})))[0] == \
        "non_monotonic_time"
    assert code_of(frame(transient(time={"times_s": [0.0, 0.001, 0.0021]})))[0] == \
        "non_uniform_time"


def test_mixed_or_incomplete_time_forms_rejected():
    both = {"start_s": 0, "end_s": 1, "steps": 10, "times_s": [0.0, 1.0]}
    assert code_of(frame(transient(time=both)))[0] == "invalid_time_grid"
    assert code_of(frame(transient(time={"start_s": 0.0, "end_s": 1.0})))[0] == "missing_field"


def test_dof_time_budget_rejected_for_large_models():
    nodes = {f"n{index}": [0.02 * index, 0.0] for index in range(101)}
    members = {f"m{index}": {"kind": "frame2d", "i": f"n{index}", "j": f"n{index + 1}",
                             "elements": 1, "young_modulus": E, "area": A,
                             "second_moment": I, "density": RHO}
               for index in range(100)}
    doc = frame(transient(time={"start_s": 0, "end_s": 1, "steps": 5000}),
                nodes=nodes, members=members, supports={"n0": {"ux": 0, "uy": 0, "rz": 0}})
    code, path = code_of(doc)
    assert (code, path) == ("time_budget_exceeded", "/analysis/time")


def test_load_errors_are_reported():
    assert code_of(frame(harmonic(), {"nodal": {"ghost": {"fy": 1.0}}}))[0] == "unknown_node"
    assert code_of(frame(harmonic(), {"nodal": {"tip": {"fy": {"phase_deg": 0.0}}}}))[0] == \
        "unsupported_load"
    assert code_of(truss(harmonic(), {"nodal": {"b": {"mz": 1.0}}}))[0] == "unsupported_load"
    assert code_of(frame(harmonic(), {"nodal": {"tip": {"fy": float("inf")}}}))[0] == \
        "invalid_load"


def test_transient_preset_errors_are_reported():
    assert code_of(frame(transient(), {"nodal": {"tip": {"fy": {
        "kind": "table", "amplitude": 1.0}}}}))[0] == "unsupported_load"
    assert code_of(frame(transient(), {"nodal": {"tip": {"fy": {
        "kind": "constant"}}}}))[0] == "unsupported_load"
    assert code_of(frame(transient(), {"nodal": {"tip": {"fy": {
        "kind": "ramp", "amplitude": 1.0, "t0_s": 0.0, "t1_s": 0.06}}}}))[0] == "invalid_load"
    assert code_of(frame(transient(), {"nodal": {"tip": {"fy": {
        "kind": "ramp", "amplitude": 1.0, "t0_s": 0.01, "t1_s": 0.01}}}}))[0] == "invalid_load"
    assert code_of(frame(transient(), {"nodal": {"tip": {"fy": {
        "kind": "harmonic", "amplitude": 1.0, "frequency_hz": 0.0}}}}))[0] == "invalid_load"


def test_initial_state_errors_are_reported():
    fixed = transient(initial_state={"displacement": {"base": {"uy": 0.001}}})
    assert code_of(frame(fixed))[0] == "unsupported_initial_condition"
    assert code_of(truss(transient(initial_state={"velocity": {"b": {"rz": 0.1}}})))[0] == \
        "unsupported_initial_condition"
    assert code_of(frame(transient(initial_state={
        "velocity": {"tip": {"uy": float("nan")}}})))[0] == "unsupported_initial_condition"
    assert code_of(frame(transient(initial_state={
        "velocity": {"ghost": {"uy": 0.1}}})))[0] == "unknown_node"


def test_unsupported_analysis_and_schema_rejected():
    assert code_of(frame({"type": "response_spectrum", "frequencies_hz": [1.0]}))[0] == \
        "unsupported_analysis"
    doc = frame(harmonic())
    doc["schema_version"] = "1.6"
    assert code_of(doc)[0] == "unsupported_schema"


def test_modal_declarations_still_validate():
    doc = frame({"type": "modal", "modes": 4})
    doc["schema_version"] = "1.4"
    spec = validate_dynamics_model(doc)
    assert (spec.analysis_type, spec.modes) == ("modal", 4)


def test_mass_assembly_accepts_both_schemas():
    modal = frame({"type": "modal", "modes": 6})
    modal["schema_version"] = "1.4"
    dynamic = frame(harmonic())
    assert np.array_equal(assemble_mass(modal).matrix, assemble_mass(dynamic).matrix)
    assert assemble_mass(dynamic).dof_labels == assemble_mass(modal).dof_labels


def test_input_hash_is_stable_and_sensitive():
    doc = frame(harmonic(), {"nodal": {"tip": {"fy": {"amplitude": 1000.0, "phase_deg": 0.0}}}})
    base = canonical_dynamics_input_hash(doc)
    assert base == canonical_dynamics_input_hash(copy.deepcopy(doc))
    assert base != canonical_dynamics_input_hash(frame(
        harmonic(), {"nodal": {"tip": {"fy": {"amplitude": 1000.0, "phase_deg": 90.0}}}}))
    assert base != canonical_dynamics_input_hash(frame(harmonic(frequencies_hz=[1.0, 2.0])))
    assert base != canonical_dynamics_input_hash(frame(transient()))


def test_normalized_declaration_round_trips():
    doc = frame(transient(initial_state={"displacement": {"tip": {"uy": 0.001}}}),
                {"nodal": {"tip": {"fy": {"kind": "ramp", "amplitude": 1000.0,
                                          "t0_s": 0.0, "t1_s": 0.01}}}})
    spec = validate_dynamics_model(doc)
    again = validate_dynamics_model(spec.to_dict())
    assert again.to_dict() == spec.to_dict()
    assert canonical_dynamics_input_hash(again) == canonical_dynamics_input_hash(spec)
    assert again.dynamics.time.times_s == spec.dynamics.time.times_s

# --- harmonic response (D5/D7/D8) -------------------------------------------------

def _linalg():
    return pytest.importorskip("scipy.linalg")


def _sdof():
    stiffness = E * A / L
    mass = RHO * A * L / 3
    return stiffness, mass, math.sqrt(stiffness / mass)


def _cantilever_roots(count=40):
    from scipy.optimize import brentq

    roots = []
    for order in range(1, count + 1):
        guess = (2 * order - 1) * math.pi / 2
        roots.append(brentq(lambda x: math.cos(x) * math.cosh(x) + 1,
                            guess - 1.2, guess + 1.2))
    return roots


def _tip_mode(value):
    """Stable evaluation of phi_n(L) for the cantilever shape functions."""
    small = math.exp(-2 * value)
    numerator = (1 + small) * math.sin(value) - (1 - small) * math.cos(value)
    denominator = (1 - small) / 2 + math.sin(value) * math.exp(-value)
    return numerator / denominator


def _cantilever_series(frequencies, alpha, beta, count=40):
    roots = _cantilever_roots(count)
    omega = np.array([root**2 * math.sqrt(E * I / (RHO * A * L**4)) for root in roots])
    tip = np.array([_tip_mode(root) for root in roots])
    generalized_mass = RHO * A * L
    response = []
    for frequency in frequencies:
        value = 2 * math.pi * frequency
        zeta = alpha / (2 * omega) + beta * omega / 2
        denominator = generalized_mass * (omega**2 - value**2
                                          + 2j * zeta * omega * value)
        response.append(np.sum(tip**2 / denominator))
    return np.array(response)


def test_harmonic_sdof_matches_analytic_amplitude_and_phase():
    _linalg()
    stiffness, mass, omega_n = _sdof()
    zeta, force = 0.02, 1.0e4
    alpha = 2 * zeta * omega_n
    ratios = (0.0, 0.2, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0)
    frequencies = [ratio * omega_n / (2 * math.pi) for ratio in ratios]
    result = harmonic_response(truss(
        harmonic(frequencies_hz=frequencies,
                 damping={"kind": "rayleigh", "alpha": alpha, "beta": 0.0}),
        {"nodal": {"b": {"fx": force}}}))
    row = result.dof_labels.index(("input", "b", "ux"))
    omega = np.array([ratio * omega_n for ratio in ratios])
    exact = force / (stiffness - mass * omega**2 + 1j * (alpha * mass) * omega)
    assert np.allclose(result.displacements[row], exact, rtol=1e-12, atol=1e-18)
    assert np.allclose(result.magnitudes[row], np.abs(exact), rtol=1e-12, atol=0.0)
    assert np.allclose(result.phases_deg[row], np.degrees(np.angle(exact)),
                       rtol=0.0, atol=1e-10)
    assert np.all(result.backward_errors <= 1e-12)
    assert np.allclose(result.omega_rad_s, 2 * math.pi * np.array(frequencies), rtol=1e-15)


def test_harmonic_zero_frequency_is_the_static_limit():
    _linalg()
    result = harmonic_response(frame(harmonic(frequencies_hz=[0.0]),
                                     {"nodal": {"tip": {"fy": 1.0}}}))
    tip = result.dof_labels.index(("input", "tip", "uy"))
    exact = L**3 / (3 * E * I)
    assert abs(result.magnitudes[tip, 0] - exact) / exact <= 1e-9


def test_harmonic_complex_load_phase_is_preserved():
    _linalg()
    stiffness, _, _ = _sdof()
    result = harmonic_response(truss(
        harmonic(frequencies_hz=[0.0], damping={"kind": "rayleigh"}),
        {"nodal": {"b": {"fx": {"amplitude": 2.0, "phase_deg": 90.0}}}}))
    row = result.dof_labels.index(("input", "b", "ux"))
    assert result.magnitudes[row, 0] == pytest.approx(2.0 / stiffness)
    assert result.phases_deg[row, 0] == pytest.approx(90.0)


def test_harmonic_cantilever_matches_analytic_series():
    _linalg()
    alpha, beta = 1.720507, 4.743437e-5
    first = 1.875104068711961**2 * math.sqrt(E * I / (RHO * A * L**4)) / (2 * math.pi)
    ratios = (0.2, 0.5, 0.9, 0.99, 1.01, 1.5, 2.0)
    frequencies = [ratio * first for ratio in ratios]
    result = harmonic_response(frame(
        harmonic(frequencies_hz=frequencies,
                 damping={"kind": "rayleigh", "alpha": alpha, "beta": beta}),
        {"nodal": {"tip": {"fy": 1.0}}}))
    tip = result.dof_labels.index(("input", "tip", "uy"))
    reference = _cantilever_series(frequencies, alpha, beta)
    error = np.abs(result.displacements[tip] - reference) / np.abs(reference)
    assert np.all(error <= 2e-5), error


def test_harmonic_singular_frequency_is_rejected():
    _linalg()
    _, _, omega_n = _sdof()
    with pytest.raises(HarmonicSolveError) as info:
        harmonic_response(truss(
            harmonic(frequencies_hz=[omega_n / (2 * math.pi)],
                     damping={"kind": "rayleigh"}),
            {"nodal": {"b": {"fx": 1.0}}}))
    assert info.value.code == "harmonic_singular"
    assert info.value.index == 0


def test_harmonic_ill_conditioned_frequency_is_reported():
    _linalg()
    _, _, omega_n = _sdof()
    result = harmonic_response(truss(
        harmonic(frequencies_hz=[omega_n / (2 * math.pi)],
                 damping={"kind": "rayleigh", "alpha": 2e-12 * omega_n, "beta": 0.0}),
        {"nodal": {"b": {"fx": 1.0}}}))
    assert result.diagnostics == ("harmonic_ill_conditioned:0",)
    assert result.conditions[0] > 1e11


def test_harmonic_load_on_fixed_dof_is_rejected():
    _linalg()
    with pytest.raises(ModelValidationError) as info:
        harmonic_response(frame(harmonic(frequencies_hz=[1.0]),
                                {"nodal": {"base": {"fy": 1.0}}}))
    assert (info.value.code, info.value.path) == ("unsupported_load", "/loads/nodal/base/fy")


def test_harmonic_requires_a_harmonic_declaration():
    _linalg()
    doc = frame({"type": "modal", "modes": 4})
    doc["schema_version"] = "1.4"
    with pytest.raises(ModelValidationError) as info:
        harmonic_response(doc)
    assert info.value.code == "unsupported_analysis"


def test_harmonic_requires_free_dofs():
    _linalg()
    built = frame(harmonic(frequencies_hz=[1.0]), elements=1,
                  supports={"base": {"ux": 0, "uy": 0, "rz": 0},
                            "tip": {"ux": 0, "uy": 0, "rz": 0}})
    with pytest.raises(ModelValidationError) as info:
        harmonic_response(built)
    assert info.value.code == "no_free_dofs"


def test_harmonic_result_contract():
    _linalg()
    result = harmonic_response(frame(harmonic(frequencies_hz=[0.0, 20.0])))
    assert isinstance(result, HarmonicResult)
    assert result.displacements.shape == (51, 2)
    assert result.magnitudes.shape == result.displacements.shape
    assert result.phases_deg.shape == result.displacements.shape
    assert len(result.dof_labels) == 51
    fixed = [index for index, label in enumerate(result.dof_labels) if label[1] == "base"]
    assert np.all(result.displacements[fixed] == 0)
    assert np.allclose(result.omega_rad_s, 2 * math.pi * np.array([0.0, 20.0]),
                       rtol=1e-15)
    assert result.input_hash == canonical_dynamics_input_hash(result.model)


def test_harmonic_artifact_round_trip(tmp_path):
    _linalg()
    result = harmonic_response(frame(harmonic(),
                                     {"nodal": {"tip": {"fy": {"amplitude": 1000.0,
                                                               "phase_deg": 30.0}}}}))
    directory = save_harmonic_result(result, tmp_path, "harmonic")
    restored = load_harmonic_result(directory)
    assert np.array_equal(restored.displacements, result.displacements)
    assert np.array_equal(restored.frequencies_hz, result.frequencies_hz)
    assert np.array_equal(restored.conditions, result.conditions)
    assert restored.diagnostics == result.diagnostics
    assert restored.input_hash == result.input_hash
    assert restored.dof_labels == result.dof_labels


def test_harmonic_artifact_rejects_tampering(tmp_path):
    _linalg()
    result = harmonic_response(frame(harmonic(), {"nodal": {"tip": {"fy": 1.0}}}))
    directory = save_harmonic_result(result, tmp_path, "harmonic")

    arrays = directory / "arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"x")
    with pytest.raises(HarmonicArtifactError):
        load_harmonic_result(directory)

    directory = save_harmonic_result(result, tmp_path, "second")
    manifest = directory / "manifest.json"
    manifest.write_text(manifest.read_text().replace(
        result.input_hash, "0" * 64), encoding="utf-8")
    with pytest.raises(HarmonicArtifactError):
        load_harmonic_result(directory)

    directory = save_harmonic_result(result, tmp_path, "third")
    (directory / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(HarmonicArtifactError):
        load_harmonic_result(directory)


def test_run_harmonic_application_entry():
    _linalg()
    from anf.application import run_harmonic

    result = run_harmonic(frame(harmonic()), {"": None} if False else None)
    assert isinstance(result, HarmonicResult)
    with pytest.raises(ModelValidationError):
        run_harmonic(frame(harmonic()), {"unknown": True})


# --- transient response (D6/D7/D8) ------------------------------------------------

def _step_analytic(times, static, zeta, omega_n):
    damped = omega_n * math.sqrt(1 - zeta**2)
    return static * (1 - np.exp(-zeta * omega_n * times)
                     * (np.cos(damped * times)
                        + zeta / math.sqrt(1 - zeta**2) * np.sin(damped * times)))


def _transient_declaration(start, end, steps, *, damping, initial=None, loads=None):
    block = {"type": "transient", "time": {"start_s": start, "end_s": end, "steps": steps},
             "damping": damping}
    if initial is not None:
        block["initial_state"] = initial
    return truss(block, loads or {})


def test_transient_sdof_step_converges_at_second_order():
    _linalg()
    stiffness, _, omega_n = _sdof()
    zeta = 0.02
    alpha = 2 * zeta * omega_n
    period = 2 * math.pi / omega_n
    load = {"nodal": {"b": {"fx": {"kind": "constant", "amplitude": 1.0e4}}}}
    errors = {}
    for per_period in (20, 40, 80):
        result = transient_response(_transient_declaration(
            0.0, 2 * period, 2 * per_period,
            damping={"kind": "rayleigh", "alpha": alpha}, loads=load))
        row = result.dof_labels.index(("input", "b", "ux"))
        exact = _step_analytic(result.times_s, 1.0e4 / stiffness, zeta, omega_n)
        errors[per_period] = (np.linalg.norm(result.displacement[:, row] - exact)
                              / np.linalg.norm(exact))
        assert result.worst_backward_error <= 1e-10
    assert errors[40] <= 1.2e-2
    assert errors[80] <= 3e-3
    for coarse, fine in ((20, 40), (40, 80)):
        assert 3.5 <= errors[coarse] / errors[fine] <= 4.5


def test_transient_undamped_energy_is_conserved():
    _linalg()
    _, _, omega_n = _sdof()
    steps = 5000
    result = transient_response(_transient_declaration(
        0.0, steps * 2 * math.pi / omega_n / 20, steps,
        damping={"kind": "rayleigh"},
        initial={"velocity": {"b": {"ux": 1.0}}}))
    drift = np.max(np.abs(result.energy - result.energy[0])) / result.energy[0]
    assert drift <= 1e-10
    assert result.diagnostics == ("rayleigh_zero",)


def test_transient_damped_energy_never_increases():
    _linalg()
    _, _, omega_n = _sdof()
    zeta = 0.02
    steps = 2000
    result = transient_response(_transient_declaration(
        0.0, steps * 2 * math.pi / omega_n / 20, steps,
        damping={"kind": "rayleigh", "alpha": 2 * zeta * omega_n},
        initial={"displacement": {"b": {"ux": 0.001}}}))
    assert np.diff(result.energy).max() <= 1e-12 * result.energy[0]
    assert result.energy[-1] < result.energy[0]


def test_transient_nonzero_initial_state_matches_analytic():
    _linalg()
    _, _, omega_n = _sdof()
    zeta = 0.02
    period = 2 * math.pi / omega_n
    result = transient_response(_transient_declaration(
        0.0, 4 * period, 4 * 80,
        damping={"kind": "rayleigh", "alpha": 2 * zeta * omega_n},
        initial={"displacement": {"b": {"ux": 1e-3}}, "velocity": {"b": {"ux": 0.1}}}))
    row = result.dof_labels.index(("input", "b", "ux"))
    damped = omega_n * math.sqrt(1 - zeta**2)
    exact = np.exp(-zeta * omega_n * result.times_s) * (
        1e-3 * np.cos(damped * result.times_s)
        + (0.1 + zeta * omega_n * 1e-3) / damped * np.sin(damped * result.times_s))
    error = np.linalg.norm(result.displacement[:, row] - exact) / np.linalg.norm(exact)
    assert error <= 2e-2
    assert result.worst_backward_error <= 1e-10


def test_transient_summary_matches_the_stored_history():
    _linalg()
    from anf.dynamics.system import assemble_stiffness

    result = transient_response(frame(
        transient(time={"start_s": 0.0, "end_s": 0.004, "steps": 100}),
        {"nodal": {"tip": {"fy": {"kind": "constant", "amplitude": 1.0}}}}))
    absolute = np.abs(result.displacement)
    assert np.array_equal(result.peak_displacement, absolute.max(axis=0))
    assert np.array_equal(result.peak_time_s, result.times_s[absolute.argmax(axis=0)])
    stiffness = assemble_stiffness(result.model)
    mass = assemble_mass(result.model).matrix
    for index in range(len(result.times_s)):
        expected = (0.5 * float(result.velocity[index] @ mass @ result.velocity[index])
                    + 0.5 * float(result.displacement[index] @ stiffness
                                  @ result.displacement[index]))
        assert result.energy[index] == expected

    def peak(values):
        flat = int(np.abs(values).argmax())
        row, column = np.unravel_index(flat, values.shape)
        return (float(abs(values[row, column])), float(result.times_s[row]), int(column))

    assert result.global_peaks["displacement"] == peak(result.displacement)
    assert result.global_peaks["velocity"] == peak(result.velocity)
    assert result.global_peaks["acceleration"] == peak(result.acceleration)


def test_transient_result_contract():
    _linalg()
    result = transient_response(frame(
        transient(time={"start_s": 0.0, "end_s": 0.004, "steps": 4}),
        {"nodal": {"tip": {"fy": {"kind": "constant", "amplitude": 1.0}}}}))
    assert isinstance(result, TransientResult)
    assert result.displacement.shape == (5, 51)
    assert result.velocity.shape == (5, 51)
    assert result.acceleration.shape == (5, 51)
    assert result.energy.shape == (5,)
    assert result.peak_displacement.shape == (51,)
    assert result.peak_time_s.shape == (51,)
    fixed = [index for index, label in enumerate(result.dof_labels) if label[1] == "base"]
    assert np.all(result.displacement[:, fixed] == 0)
    assert result.dt_s == pytest.approx(0.001)
    assert result.start_s == 0.0
    assert result.diagnostics == ("rayleigh_zero",)


def test_transient_ramp_is_zero_before_it_starts():
    _linalg()
    result = transient_response(_transient_declaration(
        0.0, 0.004, 40, damping={"kind": "rayleigh"},
        loads={"nodal": {"b": {"fx": {"kind": "ramp", "amplitude": 1e4,
                                      "t0_s": 0.001, "t1_s": 0.002}}}}))
    row = result.dof_labels.index(("input", "b", "ux"))
    assert np.all(result.displacement[:11, row] == 0.0)
    assert np.any(result.displacement[11:, row] != 0.0)


def test_transient_harmonic_preset_reaches_the_harmonic_solution():
    _linalg()
    _, _, omega_n = _sdof()
    zeta = 0.1
    alpha = 2 * zeta * omega_n
    frequency = 0.5 * omega_n / (2 * math.pi)
    periods, per_period = 25, 160
    period = 1 / frequency
    result = transient_response(_transient_declaration(
        0.0, periods * period, periods * per_period,
        damping={"kind": "rayleigh", "alpha": alpha},
        loads={"nodal": {"b": {"fx": {"kind": "harmonic", "amplitude": 1.0,
                                      "frequency_hz": frequency, "phase_deg": 0.0}}}}))
    steady = harmonic_response(truss(
        harmonic(frequencies_hz=[frequency],
                 damping={"kind": "rayleigh", "alpha": alpha}),
        {"nodal": {"b": {"fx": 1.0}}}))
    row = result.dof_labels.index(("input", "b", "ux"))
    expected = abs(steady.displacements[row, 0])
    window = result.times_s >= result.times_s[-1] - period
    peak = np.abs(result.displacement[window, row]).max()
    assert abs(peak - expected) / expected <= 1e-3


def test_transient_requires_transient_declaration():
    _linalg()
    with pytest.raises(ModelValidationError) as info:
        transient_response(frame(harmonic()))
    assert info.value.code == "unsupported_analysis"


def test_transient_fixed_dof_load_is_rejected():
    _linalg()
    with pytest.raises(ModelValidationError) as info:
        transient_response(frame(transient(), {"nodal": {"base": {"fy": {
            "kind": "constant", "amplitude": 1.0}}}}))
    assert (info.value.code, info.value.path) == ("unsupported_load", "/loads/nodal/base/fy")


def test_transient_requires_free_dofs():
    _linalg()
    built = frame(transient(), elements=1,
                  supports={"base": {"ux": 0, "uy": 0, "rz": 0},
                            "tip": {"ux": 0, "uy": 0, "rz": 0}})
    with pytest.raises(ModelValidationError) as info:
        transient_response(built)
    assert info.value.code == "no_free_dofs"


def test_transient_singular_factorization_is_reported(monkeypatch):
    _linalg()
    import scipy.linalg

    def explode(*args, **kwargs):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(scipy.linalg, "cho_factor", explode)
    with pytest.raises(NewmarkSolveError) as info:
        transient_response(frame(transient(), {"nodal": {"tip": {"fy": {
            "kind": "constant", "amplitude": 1.0}}}}))
    assert info.value.code == "newmark_singular"


def test_transient_artifact_round_trip(tmp_path):
    _linalg()
    result = transient_response(frame(
        transient(time={"start_s": 0.0, "end_s": 0.004, "steps": 20}),
        {"nodal": {"tip": {"fy": {"kind": "ramp", "amplitude": 1.0,
                                  "t0_s": 0.0, "t1_s": 0.002}}}}))
    directory = save_transient_result(result, tmp_path, "transient")
    restored = load_transient_result(directory)
    assert np.array_equal(restored.displacement, result.displacement)
    assert np.array_equal(restored.velocity, result.velocity)
    assert np.array_equal(restored.acceleration, result.acceleration)
    assert np.array_equal(restored.energy, result.energy)
    assert np.array_equal(restored.peak_time_s, result.peak_time_s)
    assert restored.global_peaks == result.global_peaks
    assert restored.worst_backward_error == result.worst_backward_error
    assert restored.diagnostics == result.diagnostics
    assert restored.input_hash == result.input_hash
    assert restored.dof_labels == result.dof_labels


def test_transient_artifact_rejects_tampering(tmp_path):
    _linalg()
    result = transient_response(frame(
        transient(time={"start_s": 0.0, "end_s": 0.002, "steps": 10}),
        {"nodal": {"tip": {"fy": {"kind": "constant", "amplitude": 1.0}}}}))
    directory = save_transient_result(result, tmp_path, "transient")
    arrays = directory / "arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"x")
    with pytest.raises(TransientArtifactError):
        load_transient_result(directory)

    directory = save_transient_result(result, tmp_path, "second")
    manifest = directory / "manifest.json"
    manifest.write_text(manifest.read_text().replace(
        result.input_hash, "0" * 64), encoding="utf-8")
    with pytest.raises(TransientArtifactError):
        load_transient_result(directory)

    directory = save_transient_result(result, tmp_path, "third")
    (directory / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(TransientArtifactError):
        load_transient_result(directory)


def test_run_transient_application_entry():
    _linalg()
    from anf.application import run_transient

    result = run_transient(frame(
        transient(time={"start_s": 0.0, "end_s": 0.002, "steps": 10}),
        {"nodal": {"tip": {"fy": {"kind": "constant", "amplitude": 1.0}}}}))
    assert isinstance(result, TransientResult)
    with pytest.raises(ModelValidationError):
        run_transient(frame(transient()), {"unknown": True})
