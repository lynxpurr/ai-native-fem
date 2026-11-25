"""Schema 1.5 forced-dynamics declaration checks."""

import copy

import numpy as np
import pytest

from anf.dynamics import assemble_mass
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