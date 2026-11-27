"""Export validated static result artifacts as offline, interactive reports."""

from dataclasses import asdict, dataclass
from importlib.resources import files
import json
from pathlib import Path

import numpy as np

from ..artifacts import load_result


class ViewerOutputError(ValueError):
    """The viewer report could not be written."""


@dataclass(frozen=True)
class FieldDescriptor:
    """The identity, units and association of a displayed scalar field."""

    name: str
    unit: str
    position: str
    coordinates: str


def open_result_artifact(directory):
    """Read a static 1.0 or 1.1 artifact using the standard result reader."""
    return load_result(directory)


def _structure_geometry(model):
    mesh = model.mesh
    order = mesh.node_order
    labels = [key[1] if key[0] == "input" else f"{key[1]} / subdivision {key[2]}"
              for key in order]
    indices = {key: index for index, key in enumerate(order)}
    named = {key[1]: indices[key] for key in order if key[0] == "input"}
    supports = [(named[name], component, "rad" if component == "rz" else "m", value)
                for name, row in model.model["supports"].items() for component, value in row.items()]
    loads = [(named[name], labels[named[name]], component, "N·m" if component == "mz" else "N", value)
             for name, row in model.loads["nodal"].items() for component, value in row.items()]
    first_segments = {}
    last_segments = {}
    for segment in mesh.segments:
        first_segments.setdefault(segment.member, segment.nodes)
        last_segments[segment.member] = segment.nodes
    ends = [(name, first_segments[name], last_segments[name][::-1])
            for name in mesh.member_order]
    for name, first, last in ends:
        if name in model.loads["uniform_transverse"]:
            loads.append(([first[0], last[0]], name, "uniform transverse (local +y)", "N/m",
                          model.loads["uniform_transverse"][name]))
    return {
        "coordinates": "Global X–Y",
        "points": [(*point, 0) for point in mesh.coordinates],
        "labels": labels,
        "cells": [segment.nodes for segment in mesh.segments],
        "supports": supports,
        "loads": loads,
        "frame_nodes": [indices[key] for key in mesh.frame_node_order],
        "member_ends": ends,
    }


def _beam_geometry(model):
    count = model.model.elements + 1
    return {
        "coordinates": "Beam x, transverse w",
        "points": [(x, 0, 0) for x in np.linspace(0.0, model.model.length, count)],
        "labels": [f"x station {i}" for i in range(count)],
        "cells": [(i, i + 1) for i in range(count - 1)],
        "supports": [(0, "w", "m", 0), (0, "slope", "rad", 0)],
        "loads": [(count - 1, "Tip", "force (+w)", "N", model.loads["tip_force"]),
                  ([0, count - 1], "Span", "uniform load (+w)", "N/m", model.loads["uniform_load"])],
    }


def _plate_geometry(model):
    plate = model.model
    nx, ny = plate.nx, plate.ny
    stride = nx + 1
    points = [(x, y, 0) for y in np.linspace(0.0, plate.ly, ny + 1)
              for x in np.linspace(0.0, plate.lx, stride)]
    cells = []
    supports = []
    for j in range(ny + 1):
        for i in range(stride):
            index = j * stride + i
            if i < nx and j < ny:
                cells.append((index, index + 1, index + stride + 1, index + stride))
            on_x, on_y = i in (0, nx), j in (0, ny)
            if on_x or on_y:
                supports.append((index, "w", "m", 0))
                if on_y or plate.support == "clamped":
                    supports.append((index, "beta_x", "rad", 0))
                if on_x or plate.support == "clamped":
                    supports.append((index, "beta_y", "rad", 0))
    return {
        "coordinates": "Plate x–y, normal w",
        "points": points,
        "labels": [f"Grid [{j}, {i}]" for j in range(ny + 1) for i in range(stride)],
        "cells": cells,
        "supports": supports,
        "loads": [(None, "Plate surface", "pressure (+w)", "Pa", model.loads["pressure"])],
    }


def _fields(result, geometry):
    fields = []
    for name, metadata in result.array_metadata.items():
        if name in ("x", "y"):
            continue
        position = metadata["position"]
        coordinates = geometry["coordinates"]
        if position == "member_end":
            coordinates = "Local member axes (i → j)"
        descriptor = FieldDescriptor(name, metadata["unit"], position, coordinates)
        fields.append({
            **asdict(descriptor),
            "values": result.arrays[name].ravel().tolist(),
        })
    return fields


def _scene(result):
    model = result.model
    problem = model.problem_type
    if model.mesh is not None:
        geometry = _structure_geometry(model)
    elif problem == "cantilever_bending":
        geometry = _beam_geometry(model)
    else:
        geometry = _plate_geometry(model)
    arrays = result.arrays
    displacement = []
    if arrays:
        zero = np.zeros(len(geometry["points"]))
        if model.mesh is not None:
            components = (arrays["node_ux"], arrays["node_uy"], zero)
        elif problem == "cantilever_bending":
            components = (zero, arrays["deflection"], zero)
        else:
            components = (zero, zero, arrays["deflection"].ravel())
        displacement = list(zip(*(component.tolist() for component in components), strict=True))
    return {
        **geometry,
        "problem": problem,
        "schema_version": result.schema_version,
        "input_hash": result.input_hash,
        "solve_status": result.solve_status,
        "verification_status": result.verification_status,
        "diagnostics": list(result.diagnostics),
        "strain_energy": result.strain_energy,
        "relative_residual": result.relative_residual,
        "fields": _fields(result, geometry),
        "displacement": displacement,
    }


def export_result_view(result, output):
    """Write a self-contained HTML report at a new path and return that path."""
    output = Path(output)
    if output.suffix.lower() != ".html":
        raise ViewerOutputError("the viewer output must have an .html extension")
    resources = files(__package__)
    template = resources.joinpath("report.html").read_text(encoding="utf-8")
    script = resources.joinpath("report.js").read_text(encoding="utf-8")
    head, body = template.split("<!-- REPORT -->")
    payload = json.dumps(_scene(result), allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = (head + '<script type="application/json" id="result-data">' + payload + "</script>\n"
            + "<script>\n" + script + "\n</script>" + body)
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(page)
    except OSError as error:
        raise ViewerOutputError(str(error)) from error
    return output
