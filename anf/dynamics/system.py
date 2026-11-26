"""Shared dense assembly, version and free-DOF helpers for planar dynamics."""

from importlib.metadata import PackageNotFoundError, version
import platform

import numpy as np

from ..assembly import DenseAssembly
from ..model import ModelValidationError
from ..structures import DISPLACEMENTS, _element


def assemble_stiffness(spec):
    """Return the dense global stiffness matrix for a planar structure."""
    mesh = spec.mesh
    assembly = DenseAssembly(mesh.dof_count)
    for segment in mesh.segments:
        member = spec.model["members"][segment.member]
        width = 3 if member["kind"] == "frame2d" else 2
        dofs = (*mesh.node_dofs[segment.nodes[0]][:width],
                *mesh.node_dofs[segment.nodes[1]][:width])
        assembly.add(dofs, _element(member, segment.geometry).stiffness())
    return assembly.finish()


def _fixed_dofs(spec, mesh):
    node_indices = {key[1]: index for index, key in enumerate(mesh.node_order)
                    if key[0] == "input"}
    fixed = []
    for name, row in spec.model["supports"].items():
        for component, value in row.items():
            if value != 0:
                raise ModelValidationError(
                    f"/model/supports/{name}/{component}", "unsupported_constraint",
                    "dynamic supports must be zero displacements")
            fixed.append(mesh.node_dofs[node_indices[name]][DISPLACEMENTS.index(component)])
    return fixed


def free_dofs(spec, mesh):
    """Return the free DOF indices and whether the model has any support."""
    fixed = _fixed_dofs(spec, mesh)
    return np.setdiff1d(np.arange(mesh.dof_count), fixed), bool(fixed)


def dof_labels(mesh):
    """Return labels of the form ``(kind, node, component)`` for every global DOF."""
    labels = []
    for key, dofs in zip(mesh.node_order, mesh.node_dofs):
        labels.extend((*key, component) for component in ("ux", "uy", "rz")[:len(dofs)])
    return tuple(labels)

def versions():
    """Return the package, interpreter and numerical library versions."""
    try:
        package = version("ai-native-fem")
    except PackageNotFoundError:
        package = "uninstalled"
    try:
        import scipy
    except ImportError:  # pragma: no cover - the sparse backend imports it first
        scipy_version = "uninstalled"
    else:
        scipy_version = scipy.__version__
    return {"anf": package, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy_version}