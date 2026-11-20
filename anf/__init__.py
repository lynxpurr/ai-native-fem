"""Small classical beam and plate research models."""
from .beam import Beam, BeamResult, solve_cantilever
from .plate import Plate, PlateResult, solve_plate

__all__ = ["Beam", "BeamResult", "solve_cantilever", "Plate", "PlateResult", "solve_plate"]
