"""Square plate refinement against continuous sine-mode compliance."""
import argparse
import math
from anf import Plate, solve_plate


def reference_center(model, pressure, modes=31):
    """Independent continuous energy solution; see docs/methods.md."""
    t, nu, E = model.thickness, model.poisson_ratio, model.young_modulus
    bending = E * t**3 / (12 * (1 - nu**2))
    shear = model.shear_factor * E * t / (2 * (1 + nu))
    terms = []
    for m in range(1, modes + 1, 2):
        for n in range(1, modes + 1, 2):
            wave = (m * math.pi / model.lx)**2 + (n * math.pi / model.ly)**2
            sign = (-1)**((m - 1)//2 + (n - 1)//2)
            terms.append(sign * 16 * pressure / (math.pi**2 * m * n)
                         * (1 / (bending * wave**2) + 1 / (shear * wave)))
    return math.fsum(terms)


def run_case(n, thickness, *, formulation="q4_full", modes=31):
    model = Plate(1, 1, thickness, 70e9, 0.3, nx=n, ny=n,
                  support="hard_simply_supported", formulation=formulation)
    result = solve_plate(model, pressure=1000)
    computed = result.deflection[n//2, n//2]
    reference = reference_center(model, 1000, modes=modes)
    error = abs(computed / reference - 1)
    print(f"{thickness:g},{n},{computed:.8e},{reference:.8e},{error:.6%}")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", choices=("q4_full", "mitc4_rectangle"),
                        default="q4_full", help="compare thick Q4 or thin MITC4 plates")
    formulation = parser.parse_args().formulation
    if formulation == "mitc4_rectangle":
        thickness, modes, tolerance = 0.005, 255, 0.03
    else:
        thickness, modes, tolerance = 0.2, 31, 0.08
    print("thickness_m,mesh_per_side,deflection_m,reference_m,relative_error")
    errors = [run_case(n, thickness, formulation=formulation, modes=modes) for n in (4, 8, 16)]
    if not errors[2] < errors[1] < errors[0] or errors[2] >= tolerance:
        raise RuntimeError(f"{formulation} refinement failed to approach the reference")


if __name__ == "__main__":
    main()
