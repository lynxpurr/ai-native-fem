"""Thick square plate refinement against continuous sine-mode compliance."""
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


def run_case(n, thickness):
    model = Plate(1, 1, thickness, 70e9, 0.3, nx=n, ny=n,
                  support="hard_simply_supported")
    result = solve_plate(model, pressure=1000)
    computed = result.deflection[n//2, n//2]
    reference = reference_center(model, 1000)
    error = abs(computed / reference - 1)
    print(f"{thickness:g},{n},{computed:.8e},{reference:.8e},{error:.6%}")
    return error


def main():
    print("thickness_m,mesh_per_side,deflection_m,reference_m,relative_error")
    errors = [run_case(n, 0.2) for n in (4, 8, 16)]
    if not errors[2] < errors[1] < errors[0] or errors[2] >= 0.08:
        raise RuntimeError("thick-plate refinement failed to approach the reference")


if __name__ == "__main__":
    main()
