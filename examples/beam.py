"""Cantilever with a tip force: compare with the integrated beam equation."""
from anf import Beam, solve_cantilever


def main():
    model = Beam(length=2, young_modulus=210e9, second_moment=1e-6)
    force = 1000.0
    result = solve_cantilever(model, tip_force=force)
    reference = force * model.length**3 / (3 * model.young_modulus * model.second_moment)
    error = abs(result.deflection[-1] / reference - 1)
    print("deflection_m,reference_m,relative_error,relative_residual")
    print(f"{result.deflection[-1]:.10e},{reference:.10e},{error:.6e},{result.relative_residual:.6e}")
    if error > 1e-8:
        raise RuntimeError("beam result differs from the analytical reference")


if __name__ == "__main__":
    main()
