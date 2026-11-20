"""Expose the known thin-plate locking limitation; this is a negative example."""
from plate_convergence import run_case


def main():
    print("thickness_m,mesh_per_side,deflection_m,reference_m,relative_error")
    error = run_case(8, 0.005)
    print("Known shear locking: this result is unsuitable for thin-plate accuracy claims.")
    if error <= 0.8:
        raise RuntimeError("the documented locking baseline has changed; review the result")


if __name__ == "__main__":
    main()
