"""Run the cantilever grid and local sensitivity into a new or resumable directory."""

from argparse import ArgumentParser
from pathlib import Path

from anf.parameter_studies import ParameterGrid, finite_difference_sensitivity, run_parameter_study


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    output = parser.parse_args().directory
    output.mkdir(exist_ok=True)
    grid = ParameterGrid.from_dict(Path(__file__).with_suffix(".json").read_bytes())
    cache = output / "cache"
    scan = run_parameter_study(grid, output / "scan", cache_dir=cache)
    sensitivity = finite_difference_sensitivity(grid, output / "sensitivity", cache_dir=cache)
    print(f"Grid samples solved: {scan['counts']['solved']} / {len(scan['samples'])}")
    for path, row in sensitivity["sensitivities"].items():
        derivative = row["derivative"]
        value = row["reason"] if derivative is None else f"{derivative:.7g} {row['unit']}"
        print(f"{path}: {value}")


if __name__ == "__main__":
    main()
