"""Write replayable convergence reports and their referenced result bundles."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .artifacts import ArtifactError, _safe_name, save_result
from .studies import StudyResult


def _markdown(report):
    study = report["study"]
    lines = [
        "# Mesh convergence",
        "",
        f"Base fingerprint `{study['base_model_hash']}`",
        "",
        f"Observable `{study['observable']}` in m. Study status `{report['status']}`.",
        f"Reference `{study['reference'] or 'none'}`. Mesh failure policy `{study['failure_policy']}`.",
        "",
        "| Mesh | Solve | Verification | Deflection (m) | Adjacent change (m) | Reference error |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]

    def show(number):
        return "—" if number is None else format(number, ".7g")

    for step in report["steps"]:
        mesh = " × ".join(str(count) for count in step["mesh"].values())
        check = step["reference_check"]
        comparison = f"{show(check['error'])} ({check['metric']})" if check and check["error"] is not None else "—"
        artifact = step["artifact"]
        label = f"[{mesh}]({artifact['path']}/manifest.json)" if artifact else mesh
        lines.append(f"| {label} | {step['solve_status']} | {step['verification_status']} "
                     f"| {show(step['observed_m'])} | {show(step['successive_mesh_change_m'])} | {comparison} |")
    estimate = report["estimate"]
    lines.extend(("", "![Deflection by mesh step](convergence.svg)", "", "## Discretization estimate", ""))
    if estimate["status"] == "estimated":
        lines.append(f"Observed order {show(estimate['observed_order'])}. "
                     f"Finest-mesh estimate {show(estimate['finest_discretization_estimate_m'])} m.")
    else:
        lines.append(f"Unavailable (`{estimate['reason']}`).")
    diagnostics = [(index, item["message"]) for index, step in enumerate(report["steps"], 1)
                   for item in step["diagnostics"]]
    if diagnostics:
        lines.extend(("", "## Diagnostics", ""))
        lines.extend(f"- Mesh {index} — {' '.join(message.splitlines())}" for index, message in diagnostics)
    return "\n".join(lines) + "\n"


def _plot(report):
    steps = report["steps"]
    values = [(index, step["observed_m"]) for index, step in enumerate(steps)
              if step["observed_m"] is not None]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 580 280" role="img">',
        '<title>Deflection by mesh step</title>',
        '<rect width="580" height="280" fill="white"/>',
        '<path d="M 112 22 V 220 H 548" fill="none" stroke="#344054"/>',
        '<g fill="#253044" font-family="sans-serif" font-size="12">',
        '<text x="12" y="17">Deflection (m)</text>', '<text x="308" y="268">Mesh step</text>',
    ]
    spacing = 426 / max(len(steps) - 1, 1)
    for index in range(len(steps)):
        x = 116 + index * spacing
        svg.append(f'<text x="{x:.1f}" y="240" text-anchor="middle">{index + 1}</text>')
    if values:
        lo, hi = min(value for _, value in values), max(value for _, value in values)
        scale = max(abs(hi), abs(lo)) or 1.0
        span = hi / scale - lo / scale
        svg.append(f'<text x="105" y="35" text-anchor="end">{hi:.4g}</text>')
        if span:
            svg.append(f'<text x="105" y="215" text-anchor="end">{lo:.4g}</text>')
        for index, value in values:
            fraction = (value / scale - lo / scale) / span if span else 0.5
            svg.append(f'<circle cx="{116 + index * spacing:.1f}" cy="{212 - 180*fraction:.1f}" '
                       'r="4" fill="#175e65"/>')
    else:
        svg.append('<text x="180" y="130">No solved meshes</text>')
    svg.append("</g></svg>")
    return "\n".join(svg) + "\n"


def write_report(study, output_dir):
    """Write JSON, Markdown, a plot and results to a new directory. Return report.json."""
    if not isinstance(study, StudyResult):
        raise TypeError("expected a StudyResult")
    requested = Path(output_dir)
    _safe_name(requested.name)
    try:
        root = requested.parent.resolve(strict=True)
        output = root / requested.name
        if output.exists() or output.is_symlink():
            raise ArtifactError("study output already exists")
        report = study.to_dict()
        with TemporaryDirectory(prefix=".anf-study-", dir=root) as temporary:
            staging = Path(temporary)
            for index, (step, row) in enumerate(zip(study.steps, report["steps"], strict=True), 1):
                row["artifact"] = None
                if step.result is not None:
                    path = save_result(step.result, staging, f"mesh-{index:02}")
                    digest = sha256((path / "manifest.json").read_bytes()).hexdigest()
                    row["artifact"] = {"path": path.name, "manifest_sha256": digest}
            (staging / "report.md").write_text(_markdown(report), encoding="utf-8")
            (staging / "convergence.svg").write_text(_plot(report), encoding="utf-8")
            (staging / "report.json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8")
            size = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
            if size > study.spec.max_output_bytes:
                raise ArtifactError("study output exceeds its byte budget")
            output.mkdir(mode=0o700)
            for path in staging.iterdir():
                if path.name != "report.json":
                    path.replace(output / path.name)
            return (staging / "report.json").replace(output / "report.json")
    except OSError as error:
        raise ArtifactError(f"could not write study report: {error}") from error
