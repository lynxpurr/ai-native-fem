"""Build or recheck release packages from the exact checked-out Git commit."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from shutil import copyfile, copytree

INVENTORY = ".github/workflows/release-files.json"
MANIFEST = "release-manifest.json"
CHECKSUMS = "SHA256SUMS"
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
EXAMPLES = (
    ("beam.py",),
    ("plate_convergence.py",),
    ("plate_convergence.py", "--formulation", "mitc4_rectangle"),
    ("plate_locking.py",),
)
CHECK_INSTALL = """
import json
import sys
from importlib.metadata import version
from pathlib import Path

import anf
from anf.verification import (
    REFERENCE_REGISTRY, ReferenceKind, ReferenceRecord, VerificationCheck,
    VerificationReport, evaluate_checks, input_parameters, verify_cantilever, verify_plate,
)

assert Path(anf.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert sorted(anf.__all__) == [
    "Beam", "BeamResult", "Plate", "PlateResult", "solve_cantilever", "solve_plate"
]
assert version("ai-native-fem") == sys.argv[1]
print(json.dumps({name: version(name) for name in ("ai-native-fem", "numpy", "pytest", "build")}))
"""


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def git(*args):
    return subprocess.check_output(["git", *args])


def snapshot(directory):
    commit, tree = git("show", "-s", "--format=%H %T", "HEAD").decode().split()
    groups = json.loads(git("show", f"{commit}:{INVENTORY}"))
    approved = [name for paths in groups.values() for name in paths]
    require(len(approved) == len(set(approved)), "Duplicate path in the repository inventory.")
    entries = {}
    for entry in git("ls-tree", "-rlz", commit).split(b"\0")[:-1]:
        metadata, name = entry.decode().split("\t", 1)
        mode, kind, blob, size = metadata.split()
        require(
            re.fullmatch(r"[\w.-]+(?:/[\w.-]+)*", name, re.ASCII)
            and mode in {"100644", "100755"} and kind == "blob",
            f"Unsupported repository entry: {name!r}",
        )
        require(int(size) <= MAX_MEMBER_BYTES, f"Repository file exceeds size limit: {name}")
        entries[name] = blob
    require(
        set(entries) == set(approved),
        f"Repository inventory differs: extra={sorted(entries.keys() - set(approved))}, "
        f"missing={sorted(set(approved) - entries.keys())}",
    )
    source_files = {}
    for name, blob in entries.items():
        content = git("cat-file", "blob", blob)
        source_files[name] = content
        target = directory / name
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(content)
    project = tomllib.loads(source_files["pyproject.toml"].decode())["project"]
    source = {
        "commit": commit, "tree": tree, "project": project["name"],
        "version": project["version"], "files": groups,
    }
    return source, source_files, project


def archive_files(path, expected):
    parents = {
        str(parent) for name in expected for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    contents = {}
    seen = set()

    def read(name, directory, regular, size, open_file):
        require(name not in seen, f"Duplicate archive entry: {name!r}")
        seen.add(name)
        if directory:
            require(name in parents, f"Unexpected archive directory: {name!r}")
            return
        require(regular and name in expected, f"Unexpected archive member: {name!r}")
        require(size <= MAX_MEMBER_BYTES, f"Archive member exceeds size limit: {name}")
        with open_file() as stream:
            contents[name] = stream.read(MAX_MEMBER_BYTES + 1)
        require(len(contents[name]) == size, f"Wrong archive member size: {name}")
        original = expected[name]
        require(original is None or contents[name] == original, f"Source bytes differ: {name}")

    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                mode = stat.S_IFMT(member.external_attr >> 16)
                directory = member.is_dir() and mode in {0, stat.S_IFDIR}
                read(
                    member.filename.removesuffix("/"), directory,
                    not member.is_dir() and mode in {0, stat.S_IFREG}, member.file_size,
                    lambda: archive.open(member),
                )
    else:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                read(
                    member.name, member.isdir(), member.isreg(), member.size,
                    lambda: archive.extractfile(member),
                )
    require(contents.keys() == expected.keys(), f"Missing archive files: {sorted(expected.keys() - contents.keys())}")
    return contents


def wheel_record(contents, record_path):
    rows = list(csv.reader(io.StringIO(contents[record_path].decode("utf-8"))))
    expected = {}
    for name, data in contents.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        expected[name] = (f"sha256={digest}", str(len(data)))
    expected[record_path] = ("", "")
    require(
        len(rows) == len(expected) and all(len(row) == 3 for row in rows)
        and {row[0]: tuple(row[1:]) for row in rows} == expected,
        "Wheel RECORD paths, hashes or sizes differ from its contents.",
    )


def package_metadata(data, project):
    metadata = BytesParser().parsebytes(data)
    fields = {
        "Name": [project["name"]],
        "Version": [project["version"]],
        "Requires-Python": [project["requires-python"]],
        "License-Expression": [project["license"]],
        "License-File": sorted(project["license-files"]),
        "Provides-Extra": sorted(project.get("optional-dependencies", {})),
    }
    for name, expected in fields.items():
        require(sorted(metadata.get_all(name, [])) == expected, f"Package metadata differs: {name}")


def sha256(path):
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, f"Asset exceeds size limit: {path.name}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_paths(directory, source):
    stem = f"{source['project'].replace('-', '_')}-{source['version']}"
    return directory / f"{stem}-py3-none-any.whl", directory / f"{stem}.tar.gz"


def verify_release(wheel, sdist, source, source_files, project):
    stem = sdist.name.removesuffix(".tar.gz")
    dist_info = f"{stem}.dist-info"
    groups = source["files"]
    wheel_expected = {name: source_files[name] for name in groups["package"]}
    wheel_expected.update({f"{dist_info}/licenses/{name}": source_files[name] for name in groups["licenses"]})
    wheel_expected.update({f"{dist_info}/{name}": None for name in ("METADATA", "WHEEL", "top_level.txt", "RECORD")})
    wheel_contents = archive_files(wheel, wheel_expected)
    wheel_record(wheel_contents, f"{dist_info}/RECORD")
    tags = BytesParser().parsebytes(wheel_contents[f"{dist_info}/WHEEL"])
    require(tags.get_all("Tag") == ["py3-none-any"] and tags["Root-Is-Purelib"] == "true", "Wheel tags differ.")

    sdist_expected = {
        f"{stem}/{name}": source_files[name]
        for group in ("package", "licenses", "source_distribution") for name in groups[group]
    }
    sdist_expected.update({f"{stem}/{name}": None for name in ("PKG-INFO", "setup.cfg")})
    egg_info = f"{stem}/{source['project'].replace('-', '_')}.egg-info"
    sdist_expected.update({
        f"{egg_info}/{name}": None
        for name in ("PKG-INFO", "SOURCES.txt", "dependency_links.txt", "requires.txt", "top_level.txt")
    })
    sdist_contents = archive_files(sdist, sdist_expected)
    for data in (
        wheel_contents[f"{dist_info}/METADATA"], sdist_contents[f"{stem}/PKG-INFO"],
        sdist_contents[f"{egg_info}/PKG-INFO"],
    ):
        package_metadata(data, project)
    return {wheel.name: sorted(wheel_contents), sdist.name: sorted(sdist_contents)}


def install_and_test(asset, source_dir, version, directory):
    print(f"Testing installed {asset.name}", flush=True)
    directory.mkdir()
    environment = directory / "venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", str(environment)], check=True, cwd=directory)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run = [str(python), "-I"]
    subprocess.run(
        [*run, "-m", "pip", "--disable-pip-version-check", "install", f"{asset}[test]"],
        check=True, cwd=directory,
    )
    versions = json.loads(subprocess.check_output([*run, "-c", CHECK_INSTALL, version], cwd=directory))
    for name in ("tests", "examples"):
        copytree(source_dir / name, directory / name)
    subprocess.run([*run, "-m", "pytest", "-q", "tests"], check=True, cwd=directory)
    for example in EXAMPLES:
        script, *args = example
        subprocess.run(
            [str(python), "-E", "-s", str(directory / "examples" / script), *args],
            check=True, cwd=directory,
        )
    return versions


def workflow_evidence(commit):
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return None
    env = os.environ
    require(commit == env["GITHUB_SHA"], "GitHub Actions SHA differs from the checkout.")
    event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    pr = event.get("pull_request")
    return {
        "run": f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}",
        "attempt": int(env["GITHUB_RUN_ATTEMPT"]),
        "job": env["GITHUB_JOB"],
        "event": env["GITHUB_EVENT_NAME"],
        "ref": env["GITHUB_REF"],
        "workflow_commit": env["GITHUB_WORKFLOW_SHA"],
        "pull_request_head": pr["head"]["sha"] if pr else None,
        "pull_request_base": pr["base"]["sha"] if pr else None,
    }


def verify_download(args, assets, source, hashes, sums):
    trusted = json.loads(args.manifest.read_bytes())
    require(trusted["format"] == 1 and trusted["release"] is None, "Expected an approved build manifest.")
    require(trusted["source"] == source, "Approved build source differs from the checkout.")
    require({name: item["sha256"] for name, item in trusted["artifacts"].items()} == hashes,
            "Downloaded packages differ from the approved build.")
    require((assets / CHECKSUMS).read_bytes() == sums
            and trusted["checksums_sha256"] == hashlib.sha256(sums).hexdigest(),
            "Downloaded SHA256SUMS differs.")
    require(args.tag == f"v{source['version']}", "Tag must match the package version.")
    ref = f"refs/tags/{args.tag}"
    tag_object = git("rev-parse", "--verify", ref).decode().strip()
    require(tag_object == args.tag_object, "Tag object differs from the approved tag.")
    require(git("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip() == source["commit"],
            "Tag does not resolve to the package commit.")
    return {"tag": args.tag, "tag_object": tag_object, "approved_ci": trusted["ci"]}


def require_files(directory, names):
    require({path.name for path in directory.iterdir()} == set(names), f"Unexpected files in {directory}.")
    for name in names:
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"Expected regular asset: {path}")
        require(path.stat().st_size <= MAX_ARCHIVE_BYTES, f"Asset exceeds size limit: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build and test candidate assets")
    build.add_argument("directory", type=Path, help="new output directory")
    download = commands.add_parser("verify-download", help="Test independently downloaded release assets")
    download.add_argument("directory", type=Path, help="directory containing only the three release assets")
    download.add_argument("--manifest", type=Path, required=True, help="approved prepublication manifest")
    download.add_argument("--tag", required=True)
    download.add_argument("--tag-object", required=True, help="previously recorded tag object SHA")
    download.add_argument("--receipt", type=Path, required=True, help="new receipt file outside the assets")
    args = parser.parse_args()
    args.directory = args.directory.resolve()

    with tempfile.TemporaryDirectory(prefix="anf-release-") as temp:
        work = Path(temp).resolve()
        source_dir = work / "source"
        source, source_files, project = snapshot(source_dir)
        building = args.command == "build"
        assets = work / "assets"
        wheel, sdist = package_paths(assets, source)
        names = [wheel.name, sdist.name]
        if building:
            require(not args.directory.exists(), f"Output directory already exists: {args.directory}")
            subprocess.run(
                [sys.executable, "-I", "-m", "build", "--outdir", str(assets)],
                check=True, cwd=source_dir,
            )
            require_files(assets, names)
        else:
            require(not args.receipt.exists() and not args.receipt.resolve().is_relative_to(args.directory),
                    "Use a new receipt path outside the downloaded assets.")
            require_files(args.directory, [*names, CHECKSUMS])
            assets.mkdir()
            for name in (*names, CHECKSUMS):
                copyfile(args.directory / name, assets / name)

        hashes = {asset.name: sha256(asset) for asset in (wheel, sdist)}
        sums = "".join(f"{hashes[name]}  {name}\n" for name in sorted(names)).encode("ascii")
        release = None if building else verify_download(args, assets, source, hashes, sums)
        files = verify_release(wheel, sdist, source, source_files, project)
        receipt = {
            "format": 1,
            "source": source,
            "artifacts": {name: {"sha256": hashes[name], "files": files[name]} for name in names},
            "checksums_sha256": hashlib.sha256(sums).hexdigest(),
            "ci": workflow_evidence(source["commit"]),
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "installations": {
                asset.name: install_and_test(asset, source_dir, source["version"], work / f"test-{index}")
                for index, asset in enumerate((wheel, sdist))
            },
            "release": release,
        }
        if building:
            (assets / CHECKSUMS).write_bytes(sums)
            (assets / MANIFEST).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            copytree(assets, args.directory)
            print(args.directory / MANIFEST)
        else:
            with args.receipt.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2)
                stream.write("\n")
            print(args.receipt)


if __name__ == "__main__":
    main()
