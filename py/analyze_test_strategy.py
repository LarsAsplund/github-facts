# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Script for analyzing cloned repos with respect to what test framework being used."""

import zipfile
import argparse
from json import dump, load
from pathlib import Path
import re
from os import walk
from datetime import datetime
from github_clone import rmtree, BASIC_JSON_VERSION

TEST_JSON_VERSION = 2

VHDL_LIB_RE = re.compile(
    r"""
\b
library
\s+
(?P<id>[a-zA-Z][\w]*)
(?P<extra>(\s*,\s*[a-zA-Z][\w]*)*)
\s*
;
""",
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)

COCOTB_IMPORT_RE = re.compile(
    r"""
\b
import
\s+
(?P<id>[a-zA-Z][\w]*)
(?P<extra>(\s*,\s*[a-zA-Z][\w]*)*)
""",
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)
COCOTB_FROM_RE = re.compile(
    r"\bfrom\s+cocotb\b", re.MULTILINE | re.IGNORECASE | re.VERBOSE
)

UVM_IMPORT_RE = re.compile(
    r"""
\b
import
\s+
(?P<id>[a-zA-Z][\w]*)\s*::\s*(\*|([a-zA-Z][\w]*))
(?P<extra>(\s*,\s*[a-zA-Z][\w]*\s*::\s*(\*|([a-zA-Z][\w]*)))*)
\s*
;
""",
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)


def analyze_test_strategy(code, ext):
    """Find and return all test frameworks being used in the code."""
    ext = ext.lower()
    code = remove_comments(code, ext).lower()
    test_strategy = set()
    if ext in [".vhd", ".vhdl"]:
        libs = set()
        for match in VHDL_LIB_RE.finditer(code):
            libs.add(match.group("id"))
            if match.group("extra"):
                extras = match.group("extra").split(",")
                for index in range(1, len(extras)):
                    libs.add(extras[index].strip())

        if "vunit_lib" in libs:
            test_strategy.add("vunit")
        if "osvvm" in libs:
            test_strategy.add("osvvm")

        for lib in libs:
            if lib.startswith("osvvm"):
                test_strategy.add("osvvm")
            if lib.startswith("uvvm"):
                test_strategy.add("uvvm")
            if lib.startswith("bitvis"):
                test_strategy.add("uvvm")
    elif ext == ".py":
        if COCOTB_FROM_RE.search(code):
            test_strategy.add("cocotb")
        else:
            modules = set()
            for match in COCOTB_IMPORT_RE.finditer(code):
                modules.add(match.group("id"))
                if match.group("extra"):
                    extras = match.group("extra").split(",")
                    for index in range(1, len(extras)):
                        modules.add(extras[index].strip())

            if "cocotb" in modules:
                test_strategy.add("cocotb")
    elif ext == ".sv":
        packages = set()
        for match in UVM_IMPORT_RE.finditer(code):
            packages.add(match.group("id"))
            if match.group("extra"):
                extras = match.group("extra").split(",")
                for index in range(1, len(extras)):
                    packages.add(extras[index].split("::")[0].strip())

        if "uvm_pkg" in packages:
            test_strategy.add("uvm")

    return test_strategy


VHDL_REMOVE_COMMENT_RE = re.compile(r"--[^\n]*")
PY_REMOVE_COMMENT_RE = re.compile(r"#[^\n]*")
SV_REMOVE_COMMENT_RE = re.compile(r"//[^\n]*")


def remove_comments(code, ext):
    """Return the code with comments removed."""
    if ext in [".vhd", ".vhdl"]:
        return VHDL_REMOVE_COMMENT_RE.sub("", code)

    if ext == ".py":
        return PY_REMOVE_COMMENT_RE.sub("", code)

    if ext == ".sv":
        return SV_REMOVE_COMMENT_RE.sub("", code)

    raise RuntimeError("Unknown extension %s" % ext)


def get_source_files(repo_dir, extensions=None):
    """
    Find and return all files matching the given extensions in repo_dir.

    If no extensions are given find files with extensions vhd, vhdl, py and sv.
    """
    if extensions is None:
        extensions = [".vhd", ".vhdl", ".py", ".sv"]
    source_files = []
    for root, _dirs, files in walk(repo_dir):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                source_files.append((Path(root) / Path(file)).relative_to(repo_dir))

    return source_files


def analyze_repo(repo_dir):
    """Collect verification related for the repository in the given directory."""
    source_files = get_source_files(repo_dir)

    n_vhdl_files = 0
    repo_has_tests = False
    test_strategies = set()

    for source_file in source_files:
        code = (repo_dir / source_file).read_text(encoding="latin-1")
        test_strategies |= analyze_test_strategy(code, source_file.suffix)

        if source_file.suffix.lower() in [".vhd", ".vhdl"]:
            n_vhdl_files += 1
            repo_has_tests |= (
                "test" in source_file.name.lower() or "tb" in source_file.name.lower()
            )

    if n_vhdl_files == 0:
        test_strategies = set()

    repo_stat = dict(
        version=TEST_JSON_VERSION,
        n_vhdl_files=n_vhdl_files,
        has_tests=repo_has_tests,
        test_strategies=list(test_strategies),
    )

    return repo_stat


def remove_fp(repo_full_name, repo_stat, fp_repos):
    """Remove false positive repositories."""
    user_name = repo_full_name.split("/")[0]
    if "%s/*" % user_name in fp_repos:
        for framework in fp_repos["%s/*" % user_name]:
            if framework in repo_stat["test_strategies"]:
                repo_stat["test_strategies"].remove(framework)

    if repo_full_name in fp_repos:
        for framework in fp_repos[repo_full_name]:
            if framework in repo_stat["test_strategies"]:
                repo_stat["test_strategies"].remove(framework)

    return repo_stat


class GithubStat:
    """Class accumulating statistics from all analyzed repositories."""

    def __init__(self):
        self._data = dict(
            test_strategy_numbers=dict(
                vunit=0, osvvm=0, uvvm=0, cocotb=0, uvm=0, unknown=0, other=0
            ),
            repo_stat=dict(),
            created_at=dict(),
            total_n_vhdl_files=0,
            total_n_repos=0,
            n_repos_with_standard_vhdl_strategy=0,
            n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files=0,
        )

    @property
    def stat(self):
        """Return collected stats as property."""
        return self._data

    def update(self, basic_data, user_name, repo_name, repo_stat):
        """Update with statistics from the given repo."""
        if not basic_data:
            return

        if "created_at" not in basic_data:
            return

        repo_full_name = "%s/%s" % (user_name, repo_name)

        if not repo_stat["has_tests"] and not repo_stat["test_strategies"]:
            self._data["test_strategy_numbers"]["unknown"] += 1
        elif not repo_stat["test_strategies"]:
            self._data["test_strategy_numbers"]["other"] += 1
        else:
            repo_with_standard_vhdl_strategy = False
            repo_with_standard_vhdl_strategy_but_no_obvious_test_files = False
            for strategy in ["vunit", "osvvm", "uvvm", "cocotb", "uvm"]:
                if strategy in repo_stat["test_strategies"]:
                    self._data["test_strategy_numbers"][strategy] += 1
                    print("%s uses %s" % (repo_full_name, strategy))

                    if strategy in ["vunit", "osvvm", "uvvm"]:
                        repo_with_standard_vhdl_strategy = True
                        if not repo_stat["has_tests"]:
                            repo_with_standard_vhdl_strategy_but_no_obvious_test_files = (
                                True
                            )
            if repo_with_standard_vhdl_strategy:
                self._data["n_repos_with_standard_vhdl_strategy"] += 1
            if repo_with_standard_vhdl_strategy_but_no_obvious_test_files:
                self._data[
                    "n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"
                ] += 1
        self._data["total_n_vhdl_files"] += repo_stat["n_vhdl_files"]
        self._data["total_n_repos"] += 1
        self._data["repo_stat"][repo_full_name] = repo_stat

        self._data["created_at"][repo_full_name] = int(
            datetime.strptime(
                basic_data["created_at"], "%Y-%m-%dT%H:%M:%SZ"
            ).timestamp()
        )

    def dump(self, output_path):
        """Write statistics to a JSON file."""
        with open(output_path, "w") as json:
            dump(self._data, json)

    def __str__(self):
        """Return string representation for the most basic statistics."""
        result = "%d repos, %d files\n" % (
            self._data["total_n_repos"],
            self._data["total_n_vhdl_files"],
        )
        result += "Number of repos for each strategy:\n"
        result += str(self._data["test_strategy_numbers"]) + "\n"
        result += (
            "Number of repos using a VHDL strategy is %d but %d of them have no obvious test file\n"
            % (
                self._data["n_repos_with_standard_vhdl_strategy"],
                self._data[
                    "n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"
                ],
            )
        )

        return result


def analyze(repos_root, output_path, fp_path, redo):
    """Analyze all repos under the given root directory."""
    github_stat = GithubStat()
    total_n_repos = 0

    if fp_path:
        with open(fp_path) as json:
            fp_repos = load(json)
    else:
        fp_repos = None

    for root, _dirs, files in walk(repos_root):
        for file in files:
            if Path(file).suffix == ".zip":
                total_n_repos += 1
                if total_n_repos % 100 == 0:
                    print("Analyzed %d repos" % total_n_repos)
                repo_name = Path(file).stem
                user_dir = (Path(root) / Path(file)).parent
                user_name = user_dir.name
                repo_full_name = "%s/%s" % (user_name, repo_name)

                repo_stat = None
                repo_dir = None

                test_stat_path = Path(root) / (
                    repo_name + f".test.{TEST_JSON_VERSION}.json"
                )

                if test_stat_path.exists() and not redo:
                    with open(test_stat_path) as json:
                        repo_stat = load(json)

                else:
                    with zipfile.ZipFile(Path(root) / Path(file)) as zip_file:
                        zip_file.extractall(Path(root) / "unzipped")

                    repo_dir = Path(root) / "unzipped"
                    repo_stat = analyze_repo(repo_dir)

                    with open(test_stat_path, "w",) as json:
                        dump(repo_stat, json)

                if fp_repos:
                    repo_stat = remove_fp(repo_full_name, repo_stat, fp_repos)

                basic_data_path = (
                    repos_root
                    / user_name
                    / f"{repo_name}.basic.{BASIC_JSON_VERSION}.json"
                )
                if basic_data_path.exists():
                    with open(basic_data_path) as json:
                        basic_data = load(json)
                        github_stat.update(basic_data, user_name, repo_name, repo_stat)

                if (Path(root) / "unzipped").exists():
                    rmtree(str(Path(root) / "unzipped"))

    github_stat.dump(output_path)
    print(github_stat)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyses all cloned repositories for test strategy and store a metadata"
        "JSON file in each repository"
    )
    parser.add_argument(
        "repo_dir",
        help="Directory where the cloned repositories are located",
        type=Path,
    )
    parser.add_argument(
        "output_path",
        help="Path to JSON file listing the analysis result for all repositories",
        type=Path,
    )
    parser.add_argument(
        "--fp", help="Path to JSON file with false positives", type=Path
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        default=False,
        help="Redo repository analysis even if previous results exists",
    )

    args = parser.parse_args()

    analyze(
        args.repo_dir, args.output_path, args.fp, args.redo,
    )


if __name__ == "__main__":
    main()
