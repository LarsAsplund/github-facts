# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Test github_clone."""

from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import subprocess
import os
from json import dump, load
from test.common import create_git_repo
from git import Repo
from analyze_users import (
    get_repos_using_a_std_framework,
    get_source_files,
    get_repo_user_data,
    get_potential_aliases,
    remove_aliases,
    get_user_data,
    USER_JSON_VERSION,
    analyze,
    main,
)


class TestAnalyzeUsers(TestCase):
    """Tests analyze_users.py."""

    @staticmethod
    def prevent_repo_from_locking_files_to_be_deleted(repo):
        repo.git.clear_cache()

    def test_get_repos_using_a_std_framework(self):
        repos_stat = dict(
            repo_stat={
                "a/b": dict(test_strategies=["vunit"]),
                "c/d": dict(test_strategies=[]),
                "e/f": dict(test_strategies=["uvm"]),
            }
        )
        std_repos = get_repos_using_a_std_framework(repos_stat)
        self.assertCountEqual(std_repos, ["a/b", "e/f"])

    def test_get_source_files(self):
        with TemporaryDirectory() as repo_dir:
            file_names = ["a.txt", "b.py", "c.sv", "d.vhdl", "e.vhd", "f/g.py"]
            create_git_repo(file_names, repo_dir)
            repo = Repo(repo_dir)
            source_files = get_source_files(repo)
            self.assertCountEqual(
                source_files, ["b.py", "c.sv", "d.vhdl", "e.vhd", "f/g.py"]
            )

            self.prevent_repo_from_locking_files_to_be_deleted(repo)

    @staticmethod
    def _create_repo(repo_dir):
        subprocess.call(["git", "init"], cwd=repo_dir)
        subprocess.call(
            ["git", "config", "--local", "user.name", "Your Name"], cwd=repo_dir
        )
        subprocess.call(
            ["git", "config", "--local", "user.email", "you@example.com"], cwd=repo_dir,
        )

        repo = Repo(repo_dir)

        return repo

    @staticmethod
    def _commit_files(repo_dir, seconds_after_epoch, author, files, rm_files=None):
        for file, code in files.items():
            path = Path(repo_dir) / file
            parent = path.parent
            if parent != Path(repo_dir):
                parent.mkdir(parents=True)
            path.write_text(code)
            subprocess.call(["git", "add", str(path)], cwd=repo_dir)
        if rm_files:
            for file in rm_files:
                path = Path(repo_dir) / file
                subprocess.call(["git", "rm", str(path)], cwd=repo_dir)
        os.environ["GIT_AUTHOR_DATE"] = f"1970-01-01 00:00:0{seconds_after_epoch} +0000"
        subprocess.call(
            ["git", "commit", "--author", f'"{author}"', "-m", "commit"], cwd=repo_dir,
        )

    def test_get_repo_user_data(self):

        with TemporaryDirectory() as repo_dir:
            repo = self._create_repo(repo_dir)

            self._commit_files(
                repo_dir,
                0,
                "a <a@b.c>",
                {"a.py": "", "c.vhdl": "", "g.txt": "", "h.vhd": ""},
            )

            self._commit_files(
                repo_dir,
                1,
                "b <c@d.e>",
                {"b.vhd": "library osvvm;\n", "e.py": "import cocotb\n"},
                ["h.vhd"],
            )

            self._commit_files(
                repo_dir,
                2,
                "b <c@d.e>",
                {
                    "b.vhd": "library osvvm;\nlibrary ieee;\n",
                    "c.vhdl": "library vunit_lib;\n",
                    "src/f.sv": "import uvm_pkg::foo;\n",
                },
            )

            self._commit_files(
                repo_dir,
                3,
                "c <f@g.h>",
                {"b.vhd": "library osvvm;\nlibrary ieee, foo;\n"},
            )

            self._commit_files(
                repo_dir,
                4,
                "d <i@j.k>",
                {"d.vhd": "library vunit_lib;\nlibrary uvvm_util;\n", "e.py": ""},
            )

            (repo_user_data, dropped_test_strategies,) = get_repo_user_data(
                repo, [], None
            )
            self.assertCountEqual(dropped_test_strategies, [])
            self.assertDictEqual(repo_user_data, dict())

            (repo_user_data, dropped_test_strategies,) = get_repo_user_data(
                repo,
                ["a.py", "b.vhd", "c.vhdl", "d.vhd", "e.py", "src/f.sv"],
                "professional",
            )

            self.prevent_repo_from_locking_files_to_be_deleted(repo)

            self.assertCountEqual(dropped_test_strategies, ["cocotb"])
            self.assertEqual(len(repo_user_data), 3)
            self.assertDictEqual(
                repo_user_data["b <c@d.e>"]["test_strategies"],
                dict(osvvm=1.0, vunit=2.0, uvm=2.0),
            )
            self.assertDictEqual(
                repo_user_data["c <f@g.h>"]["test_strategies"], dict(osvvm=3.0)
            )
            self.assertDictEqual(
                repo_user_data["d <i@j.k>"]["test_strategies"],
                dict(vunit=4.0, uvvm=4.0),
            )

            for data in repo_user_data.values():
                self.assertEqual(data["classification"], "professional")

    def test_get_user_data(self):
        with TemporaryDirectory() as user_dir:
            user = Path(user_dir).name
            repo_dirs = dict()
            repos = dict()
            classifications = ["professional", "academic", "unknown"]
            repo_list = []
            repo_classification = dict()
            for classification in classifications:
                repo_dir = Path(user_dir) / f"{classification}_repo"
                repo_dir.mkdir()
                repo_dirs[classification] = repo_dir
                repos[classification] = self._create_repo(repo_dirs[classification])
                repo_list.append(f"{user}/{classification}_repo")
                repo_classification[f"{user}/{classification}_repo"] = classification

            self._commit_files(
                repos["professional"].working_tree_dir,
                2,
                "author_1 <author_1@mail.com>",
                {"a.vhd": "library vunit_lib;\n"},
            )

            self._commit_files(
                repos["academic"].working_tree_dir,
                1,
                "author_1 <author_1@mail.com>",
                {"a.vhd": "library vunit_lib;\n"},
            )

            self._commit_files(
                repos["academic"].working_tree_dir,
                0,
                "author_2 <author_2@mail.com>",
                {"b.sv": "import uvm_pkg::*;\n"},
            )

            self._commit_files(
                repos["academic"].working_tree_dir,
                1,
                "author_2 <author_2@mail.com>",
                {"c.py": "import cocotb\n"},
            )

            self._commit_files(
                repos["unknown"].working_tree_dir,
                0,
                "author_1 <author_1@mail.com>",
                {"a.vhd": "library vunit_lib;\n"},
            )

            self._commit_files(
                repos["unknown"].working_tree_dir,
                1,
                "author_2 <author_2@mail.com>",
                {"b.sv": "import uvm_pkg::*;\n"},
            )

            self._commit_files(
                repos["unknown"].working_tree_dir,
                0,
                "author_2 <author_2@mail.com>",
                {"c.py": "import cocotb\n"},
            )

            self._commit_files(
                repos["unknown"].working_tree_dir,
                1,
                "author_3 <author_3@mail.com>",
                {"d.vhd": "library osvvm;\n"},
            )

            user_data = get_user_data(
                Path(user_dir).parent, repo_list, repo_classification, dict(), False
            )

            for classification in classifications:
                self.prevent_repo_from_locking_files_to_be_deleted(
                    repos[classification]
                )

            self.assertEqual(len(user_data), 3)

            self.assertEqual(
                user_data["author_1 <author_1@mail.com>"]["classification"],
                "professional",
            )
            self.assertEqual(
                user_data["author_2 <author_2@mail.com>"]["classification"], "academic"
            )
            self.assertEqual(
                user_data["author_3 <author_3@mail.com>"]["classification"], "unknown"
            )

            for classification in classifications:
                repo_name = repo_dirs[classification].name
                self.assertTrue(
                    (
                        repo_dirs[classification].parent
                        / f"{repo_name}.user.{USER_JSON_VERSION}.json"
                    ).exists()
                )

            self.assertDictEqual(
                user_data["author_1 <author_1@mail.com>"]["test_strategies"],
                dict(vunit=0),
            )
            self.assertDictEqual(
                user_data["author_2 <author_2@mail.com>"]["test_strategies"],
                dict(uvm=0, cocotb=0),
            )
            self.assertDictEqual(
                user_data["author_3 <author_3@mail.com>"]["test_strategies"],
                dict(osvvm=1),
            )

    def test_get_potential_aliases(self):
        user_experience = {
            "james.bond <top@secret.uk>": None,
            "James Bond <007@mi6.uk>": None,
            "undercover <007@mi6.uk>": None,
            "ian.flemming <author@somewhere.uk": None,
        }
        potential_aliases = get_potential_aliases(user_experience)
        self.assertCountEqual(
            potential_aliases,
            [
                ("james.bond <top@secret.uk>", "James Bond <007@mi6.uk>"),
                ("James Bond <007@mi6.uk>", "undercover <007@mi6.uk>"),
            ],
        )

    def test_remove_aliases(self):
        user_experience = {
            "james.bond <top@secret.uk>": dict(test_strategies=dict(vunit=0, osvvm=17)),
            "James Bond <007@mi6.uk>": dict(test_strategies=dict(vunit=1, uvvm=21)),
            "undercover <007@mi6.uk>": dict(test_strategies=dict(uvm=13, uvvm=7)),
            "ian.flemming <author@somewhere.uk": dict(test_strategies=dict(cocotb=100)),
        }
        user_aliases = {
            "james.bond <top@secret.uk>": "James Bond <007@mi6.uk>",
            "undercover <007@mi6.uk>": "James Bond <007@mi6.uk>",
        }

        user_experience = remove_aliases(user_experience, user_aliases)

        self.assertEqual(len(user_experience), 2)
        self.assertDictEqual(
            user_experience["James Bond <007@mi6.uk>"],
            dict(test_strategies=dict(vunit=0, osvvm=17, uvvm=7, uvm=13)),
        )
        self.assertDictEqual(
            user_experience["ian.flemming <author@somewhere.uk"],
            dict(test_strategies=dict(cocotb=100)),
        )

    @patch("analyze_users.get_potential_aliases")
    @patch("analyze_users.get_user_data")
    @patch("analyze_users.clone")
    @patch("analyze_users.get_repos_using_a_std_framework")
    def test_analyze(
        self,
        get_repos_using_a_std_framework_mock,
        clone_mock,
        get_user_data_mock,
        get_potential_aliases_mock,
    ):
        with TemporaryDirectory() as temp_dir:
            repos_stat_path = Path(temp_dir) / "repos_stat"
            with open(repos_stat_path, "w") as json:
                dump("repos_stat", json)

            repo_classification_path = Path(temp_dir) / "repo_classification"
            with open(repo_classification_path, "w") as json:
                dump("repo_classification", json)

            user_aliases_path = Path(temp_dir) / "user_aliases"
            with open(user_aliases_path, "w") as json:
                dump("user_aliases", json)

            get_repos_using_a_std_framework_mock.return_value = "repo_list"
            get_user_data_mock.return_value = "user_data"
            get_potential_aliases_mock.return_value = [
                ("john.doe <jd@mail.com>", "John Doe <jd@mail.com>")
            ]

            analyze(
                Path(temp_dir),
                repos_stat_path,
                repo_classification_path,
                user_aliases_path,
                Path(temp_dir) / "output.json",
                False,
                "GithubUser",
                "github_access_token",
            )

            get_repos_using_a_std_framework_mock.assert_called_once_with("repos_stat")

            clone_mock.assert_called_once_with(
                "repo_list",
                Path(temp_dir),
                "GithubUser",
                "github_access_token",
                retry=False,
                no_zip=True,
            )

            get_user_data_mock.assert_called_once_with(
                Path(temp_dir),
                "repo_list",
                "repo_classification",
                "user_aliases",
                False,
            )

            get_potential_aliases_mock.assert_called_once_with("user_data")

            with open(Path(temp_dir) / "output.json") as json:
                output = load(json)
                self.assertEqual(output, "user_data")

    @staticmethod
    @patch(
        "sys.argv",
        [
            "analyze_users.py",
            "--redo",
            "path/to/repos_root",
            "path/to/repos_stat",
            "path/to/repo_classification",
            "path/to/user_aliases",
            "path/to/output",
            "GithubUser",
            "github_access_token",
        ],
    )
    @patch("analyze_users.analyze")
    def test_cli(analyze_mock):
        main()
        analyze_mock.assert_called_once_with(
            Path("path") / "to" / "repos_root",
            Path("path") / "to" / "repos_stat",
            Path("path") / "to" / "repo_classification",
            Path("path") / "to" / "user_aliases",
            Path("path") / "to" / "output",
            True,
            "GithubUser",
            "github_access_token",
        )
