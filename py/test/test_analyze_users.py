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
    update_timezones,
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
    def create_repo(repo_dir):
        subprocess.call(["git", "init"], cwd=repo_dir)
        subprocess.call(
            ["git", "config", "--local", "user.name", "Your Name"], cwd=repo_dir
        )
        subprocess.call(
            ["git", "config", "--local", "user.email", "you@example.com"],
            cwd=repo_dir,
        )

        repo = Repo(repo_dir)

        return repo

    @staticmethod
    def commit_files(
        repo_dir, seconds_after_epoch, timezone, author, files, rm_files=None
    ):
        for file, code in files.items():
            path = repo_dir / file
            parent = path.parent
            if parent != repo_dir:
                parent.mkdir(parents=True)
            path.write_text(code)
            subprocess.call(["git", "add", str(path)], cwd=str(repo_dir))
        if rm_files:
            for file in rm_files:
                path = repo_dir / file
                subprocess.call(["git", "rm", str(path)], cwd=str(repo_dir))
        os.environ["GIT_AUTHOR_DATE"] = (
            f"1970-01-02 00:00:0{seconds_after_epoch} %+03d00" % timezone
        )
        subprocess.call(
            ["git", "commit", "--author", f'"{author}"', "-m", "commit"],
            cwd=str(repo_dir),
        )

    def test_get_repo_user_data(self):
        with TemporaryDirectory() as repo_root:
            repo_dir = Path(repo_root) / "a" / "b"
            repo_dir.mkdir(parents=True)

            repo = self.create_repo(repo_dir)

            self.commit_files(
                repo_dir,
                0,
                -12,
                "a <a@b.c>",
                {"a.py": "", "c.vhdl": "", "g.txt": "", "h.vhd": ""},
            )

            self.commit_files(
                repo_dir,
                1,
                1,
                "b <c@d.e>",
                {"b.vhd": "library osvvm;\n", "e.py": "import cocotb\n"},
                ["h.vhd"],
            )

            self.commit_files(
                repo_dir,
                2,
                1,
                "b <c@d.e>",
                {
                    "b.vhd": "library osvvm;\nlibrary ieee;\n",
                    "c.vhdl": "library vunit_lib;\n",
                    "src/f.sv": "import uvm_pkg::foo;\n",
                },
            )

            self.commit_files(
                repo_dir,
                3,
                0,
                "c <f@g.h>",
                {"b.vhd": "library osvvm;\nlibrary ieee, foo;\n"},
            )

            self.commit_files(
                repo_dir,
                4,
                14,
                "d <i@j.k>",
                {"d.vhd": "library vunit_lib;\nlibrary uvvm_util;\n", "e.py": ""},
            )

            (
                repo_user_data,
                dropped_test_strategies,
            ) = get_repo_user_data(repo, [], None)
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
                dict(
                    osvvm=dict(
                        first_commit_time=86400 + 1 - 1 * 3600,
                        first_repo="a/b",
                        last_commit_time=86400 + 2 - 1 * 3600,
                        last_repo="a/b",
                    ),
                    vunit=dict(
                        first_commit_time=86400 + 2 - 1 * 3600,
                        first_repo="a/b",
                        last_commit_time=86400 + 2 - 1 * 3600,
                        last_repo="a/b",
                    ),
                    uvm=dict(
                        first_commit_time=86400 + 2 - 1 * 3600,
                        first_repo="a/b",
                        last_commit_time=86400 + 2 - 1 * 3600,
                        last_repo="a/b",
                    ),
                ),
            )
            self.assertDictEqual(repo_user_data["b <c@d.e>"]["timezones"], {1: 2})
            self.assertDictEqual(
                repo_user_data["c <f@g.h>"]["test_strategies"],
                dict(
                    osvvm=dict(
                        first_commit_time=86400 + 3 - 0 * 3600,
                        first_repo="a/b",
                        last_commit_time=86400 + 3 - 0 * 3600,
                        last_repo="a/b",
                    )
                ),
            )
            self.assertDictEqual(repo_user_data["c <f@g.h>"]["timezones"], {0: 1})
            self.assertDictEqual(
                repo_user_data["d <i@j.k>"]["test_strategies"],
                dict(
                    vunit=dict(
                        first_commit_time=86400 + 4 - 14 * 3600,
                        first_repo="a/b",
                        last_commit_time=86400 + 4 - 14 * 3600,
                        last_repo="a/b",
                    ),
                    uvvm=dict(
                        first_commit_time=86400 + 4 - 14 * 3600,
                        first_repo="a/b",
                        last_commit_time=86400 + 4 - 14 * 3600,
                        last_repo="a/b",
                    ),
                ),
            )
            self.assertDictEqual(repo_user_data["d <i@j.k>"]["timezones"], {14: 1})

            for data in repo_user_data.values():
                self.assertEqual(data["classification"], "professional")

    def test_update_timezones(self):
        user_data = dict()
        update_timezones(user_data, "user", {0: 1})
        self.assertDictEqual(user_data, dict(user=dict(timezones={0: 1})))
        user_data["user2"] = dict()
        update_timezones(user_data, "user2", {1: 2})
        self.assertDictEqual(
            user_data, dict(user=dict(timezones={0: 1}), user2=dict(timezones={1: 2}))
        )
        update_timezones(user_data, "user", {0: 2, 1: 3})
        self.assertDictEqual(
            user_data,
            dict(user=dict(timezones={0: 3, 1: 3}), user2=dict(timezones={1: 2})),
        )

    @classmethod
    def create_test_repos(cls, repo_root, classifications):
        user_dir = Path(repo_root) / "a"
        user = user_dir.name
        repo_dirs = dict()
        repos = dict()
        repo_list = []
        repo_classification = dict()
        for classification in classifications:
            repo_dir = user_dir / f"{classification}_repo"
            repo_dir.mkdir(parents=True)
            repo_dirs[classification] = repo_dir
            repos[classification] = cls.create_repo(repo_dirs[classification])
            repo_list.append(f"{user}/{classification}_repo")
            repo_classification[f"{user}/{classification}_repo"] = classification

        if "professional" in classifications:
            cls.commit_files(
                Path(repos["professional"].working_tree_dir),
                2,
                1,
                "author_1 <author_1@mail.com>",
                {"a.vhd": "library vunit_lib;\n"},
            )

        if "academic" in classifications:
            cls.commit_files(
                Path(repos["academic"].working_tree_dir),
                1,
                1,
                "author_1 <author_1@mail.com>",
                {"a.vhd": "library vunit_lib;\n"},
            )

            cls.commit_files(
                Path(repos["academic"].working_tree_dir),
                0,
                -12,
                "author_2 <author_2@mail.com>",
                {"b.sv": "import uvm_pkg::*;\n"},
            )

            cls.commit_files(
                Path(repos["academic"].working_tree_dir),
                1,
                0,
                "author_2 <author_2@mail.com>",
                {"c.py": "import cocotb\n"},
            )

        if "unknown" in classifications:
            cls.commit_files(
                Path(repos["unknown"].working_tree_dir),
                0,
                2,
                "author_1 <author_1@mail.com>",
                {"a.vhd": "library vunit_lib;\n"},
            )

            cls.commit_files(
                Path(repos["unknown"].working_tree_dir),
                1,
                14,
                "author_2 <author_2@mail.com>",
                {"b.sv": "import uvm_pkg::*;\n"},
            )

            cls.commit_files(
                Path(repos["unknown"].working_tree_dir),
                0,
                0,
                "author_2 <author_2@mail.com>",
                {"c.py": "import cocotb\n"},
            )

            cls.commit_files(
                Path(repos["unknown"].working_tree_dir),
                1,
                -7,
                "author_3 <author_3@mail.com>",
                {"d.vhd": "library osvvm;\n"},
            )

        return user_dir, repo_dirs, repos, repo_list, repo_classification

    def test_get_user_data(self):
        classifications = ["professional", "academic", "unknown"]
        with TemporaryDirectory() as repo_root:
            (
                user_dir,
                repo_dirs,
                repos,
                repo_list,
                repo_classification,
            ) = self.create_test_repos(repo_root, classifications)

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
                dict(
                    vunit=dict(
                        first_commit_time=86400 + 0 - 2 * 3600,
                        first_repo="a/unknown_repo",
                        last_commit_time=86400 + 2 - 1 * 3600,
                        last_repo="a/professional_repo",
                    )
                ),
            )
            self.assertDictEqual(
                user_data["author_1 <author_1@mail.com>"]["timezones"],
                {1: 2, 2: 1},
            )
            self.assertDictEqual(
                user_data["author_2 <author_2@mail.com>"]["test_strategies"],
                dict(
                    cocotb=dict(
                        first_commit_time=86400 + 0 - 0 * 3600,
                        first_repo="a/unknown_repo",
                        last_commit_time=86400 + 1 - 0 * 3600,
                        last_repo="a/academic_repo",
                    ),
                    uvm=dict(
                        first_commit_time=86400 + 1 - 14 * 3600,
                        first_repo="a/unknown_repo",
                        last_commit_time=86400 + 0 + 12 * 3600,
                        last_repo="a/academic_repo",
                    ),
                ),
            )
            self.assertDictEqual(
                user_data["author_2 <author_2@mail.com>"]["timezones"],
                {-12: 1, 0: 2, 14: 1},
            )
            self.assertDictEqual(
                user_data["author_3 <author_3@mail.com>"]["test_strategies"],
                dict(
                    osvvm=dict(
                        first_commit_time=86400 + 1 + 7 * 3600,
                        first_repo="a/unknown_repo",
                        last_commit_time=86400 + 1 + 7 * 3600,
                        last_repo="a/unknown_repo",
                    )
                ),
            )
            self.assertDictEqual(
                user_data["author_3 <author_3@mail.com>"]["timezones"],
                {-7: 1},
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
            "james.bond <top@secret.uk>": dict(
                test_strategies=dict(
                    vunit=dict(
                        first_commit_time=0,
                        first_repo="a/b",
                        last_commit_time=0,
                        last_repo="a/b",
                    ),
                    osvvm=dict(
                        first_commit_time=17,
                        first_repo="c/d",
                        last_commit_time=18,
                        last_repo="x/y",
                    ),
                )
            ),
            "James Bond <007@mi6.uk>": dict(
                test_strategies=dict(
                    vunit=dict(
                        first_commit_time=1,
                        first_repo="c/d",
                        last_commit_time=19,
                        last_repo="c/d",
                    ),
                    uvvm=dict(
                        first_commit_time=21,
                        first_repo="e/f",
                        last_commit_time=21,
                        last_repo="e/f",
                    ),
                )
            ),
            "undercover <007@mi6.uk>": dict(
                test_strategies=dict(
                    uvm=dict(
                        first_commit_time=13,
                        first_repo="a/b",
                        last_commit_time=13,
                        last_repo="a/b",
                    ),
                    uvvm=dict(
                        first_commit_time=7,
                        first_repo="e/f",
                        last_commit_time=7,
                        last_repo="u/v",
                    ),
                )
            ),
            "ian.flemming <author@somewhere.uk": dict(
                test_strategies=dict(
                    cocotb=dict(
                        first_commit_time=100,
                        first_repo="g/h",
                        last_commit_time=101,
                        last_repo="n/m",
                    )
                )
            ),
        }
        user_aliases = {
            "james.bond <top@secret.uk>": "James Bond <007@mi6.uk>",
            "undercover <007@mi6.uk>": "James Bond <007@mi6.uk>",
        }

        user_experience = remove_aliases(user_experience, user_aliases)

        self.assertEqual(len(user_experience), 2)
        self.assertDictEqual(
            user_experience["James Bond <007@mi6.uk>"],
            dict(
                test_strategies=dict(
                    vunit=dict(
                        first_commit_time=0,
                        first_repo="a/b",
                        last_commit_time=19,
                        last_repo="c/d",
                    ),
                    osvvm=dict(
                        first_commit_time=17,
                        first_repo="c/d",
                        last_commit_time=18,
                        last_repo="x/y",
                    ),
                    uvvm=dict(
                        first_commit_time=7,
                        first_repo="e/f",
                        last_commit_time=21,
                        last_repo="e/f",
                    ),
                    uvm=dict(
                        first_commit_time=13,
                        first_repo="a/b",
                        last_commit_time=13,
                        last_repo="a/b",
                    ),
                )
            ),
        )
        self.assertDictEqual(
            user_experience["ian.flemming <author@somewhere.uk"],
            dict(
                test_strategies=dict(
                    cocotb=dict(
                        first_commit_time=100,
                        first_repo="g/h",
                        last_commit_time=101,
                        last_repo="n/m",
                    )
                )
            ),
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
