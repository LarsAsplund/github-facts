# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Test github_clone."""

from unittest import TestCase
from unittest.mock import patch, call
from tempfile import TemporaryDirectory
from pathlib import Path
from json import dumps
from stat import S_IRUSR
from github_clone import (
    BASIC_JSON_VERSION,
    clone_repo,
    get_basic_data,
    rmtree,
    zip_repo,
    clone_from_file_based_list,
    main,
)
from test.common import create_git_repo


class TestGithubClone(TestCase):
    """Tests github_clone.py."""

    def test_cloning_a_repo(self):
        path_names = ["file1.vhdl", "file2.vhd", "file3.sv", "file4.py", "file5.txt"]
        with TemporaryDirectory() as origin_path, TemporaryDirectory() as full_clone_path, TemporaryDirectory() as sparse_clone_path:
            create_git_repo(path_names, origin_path)
            clone_repo(Path(full_clone_path), Path(origin_path), sparse=False)
            clone_repo(Path(sparse_clone_path), Path(origin_path), sparse=True)

            path_names.append(".git")
            for path_name in path_names:
                self.assertTrue((Path(full_clone_path) / path_name).exists())
                self.assertTrue(
                    (Path(sparse_clone_path) / path_name).exists()
                    != path_name.endswith(".txt")
                )

    class SideEffect:
        """Mock for request response object."""

        def __init__(self, ok):
            self.ok = ok

        @staticmethod
        def json():
            return dict(data="bar_data")

        def __call__(self, url, page_size):
            return self

    @patch("github_clone.GithubSearch", autospec=True)
    def test_requesting_basic_data(self, github_search_mock):
        with TemporaryDirectory() as user_dir:
            basic_data_path = Path(user_dir) / f"bar.basic.{BASIC_JSON_VERSION}.json"
            github_search_mock.return_value.request.side_effect = self.SideEffect(True)
            basic_data = get_basic_data(
                "foo/bar", basic_data_path, "UserName", "access_token"
            )
            self.assertEqual(basic_data["data"], "bar_data")
            github_search_mock.assert_called_once_with(
                "UserName", "access_token", n_attempts=2
            )
            github_search_mock.return_value.request.assert_called_once_with(
                "https://api.github.com/repos/foo/bar", page_size=None
            )

    @patch("github_clone.GithubSearch", autospec=True)
    def test_failing_requesting_basic_data(self, github_search_mock):
        with TemporaryDirectory() as user_dir:
            basic_data_path = Path(user_dir) / f"bar.basic.{BASIC_JSON_VERSION}.json"
            github_search_mock.return_value.request.side_effect = self.SideEffect(False)
            basic_data = get_basic_data(
                "foo/bar", basic_data_path, "UserName", "access_token"
            )
            self.assertIsNone(basic_data)
            github_search_mock.assert_called_once_with(
                "UserName", "access_token", n_attempts=2
            )
            github_search_mock.return_value.request.assert_called_once_with(
                "https://api.github.com/repos/foo/bar", page_size=None
            )

    @patch("github_clone.GithubSearch", autospec=True)
    def test_getting_existing_basic_data(self, github_search_mock):
        with TemporaryDirectory() as user_dir:
            basic_data_path = Path(user_dir) / f"bar.basic.{BASIC_JSON_VERSION}.json"
            basic_data_path.write_text(dumps(dict(data="existing bar data")))

            basic_data = get_basic_data(
                "foo/bar", basic_data_path, "UserName", "access_token"
            )
            self.assertEqual(basic_data["data"], "existing bar data")
            github_search_mock.assert_called_once_with(
                "UserName", "access_token", n_attempts=2
            )

    def test_rmtree(self):
        with TemporaryDirectory() as test_dir:
            test_file = Path(test_dir) / "test_file"
            test_file.touch()
            self.assertTrue(test_file.exists())
            rmtree(test_dir)
            self.assertFalse(Path(test_dir).exists())

    def test_rmtree_with_read_only_file(self):
        with TemporaryDirectory() as test_dir:
            test_file = Path(test_dir) / "test_file"
            test_file.touch()
            test_file.chmod(S_IRUSR)
            self.assertTrue(test_file.exists())
            rmtree(test_dir)
            self.assertFalse(Path(test_dir).exists())

    def test_zipping_repo(self):
        path_names = ["file1.vhdl", "file2.vhd", "file3.sv", "file4.py", "file5.txt"]

        with TemporaryDirectory() as origin_path, TemporaryDirectory() as clone_path:
            create_git_repo(path_names, origin_path)
            clone_repo(clone_path, origin_path, sparse=False)
            zip_repo(Path(clone_path))
            self.assertFalse((Path(clone_path) / ".git").exists())
            self.assertTrue(
                (Path(clone_path).parent / (Path(clone_path).name + ".zip")).exists()
            )

    @patch("github_clone.zip_repo")
    @patch("github_clone.get_basic_data")
    @patch("github_clone.clone_repo")
    def test_end_to_end(self, clone_repo_mock, get_basic_data_mock, zip_repo_mock):
        def side_effect(repo_dir, _clone_url):
            if repo_dir.name == "exception":
                raise RuntimeError("Some exception")

            return True

        clone_repo_mock.side_effect = side_effect
        get_basic_data_mock.return_value = dict(data="basic_data")

        repos = [
            "JackyRen/vimrc",
            "Jaxc/Leon3test",
            "Jaxc/project",
            "already/cloned",
            "already/failed",
            "clone/exception",
        ]

        with TemporaryDirectory() as test_dir:
            repo_list_file = Path(test_dir) / "repo_list.txt"
            with open(repo_list_file, "w") as txt:
                txt.write("\n".join(repos) + "\n")

            with TemporaryDirectory() as repo_dir:
                (Path(repo_dir) / "already").mkdir()
                (Path(repo_dir) / "already" / "cloned.zip").touch()
                (Path(repo_dir) / "already" / "failed.failed").touch()

                clone_from_file_based_list(
                    repo_list_file, Path(repo_dir), "UserName", "access_token", False,
                )

                repos.remove("already/failed")
                repos.remove("clone/exception")
                basic_data_repos = [
                    arg[0][0] for arg in get_basic_data_mock.call_args_list
                ]
                self.assertEqual(basic_data_repos, repos)

                repos.remove("already/cloned")
                expected_repos = [call(Path(repo_dir) / repo) for repo in repos]

                self.assertEqual(zip_repo_mock.call_args_list, expected_repos)

                with open(Path(repo_dir) / "clone" / "exception.failed") as txt:
                    exception_reason = txt.read()
                    self.assertEqual(exception_reason, "Some exception")

    @staticmethod
    @patch(
        "sys.argv",
        [
            "github_clone.py",
            "path/to/repo_list",
            "path/to/repo_dir",
            "UserName",
            "access_token",
        ],
    )
    @patch("github_clone.clone_from_file_based_list")
    def test_cli(clone_mock):
        main()
        clone_mock.assert_called_once_with(
            Path("path") / "to" / "repo_list",
            Path("path") / "to" / "repo_dir",
            "UserName",
            "access_token",
            False,
            False,
        )
