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
from json import load, dump
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
from test.common import create_git_repo
from github_clone import BASIC_JSON_VERSION
from analyze_test_strategy import (
    TEST_JSON_VERSION,
    remove_comments,
    analyze_test_strategy,
    get_source_files,
    analyze_repo,
    remove_fp,
    GithubStat,
    analyze,
    main,
)


class TestAnalyzeTestStrategy(TestCase):
    """Tests analyze_test_strategy.py."""

    def test_removing_comments(self):
        code = """\
{tag} Comment
  {tag}Comment
code
  code {tag}  Comment
    code
{tag} Comment"""
        code_without_comments = """\

  
code
  code 
    code
"""
        self.assertEqual(
            remove_comments(code.format(tag="--"), ".vhd"), code_without_comments,
        )
        self.assertEqual(
            remove_comments(code.format(tag="--"), ".vhdl"), code_without_comments,
        )
        self.assertEqual(
            remove_comments(code.format(tag="//"), ".sv"), code_without_comments,
        )
        self.assertEqual(
            remove_comments(code.format(tag="#"), ".py"), code_without_comments,
        )

    def test_analyze_vhdl_test_strategy(self):
        vhdl_code = """\
library foo;
library {lib};
library bar;
"""
        test_strategies = ["vunit", "osvvm", "osvvm", "uvvm", "uvvm"]
        for lib, strategy in zip(
            [
                "vunit_lib",
                "OSVVM",
                "osvvm_something",
                "Uvvm_something",
                "bitvis_someting",
            ],
            test_strategies,
        ):
            for ext in [".vhd", ".vhdl"]:
                found_strategies = analyze_test_strategy(vhdl_code.format(lib=lib), ext)
                self.assertCountEqual([strategy], found_strategies)

    def test_analyze_vhdl_test_strategy_with_library_lists(self):
        vhdl_code = """\
library foo;
library  vunit_lib, bar;
library spam, osvvm ;
library ying,uvvm_util  , yang;
"""
        found_strategies = analyze_test_strategy(vhdl_code, ".vhd")
        self.assertCountEqual(found_strategies, ["vunit", "osvvm", "uvvm"])

    def test_analyze_system_verilog_test_strategy(self):
        system_verilog_codes = [
            """\
import foo::*;
import uvm_pkg::*;
import bar::*;
""",
            "import  uvm_pkg::foo, bar::*;",
            "import foo::* ,uvm_pkg :: * ;",
            "import foo::*, uvm_pkg::spam, bar::*;",
        ]

        for system_verilog_code in system_verilog_codes:
            found_strategies = analyze_test_strategy(system_verilog_code, ".sv")
            self.assertCountEqual(["uvm"], found_strategies)

    def test_analyze_python_test_strategy(self):
        cocotb_codes = [
            """\
import foo
import cocotb
import bar
""",
            "import  cocotb, foo",
            "import bar, cocotb",
            "import spam,cocotb , eggs",
        ]
        for cocotb_code in cocotb_codes:
            found_strategies = analyze_test_strategy(cocotb_code, ".py")
            self.assertCountEqual(["cocotb"], found_strategies)

        cocotb_code = """\
import foo
from cocotb import something
import bar
"""
        found_strategies = analyze_test_strategy(cocotb_code, ".py")
        self.assertCountEqual(["cocotb"], found_strategies)

    def test_analyze_multiple_test_strategies(self):
        vhdl_code = """\
library vunit_lib;
library OSVVM;
"""
        found_strategies = analyze_test_strategy(vhdl_code, ".vhd")
        self.assertCountEqual(["vunit", "osvvm"], found_strategies)

    def test_analyze_no_test_strategies(self):
        vhdl_code = """\
library foo;
library bar;
"""
        found_strategies = analyze_test_strategy(vhdl_code, ".vhd")
        self.assertEqual(len(found_strategies), 0)

    def test_get_source_files(self):
        with TemporaryDirectory() as repo_dir:
            file_names = [
                Path("file1.vhdl"),
                Path("file2.vhd"),
                Path("file3.sv"),
                Path("dir") / "dir" / "file4.py",
                Path("file5.txt"),
            ]
            create_git_repo(file_names, repo_dir)
            source_files = get_source_files(repo_dir)
            for file_name in file_names:
                if file_name.name != "file5.txt":
                    self.assertIn(file_name, source_files)

            source_files = get_source_files(repo_dir, [".v"])
            self.assertFalse(source_files)

            source_files = get_source_files(repo_dir, [".py"])
            self.assertEqual(len(source_files), 1)
            self.assertIn(Path("dir") / "dir" / "file4.py", source_files)

    @patch("analyze_test_strategy.get_source_files")
    def test_analyze_repo(self, get_source_files_mock):
        with TemporaryDirectory() as repo_dir:
            source_files = {
                Path("foo") / "bar.vhd": "",
                Path("tb_bar.vhd"): "library vunit_lib;",
                Path("spam.VHDL"): "library OSVVM;",
                Path("egg.sv"): "",
            }
            for source_file, code in source_files.items():
                if source_file.parent != Path("."):
                    (Path(repo_dir) / source_file.parent).mkdir()
                (Path(repo_dir) / source_file).write_text(code)

            get_source_files_mock.return_value = source_files.keys()

            repo_stat = analyze_repo(Path(repo_dir))

            get_source_files_mock.assert_called_once_with(Path(repo_dir))
            self.assertEqual(repo_stat["version"], TEST_JSON_VERSION)
            self.assertEqual(repo_stat["n_vhdl_files"], 3)
            self.assertTrue(repo_stat["has_tests"])
            self.assertCountEqual(repo_stat["test_strategies"], ["vunit", "osvvm"])

            get_source_files_mock.return_value = [Path("egg.sv")]
            repo_stat = analyze_repo(Path(repo_dir))
            self.assertEqual(repo_stat["version"], TEST_JSON_VERSION)
            self.assertEqual(repo_stat["n_vhdl_files"], 0)
            self.assertFalse(repo_stat["has_tests"])
            self.assertFalse(repo_stat["test_strategies"])

    def test_remove_fp(self):
        repo_stat = dict(test_strategies=["vunit", "osvvm", "uvvm", "cocotb"])
        fp_repos = {"repo1/*": ["vunit", "osvvm"], "repo2/foo": ["cocotb", "uvm"]}

        repo_stat = remove_fp("repo0/foo", repo_stat, fp_repos)
        self.assertDictEqual(
            repo_stat, dict(test_strategies=["vunit", "osvvm", "uvvm", "cocotb"]),
        )

        repo_stat = remove_fp("repo1/foo", repo_stat, fp_repos)
        self.assertDictEqual(repo_stat, dict(test_strategies=["uvvm", "cocotb"]))

        repo_stat = dict(test_strategies=["vunit", "osvvm", "uvvm", "cocotb"])
        repo_stat = remove_fp("repo2/bar", repo_stat, fp_repos)
        self.assertDictEqual(
            repo_stat, dict(test_strategies=["vunit", "osvvm", "uvvm", "cocotb"]),
        )
        repo_stat = remove_fp("repo2/foo", repo_stat, fp_repos)
        self.assertDictEqual(
            repo_stat, dict(test_strategies=["vunit", "osvvm", "uvvm"])
        )

    def test_github_stat(self):
        github_stat = GithubStat()
        repo_stat = dict(n_vhdl_files=17, has_tests=False, test_strategies=[],)

        github_stat.update(None, "foo", "bar", repo_stat)
        github_stat.update(dict(id=123456789), "foo", "bar", repo_stat)
        test_strategy_numbers = dict(
            vunit=0, osvvm=0, uvvm=0, cocotb=0, uvm=0, unknown=0, other=0
        )
        self.assertDictEqual(
            github_stat.stat["test_strategy_numbers"], test_strategy_numbers
        )
        self.assertDictEqual(github_stat.stat["repo_stat"], dict())
        self.assertEqual(github_stat.stat["created_at"], dict())
        self.assertEqual(github_stat.stat["total_n_vhdl_files"], 0)
        self.assertEqual(github_stat.stat["total_n_repos"], 0)
        self.assertEqual(github_stat.stat["n_repos_with_standard_vhdl_strategy"], 0)
        self.assertEqual(
            github_stat.stat[
                "n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"
            ],
            0,
        )

        basic_data = dict(created_at="2020-05-31T17:26:08Z")
        created_at_second = int(datetime(2020, 5, 31, 17, 26, 8).timestamp())
        github_stat.update(basic_data, "foo", "bar", repo_stat)
        test_strategy_numbers["unknown"] = 1
        self.assertDictEqual(
            github_stat.stat["test_strategy_numbers"], test_strategy_numbers
        )
        self.assertDictEqual(github_stat.stat["repo_stat"]["foo/bar"], repo_stat)
        self.assertEqual(github_stat.stat["created_at"]["foo/bar"], created_at_second)
        self.assertEqual(github_stat.stat["total_n_vhdl_files"], 17)
        self.assertEqual(github_stat.stat["total_n_repos"], 1)
        self.assertEqual(github_stat.stat["n_repos_with_standard_vhdl_strategy"], 0)
        self.assertEqual(
            github_stat.stat[
                "n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"
            ],
            0,
        )

        repo_stat["has_tests"] = True
        github_stat.update(basic_data, "foo", "bar", repo_stat)
        test_strategy_numbers["other"] = 1
        self.assertDictEqual(
            github_stat.stat["test_strategy_numbers"], test_strategy_numbers
        )
        self.assertDictEqual(github_stat.stat["repo_stat"]["foo/bar"], repo_stat)
        self.assertEqual(github_stat.stat["created_at"]["foo/bar"], created_at_second)
        self.assertEqual(github_stat.stat["total_n_vhdl_files"], 34)
        self.assertEqual(github_stat.stat["total_n_repos"], 2)
        self.assertEqual(github_stat.stat["n_repos_with_standard_vhdl_strategy"], 0)
        self.assertEqual(
            github_stat.stat[
                "n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"
            ],
            0,
        )

        repo_stat["test_strategies"] = ["vunit", "osvvm"]
        github_stat.update(basic_data, "foo", "bar", repo_stat)
        test_strategy_numbers["vunit"] = 1
        test_strategy_numbers["osvvm"] = 1
        self.assertDictEqual(
            github_stat.stat["test_strategy_numbers"], test_strategy_numbers
        )
        self.assertDictEqual(github_stat.stat["repo_stat"]["foo/bar"], repo_stat)
        self.assertEqual(github_stat.stat["created_at"]["foo/bar"], created_at_second)
        self.assertEqual(github_stat.stat["total_n_vhdl_files"], 51)
        self.assertEqual(github_stat.stat["total_n_repos"], 3)
        self.assertEqual(github_stat.stat["n_repos_with_standard_vhdl_strategy"], 1)
        self.assertEqual(
            github_stat.stat[
                "n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"
            ],
            0,
        )

        repo_stat["has_tests"] = False
        github_stat.update(basic_data, "foo", "bar", repo_stat)
        test_strategy_numbers["vunit"] = 2
        test_strategy_numbers["osvvm"] = 2
        self.assertDictEqual(
            github_stat.stat["test_strategy_numbers"], test_strategy_numbers
        )
        self.assertDictEqual(github_stat.stat["repo_stat"]["foo/bar"], repo_stat)
        self.assertEqual(github_stat.stat["created_at"]["foo/bar"], created_at_second)
        self.assertEqual(github_stat.stat["total_n_vhdl_files"], 68)
        self.assertEqual(github_stat.stat["total_n_repos"], 4)
        self.assertEqual(github_stat.stat["n_repos_with_standard_vhdl_strategy"], 2)
        self.assertEqual(
            github_stat.stat[
                "n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"
            ],
            1,
        )

        with TemporaryDirectory() as output_dir:
            output_path = str(Path(output_dir) / "data.json")
            github_stat.dump(output_path)
            with open(output_path, "r") as json:
                result = load(json)
            self.assertDictEqual(result, github_stat.stat)

    def test_end_to_end(self):
        with TemporaryDirectory() as repos_root:
            output_path = Path(repos_root) / "stat.json"

            analyze(repos_root, output_path, None, False)

            with open(output_path) as json:
                stat = load(json)

            test_strategy_numbers = dict(
                vunit=0, osvvm=0, uvvm=0, cocotb=0, uvm=0, unknown=0, other=0
            )
            self.assertDictEqual(stat["test_strategy_numbers"], test_strategy_numbers)
            self.assertDictEqual(stat["repo_stat"], dict())
            self.assertDictEqual(stat["created_at"], dict())
            self.assertEqual(stat["total_n_vhdl_files"], 0)
            self.assertEqual(stat["total_n_repos"], 0)
            self.assertEqual(stat["n_repos_with_standard_vhdl_strategy"], 0)
            self.assertEqual(
                stat["n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"],
                0,
            )

            user1_path = Path(repos_root) / "user1"
            repo1_path = user1_path / "repo1"
            repo1_path.mkdir(parents=True)
            repo1_file_path = repo1_path / "repo1.vhdl"
            repo1_file_path.write_text("library vunit_lib;")

            repo1_zip_path = user1_path / "repo1.zip"
            with ZipFile(repo1_zip_path, "w", ZIP_DEFLATED) as zip_file:
                zip_file.write(repo1_file_path)
            repo1_basic_path = user1_path / f"repo1.basic.{BASIC_JSON_VERSION}.json"
            repo1_basic_path.write_text('{"created_at": "2016-05-31T17:26:08Z"}')

            repo1_file_path.unlink()
            repo1_path.rmdir()

            user2_path = Path(repos_root) / "user2"
            repo2_path = user2_path / "repo2"
            repo2_path.mkdir(parents=True)
            repo2_file_path = repo2_path / "tb_repo2.vhdl"
            repo2_file_path.write_text("library osvvm;")

            repo2_zip_path = user2_path / "repo2.zip"
            with ZipFile(repo2_zip_path, "w", ZIP_DEFLATED) as zip_file:
                zip_file.write(repo2_file_path)
            repo2_basic_path = user2_path / f"repo2.basic.{BASIC_JSON_VERSION}.json"
            repo2_basic_path.write_text('{"created_at": "2020-05-31T17:26:08Z"}')

            repo2_file_path.unlink()
            repo2_path.rmdir()

            analyze(Path(repos_root), output_path, None, False)
            with open(output_path) as json:
                stat = load(json)

            repo1_test_path = user1_path / f"repo1.test.{TEST_JSON_VERSION}.json"
            self.assertTrue(repo1_test_path.exists())
            repo2_test_path = user2_path / f"repo2.test.{TEST_JSON_VERSION}.json"
            self.assertTrue(repo2_test_path.exists())

            test_strategy_numbers["vunit"] = 1
            test_strategy_numbers["osvvm"] = 1
            self.assertDictEqual(stat["test_strategy_numbers"], test_strategy_numbers)

            self.assertEqual(len(stat["repo_stat"]), 2)
            self.assertEqual(len(stat["created_at"]), 2)
            repo1_key = "user1/repo1"
            self.assertEqual(stat["repo_stat"][repo1_key]["n_vhdl_files"], 1)
            self.assertFalse(stat["repo_stat"][repo1_key]["has_tests"])
            self.assertCountEqual(
                stat["repo_stat"][repo1_key]["test_strategies"], ["vunit"]
            )
            self.assertEqual(
                stat["created_at"][repo1_key],
                int(datetime(2016, 5, 31, 17, 26, 8).timestamp()),
            )

            repo2_key = "user2/repo2"
            self.assertEqual(stat["repo_stat"][repo2_key]["n_vhdl_files"], 1)
            self.assertTrue(stat["repo_stat"][repo2_key]["has_tests"])
            self.assertCountEqual(
                stat["repo_stat"][repo2_key]["test_strategies"], ["osvvm"]
            )
            self.assertEqual(
                stat["created_at"][repo2_key],
                int(datetime(2020, 5, 31, 17, 26, 8).timestamp()),
            )

            self.assertEqual(stat["total_n_vhdl_files"], 2)
            self.assertEqual(stat["total_n_repos"], 2)
            self.assertEqual(stat["n_repos_with_standard_vhdl_strategy"], 2)
            self.assertEqual(
                stat["n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"],
                1,
            )

            fp_repos = {"user1/*": ["vunit", "osvvm"]}
            fp_repos_path = Path(repos_root) / "fp_repos.json"
            with open(fp_repos_path, "w") as json:
                dump(fp_repos, json)

            analyze(Path(repos_root), output_path, fp_repos_path, False)
            with open(output_path) as json:
                stat = load(json)

            test_strategy_numbers["vunit"] = 0
            test_strategy_numbers["unknown"] = 1
            self.assertEqual(len(stat["repo_stat"]), 2)
            repo1_key = "user1/repo1"
            self.assertEqual(stat["repo_stat"][repo1_key]["n_vhdl_files"], 1)
            self.assertFalse(stat["repo_stat"][repo1_key]["has_tests"])
            self.assertCountEqual(stat["repo_stat"][repo1_key]["test_strategies"], [])
            self.assertEqual(
                stat["created_at"][repo1_key],
                int(datetime(2016, 5, 31, 17, 26, 8).timestamp()),
            )

            repo2_key = "user2/repo2"
            self.assertEqual(stat["repo_stat"][repo2_key]["n_vhdl_files"], 1)
            self.assertTrue(stat["repo_stat"][repo2_key]["has_tests"])
            self.assertCountEqual(
                stat["repo_stat"][repo2_key]["test_strategies"], ["osvvm"]
            )
            self.assertEqual(
                stat["created_at"][repo2_key],
                int(datetime(2020, 5, 31, 17, 26, 8).timestamp()),
            )

            self.assertEqual(stat["total_n_vhdl_files"], 2)
            self.assertEqual(stat["total_n_repos"], 2)
            self.assertEqual(stat["n_repos_with_standard_vhdl_strategy"], 1)
            self.assertEqual(
                stat["n_repos_with_standard_vhdl_strategy_but_no_obvious_test_files"],
                0,
            )

    @staticmethod
    @patch(
        "sys.argv",
        [
            "analyze_test_strategy.py",
            "path/to/repo_dir",
            "path/to/output_json",
            "--fp",
            "path/to/fp_json",
        ],
    )
    @patch("analyze_test_strategy.analyze")
    def test_cli(analyze_mock):
        main()
        analyze_mock.assert_called_once_with(
            Path("path") / "to" / "repo_dir",
            Path("path") / "to" / "output_json",
            Path("path") / "to" / "fp_json",
            False,
        )
