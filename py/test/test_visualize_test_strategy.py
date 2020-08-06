# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Test visualize_test_strategy."""

from unittest import TestCase
from unittest.mock import patch, call
from tempfile import TemporaryDirectory
from pathlib import Path
from json import dump
from datetime import datetime, date
from visualize_test_strategy import (
    get_percentage_repos_with_tests,
    make_graph_over_time,
    visualize,
    main,
    get_repo_framework_distribution,
    plot_euler_diagram,
)


class TestVisualizeTestStrategy(TestCase):
    """Tests visualize_test_strategy.py."""

    def test_get_repo_framework_distribution(self):
        repos_stat = dict(
            repo_stat={
                "1": dict(test_strategies=["uvm"]),
                "2": dict(test_strategies=["uvvm"]),
                "3": dict(test_strategies=["osvvm"]),
                "4": dict(test_strategies=["osvvm", "vunit"]),
                "5": dict(test_strategies=["osvvm", "vunit", "uvvm"]),
                "6": dict(test_strategies=["vunit"]),
                "7": dict(test_strategies=["vunit"]),
                "8": dict(test_strategies=["vunit", "osvvm"]),
                "9": dict(test_strategies=["vunit"]),
                "10": dict(test_strategies=["vunit"]),
                "11": dict(test_strategies=["vunit"]),
            },
        )

        repo_classification = {str(name): "professional" for name in range(1, 12)}
        repo_classification["9"] = "academic"
        repo_classification["10"] = "academic"
        repo_classification["11"] = "unknown"

        distribution = get_repo_framework_distribution(repos_stat, repo_classification)
        self.assertDictEqual(
            distribution["total"],
            {
                "UVM": 1,
                "UVVM": 1,
                "OSVVM": 1,
                "OSVVM&VUnit": 2,
                "OSVVM&UVVM&VUnit": 1,
                "VUnit": 5,
            },
        )
        self.assertDictEqual(
            distribution["professional"],
            {
                "UVM": 1,
                "UVVM": 1,
                "OSVVM": 1,
                "OSVVM&VUnit": 2,
                "OSVVM&UVVM&VUnit": 1,
                "VUnit": 2,
            },
        )
        self.assertDictEqual(
            distribution["academic"], {"VUnit": 2,},
        )
        self.assertDictEqual(
            distribution["unknown"], {"VUnit": 1,},
        )

    def test_get_percentage_repos_with_tests(self):
        repos_stat = dict(
            created_at={
                "foo/bar": int(datetime(2019, 1, 1, 17, 26, 8).timestamp()),
                "spam/eggs": int(datetime(2019, 12, 30, 23, 59, 59).timestamp()),
            },
            repo_stat={
                "foo/bar": dict(has_tests=False, test_strategies=[]),
                "spam/eggs": dict(has_tests=True, test_strategies=["vunit"]),
            },
        )
        (
            timeline,
            percentage_repos_with_tests,
            percentage_repos_with_std_test_strategy,
        ) = get_percentage_repos_with_tests(repos_stat, 1)
        self.assertListEqual(timeline, [])
        self.assertListEqual(percentage_repos_with_tests, [])
        self.assertListEqual(percentage_repos_with_std_test_strategy, [])

        repos_stat["created_at"]["spam/eggs"] += 1

        repos_stat["created_at"]["ying/yang"] = int(
            datetime(2020, 1, 1, 10, 0, 0).timestamp()
        )
        repos_stat["repo_stat"]["ying/yang"] = dict(
            has_tests=False, test_strategies=["vunit"]
        )

        repos_stat["created_at"]["ping/pong"] = repos_stat["created_at"]["ying/yang"]
        repos_stat["repo_stat"]["ping/pong"] = dict(has_tests=True, test_strategies=[])
        repos_stat["created_at"]["bill/bull"] = repos_stat["created_at"]["ying/yang"]
        repos_stat["repo_stat"]["bill/bull"] = dict(has_tests=True, test_strategies=[])

        (
            timeline,
            percentage_repos_with_tests,
            percentage_repos_with_std_test_strategy,
        ) = get_percentage_repos_with_tests(repos_stat, 1)
        self.assertListEqual(timeline, [date(2019, 12, 31), date(2020, 1, 1)])
        self.assertListEqual(percentage_repos_with_tests, [50, 100])
        self.assertListEqual(percentage_repos_with_std_test_strategy, [100, 50])

        (
            timeline,
            percentage_repos_with_tests,
            percentage_repos_with_std_test_strategy,
        ) = get_percentage_repos_with_tests(repos_stat, 2)
        self.assertListEqual(timeline, [date(2020, 1, 1)])
        self.assertListEqual(percentage_repos_with_tests, [100])
        self.assertListEqual(percentage_repos_with_std_test_strategy, [50])

    def test_make_graph_over_time(self):
        with TemporaryDirectory() as output_dir:

            output_path = Path(output_dir) / "plot.png"

            make_graph_over_time(
                [date(2019, 12, 31), date(2020, 1, 1)],
                dict(a=[50, 100], b=[1, 10]),
                "ylabel",
                "title",
                output_path,
            )

            self.assertTrue(output_path.exists())

    def test_plot_euler_diagram(self):
        with TemporaryDirectory() as output_dir:

            output_path = Path(output_dir) / "diagram.svg"

            distribution = {
                "VUnit": 2,
                "OSVVM": 1,
                "OSVVM&VUnit": 1,
            }

            plot_euler_diagram(distribution, output_path)

            self.assertTrue(output_path.exists())

    @staticmethod
    @patch("visualize_test_strategy.plot_euler_diagram")
    @patch("visualize_test_strategy.get_repo_framework_distribution")
    @patch("visualize_test_strategy.get_percentage_repos_with_tests")
    @patch("visualize_test_strategy.make_graph_over_time")
    def test_visualize(
        make_graph_over_time_mock,
        get_percentage_repos_with_tests_mock,
        get_repo_framework_distribution_mock,
        plot_euler_diagram_mock,
    ):
        with TemporaryDirectory() as work_dir:
            repos_stat_path = Path(work_dir) / "repos_stat.json"
            repos_stat = "repos_stat"
            with open(repos_stat_path, "w") as json:
                dump(repos_stat, json)

            get_percentage_repos_with_tests_mock.return_value = (
                [date(2019, 12, 31), date(2020, 1, 1)],
                [50, 100],
                [100, 50],
            )

            repo_classification_path = Path(work_dir) / "repo_classification.json"
            repo_classification = "repo_classification"
            with open(repo_classification_path, "w") as json:
                dump(repo_classification, json)

            get_repo_framework_distribution_mock.return_value = dict(
                total={
                    "UVM": 1,
                    "UVVM": 1,
                    "OSVVM": 1,
                    "OSVVM&VUnit": 2,
                    "OSVVM&UVVM&VUnit": 1,
                    "VUnit": 5,
                },
                professional={
                    "UVM": 1,
                    "UVVM": 1,
                    "OSVVM": 1,
                    "OSVVM&VUnit": 2,
                    "OSVVM&UVVM&VUnit": 1,
                    "VUnit": 2,
                },
                academic={"VUnit": 2},
                unknown={"VUnit": 1},
            )

            visualize(repos_stat_path, repo_classification_path, Path(work_dir), 1)

            get_percentage_repos_with_tests_mock.assert_called_once_with(
                "repos_stat", 1
            )
            calls = [
                call(
                    [date(2019, 12, 31), date(2020, 1, 1)],
                    dict(data=[50, 100]),
                    "Percentage",
                    "Repositories Providing Tests (1 Year Average)",
                    (Path(work_dir) / "repositories_providing_tests.png"),
                    False,
                ),
                call(
                    [date(2019, 12, 31), date(2020, 1, 1)],
                    dict(data=[100, 50]),
                    "Percentage",
                    "Repositories with Tests Using a Standard Framework (1 Year Average)",
                    (Path(work_dir) / "repositories_using_std_framework.png"),
                    False,
                ),
            ]
            make_graph_over_time_mock.assert_has_calls(calls)

            get_repo_framework_distribution_mock.assert_called_once_with(
                "repos_stat", "repo_classification"
            )
            calls = [
                call(
                    {
                        "UVM": 1,
                        "UVVM": 1,
                        "OSVVM": 1,
                        "OSVVM&VUnit": 2,
                        "OSVVM&UVVM&VUnit": 1,
                        "VUnit": 5,
                    },
                    (Path(work_dir) / "total_repo_framework_distribution.svg"),
                ),
                call(
                    {
                        "UVM": 1,
                        "UVVM": 1,
                        "OSVVM": 1,
                        "OSVVM&VUnit": 2,
                        "OSVVM&UVVM&VUnit": 1,
                        "VUnit": 2,
                    },
                    (Path(work_dir) / "professional_repo_framework_distribution.svg"),
                ),
                call(
                    {"VUnit": 2,},
                    (Path(work_dir) / "academic_repo_framework_distribution.svg"),
                ),
                call(
                    {"VUnit": 1,},
                    (Path(work_dir) / "unknown_repo_framework_distribution.svg"),
                ),
            ]
            plot_euler_diagram_mock.assert_has_calls(calls)

    @staticmethod
    @patch(
        "sys.argv",
        [
            "visualize_test_strategy.py",
            "path/to/repos_stat.json",
            "path/to/repo_classification.json",
            "path/to/output_dir",
        ],
    )
    @patch("visualize_test_strategy.visualize")
    def test_cli(visualize_mock):
        main()
        visualize_mock.assert_called_once_with(
            Path("path") / "to" / "repos_stat.json",
            Path("path") / "to" / "repo_classification.json",
            Path("path") / "to" / "output_dir",
        )
