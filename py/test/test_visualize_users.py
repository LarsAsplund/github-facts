# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Test visualize_users."""

from unittest import TestCase
from unittest.mock import patch, call
from tempfile import TemporaryDirectory
from pathlib import Path
from shutil import copy
from datetime import date
from json import dump
from visualize_users import (
    create_euler_diagrams,
    create_plots,
    create_bar_races,
    create_timeline,
    main,
    create_timezone_chart,
    get_error_color,
    create_timezone_charts,
    visualize,
)
from visualize_test_strategy import date_range


class TestVisualizeUsers(TestCase):
    """Tests visualize_users.py."""

    user_data = dict(
        user_1=dict(
            classification="professional",
            test_strategies=dict(vunit=dict(commit_time=1 * 86400, repo="a/b")),
        ),
        user_2=dict(
            classification="professional",
            test_strategies=dict(osvvm=dict(commit_time=2 * 86400, repo="c/d")),
        ),
        user_3=dict(
            classification="professional",
            test_strategies=dict(
                osvvm=dict(commit_time=3 * 86400, repo="e/f"),
                vunit=dict(commit_time=1 * 86400, repo="g/h"),
            ),
        ),
        user_4=dict(
            classification="professional",
            test_strategies=dict(vunit=dict(commit_time=5 * 86400, repo="i/j")),
        ),
        user_5=dict(
            classification="academic",
            test_strategies=dict(vunit=dict(commit_time=6 * 86400, repo="k/l")),
        ),
        user_6=dict(
            classification="unknown",
            test_strategies=dict(vunit=dict(commit_time=7 * 86400, repo="m/n")),
        ),
    )

    accumulated_histograms = dict(
        professional=dict(
            accumulated_histogram=dict(
                vunit=[0, 2, 2, 2, 2, 3],
                osvvm=[0, 0, 1, 2, 2, 2],
                uvvm=[0, 0, 0, 0, 0, 0],
                uvm=[0, 0, 0, 0, 0, 0],
                cocotb=[0, 0, 0, 0, 0, 0],
                std=[0, 2, 3, 4, 4, 5],
            ),
            first_date=date(1970, 1, 1),
            last_date=date(1970, 1, 6),
        ),
        academic=dict(
            accumulated_histogram=dict(
                vunit=[0, 1],
                osvvm=[0, 0],
                uvvm=[0, 0],
                uvm=[0, 0],
                cocotb=[0, 0],
                std=[0, 1],
            ),
            first_date=date(1970, 1, 6),
            last_date=date(1970, 1, 7),
        ),
        unknown=dict(
            accumulated_histogram=dict(
                vunit=[0, 1],
                osvvm=[0, 0],
                uvvm=[0, 0],
                uvm=[0, 0],
                cocotb=[0, 0],
                std=[0, 1],
            ),
            first_date=date(1970, 1, 7),
            last_date=date(1970, 1, 8),
        ),
        total=dict(
            accumulated_histogram=dict(
                vunit=[0, 2, 2, 2, 2, 3, 4, 5],
                osvvm=[0, 0, 1, 2, 2, 2, 2, 2],
                uvvm=[0, 0, 0, 0, 0, 0, 0, 0],
                uvm=[0, 0, 0, 0, 0, 0, 0, 0],
                cocotb=[0, 0, 0, 0, 0, 0, 0, 0],
                std=[0, 2, 3, 4, 4, 5, 6, 7],
            ),
            first_date=date(1970, 1, 1),
            last_date=date(1970, 1, 8),
        ),
    )

    @patch("visualize_users.plot_euler_diagram")
    def test_create_euler_diagrams(self, plot_euler_diagram_mock):
        create_euler_diagrams(self.user_data, Path("output_path"))

        calls = [
            call(
                {"VUnit": 4, "OSVVM": 1, "OSVVM&VUnit": 1},
                Path("output_path") / "total_user.svg",
            ),
            call(
                {"VUnit": 2, "OSVVM": 1, "OSVVM&VUnit": 1},
                Path("output_path") / "professional_user.svg",
            ),
            call({"VUnit": 1}, Path("output_path") / "academic_user.svg"),
            call({"VUnit": 1}, Path("output_path") / "unknown_user.svg"),
        ]

        plot_euler_diagram_mock.assert_has_calls(calls)

    @patch("visualize_users.make_graph_over_time")
    def test_create_plots(self, make_graph_over_time_mock):
        actual_accumulated_histograms = create_plots(
            self.user_data, Path("output_path")
        )

        for classification in actual_accumulated_histograms:
            self.assertDictEqual(
                actual_accumulated_histograms[classification]["accumulated_histogram"],
                self.accumulated_histograms[classification]["accumulated_histogram"],
            )
            self.assertEqual(
                actual_accumulated_histograms[classification]["first_date"],
                self.accumulated_histograms[classification]["first_date"],
            )
            self.assertEqual(
                actual_accumulated_histograms[classification]["last_date"],
                self.accumulated_histograms[classification]["last_date"],
            )

        calls = make_graph_over_time_mock.call_args_list

        for call_obj in calls:
            self.assertEqual(call_obj.args[2], "Number of users")

        self.assertEqual(len(calls), 4)

        for idx, classification in enumerate(
            ["professional", "academic", "unknown", "total"]
        ):
            self.assertListEqual(
                list(calls[idx].args[0]),
                list(
                    date_range(
                        self.accumulated_histograms[classification]["first_date"],
                        self.accumulated_histograms[classification]["last_date"],
                    )
                ),
            )
            self.assertDictEqual(
                calls[idx].args[1],
                dict(
                    vunit=self.accumulated_histograms[classification][
                        "accumulated_histogram"
                    ]["vunit"],
                    osvvm=self.accumulated_histograms[classification][
                        "accumulated_histogram"
                    ]["osvvm"],
                    uvvm=self.accumulated_histograms[classification][
                        "accumulated_histogram"
                    ]["uvvm"],
                    uvm=self.accumulated_histograms[classification][
                        "accumulated_histogram"
                    ]["uvm"],
                    cocotb=self.accumulated_histograms[classification][
                        "accumulated_histogram"
                    ]["cocotb"],
                ),
            )

            if classification == "total":
                self.assertEqual(calls[idx].args[3], "Number of users over time")
            else:
                self.assertEqual(
                    calls[idx].args[3], f"Number of {classification} users over time"
                )

            self.assertEqual(
                calls[idx].args[4],
                Path("output_path") / f"{classification}_users_over_time.svg",
            )

    def test_create_timeline(self):
        timeline_start_date = date(1970, 1, 1)
        framework_introduction = dict(
            fw_1=dict(date=date(1970, 1, 7)),
            fw_2=dict(date=date(1970, 1, 18)),
            fw_3=dict(date=date(1970, 1, 19)),
        )
        last_date = date(1970, 1, 24)

        frame_dates = create_timeline(
            timeline_start_date, last_date, framework_introduction
        )

        expected_timeline = (
            [date(1970, 1, 1), date(1970, 1, 6)]
            + [date(1970, 1, 7)] * 90
            + [date(1970, 1, 11), date(1970, 1, 16)]
            + [date(1970, 1, 18)] * 90
            + [date(1970, 1, 19)] * 90
            + [date(1970, 1, 21)]
            + [date(1970, 1, 24)] * 90
        )

        self.assertListEqual(list(frame_dates), expected_timeline)

    @patch("visualize_users.create_timeline")
    def test_create_bar_races(self, create_timeline_mock):
        with TemporaryDirectory() as image_dir, TemporaryDirectory() as video_dir:
            real_image_path = Path(__file__).parent.parent.parent / "img"
            for framework in ["vunit", "osvvm", "uvvm", "uvm", "cocotb"]:
                logo_file_name = f"{framework}_logo.png"
                copy(
                    str(real_image_path / logo_file_name),
                    str(Path(image_dir) / logo_file_name),
                )
            copy(
                str(real_image_path / "play.png"), str(Path(image_dir) / "play.png"),
            )
            create_timeline_mock.return_value = [
                date(1970, 1, 1),
                date(1970, 1, 1),
                date(1970, 1, 2),
            ]
            create_bar_races(
                self.accumulated_histograms,
                date(1970, 1, 1),
                Path(image_dir),
                Path(video_dir),
            )

            for classification in ["professional", "academic", "unknown", "total"]:
                self.assertTrue(
                    (Path(video_dir) / f"{classification}_user_bar_race.mp4").exists()
                )
                self.assertTrue(
                    (Path(image_dir) / f"{classification}_user_bar_race.png").exists()
                )

    @patch("visualize_users.proportion_confint")
    def test_create_timezone_chart(self, proportion_confint_mock):
        timezone_distributions = dict(
            unknown={timezone: abs(timezone) for timezone in range(-12, 15)}
        )
        timezone_distributions_as_percentage = dict(
            unknown=dict(
                timezones={
                    timezone: 100
                    * abs(timezone)
                    / sum(timezone_distributions["unknown"].values())
                    for timezone in range(-12, 15)
                }
            )
        )

        regions_percentage = dict()
        region_users = dict()
        total = sum(timezone_distributions["unknown"].values())
        region_users["Europe"] = sum(
            [
                value
                for key, value in timezone_distributions["unknown"].items()
                if key in range(-1, 4)
            ]
        )
        regions_percentage["Europe"] = 100 * region_users["Europe"] / total

        region_users["America"] = sum(
            [
                value
                for key, value in timezone_distributions["unknown"].items()
                if key in range(-10, -1)
            ]
        )
        regions_percentage["America"] = 100 * region_users["America"] / total

        region_users["Asia"] = total - region_users["Europe"] - region_users["America"]
        regions_percentage["Asia"] = (
            100 - regions_percentage["Europe"] - regions_percentage["America"]
        )

        side_effect = [
            (
                (region_users["America"] - 0) / total,
                (region_users["America"] + 1) / total,
            ),
            (
                (region_users["Europe"] - 2) / total,
                (region_users["Europe"] + 3) / total,
            ),
            ((region_users["Asia"] - 4) / total, (region_users["Asia"] + 5) / total),
        ]
        proportion_confint_mock.side_effect = side_effect

        timezone_data = create_timezone_chart(timezone_distributions, "label")

        self.assertDictEqual(
            timezone_data["unknown"]["timezones"],
            timezone_distributions_as_percentage["unknown"]["timezones"],
        )

        for region in regions_percentage:
            self.assertAlmostEqual(
                timezone_data["unknown"]["regions_percentage"][region],
                regions_percentage[region],
            )

        for iteration, region in enumerate(["America", "Europe", "Asia"]):
            self.assertAlmostEqual(
                timezone_data["unknown"]["region_lower_bound"][region],
                100 * 2 * iteration / total,
            )
            self.assertAlmostEqual(
                timezone_data["unknown"]["region_upper_bound"][region],
                100 * (2 * iteration + 1) / total,
            )

    def test_get_error_color(self):
        region = "Europe"
        regions_percentage = dict(Europe=17)
        region_upper_bound = dict(Europe=4)  # 21
        region_lower_bound = dict(Europe=7)  # 10
        reference_timezone_data = dict(
            regions_percentage=dict(Europe=None),
            region_upper_bound=dict(Europe=2),
            region_lower_bound=dict(Europe=3),
        )
        for ref_region_percentage, expected_color in zip(
            [7, 8, 9, 13, 17, 19, 20, 24, 25],
            ["red", "blue", "blue", "green", "green", "green", "blue", "blue", "red"],
        ):
            reference_timezone_data["regions_percentage"][
                "Europe"
            ] = ref_region_percentage
            self.assertEqual(
                get_error_color(
                    region,
                    regions_percentage,
                    region_lower_bound,
                    region_upper_bound,
                    reference_timezone_data,
                ),
                expected_color,
            )

    @patch("visualize_users.create_timezone_chart")
    def test_create_timezone_charts(self, create_timezone_chart_mock):
        with TemporaryDirectory() as image_dir:
            image_path = Path(image_dir)
            user_data = dict(
                user_1=dict(
                    timezones={-8: 66, -7: 34},
                    classification="professional",
                    test_strategies=dict(vunit=dict(commit_time=1 * 86400, repo="a/b")),
                ),
                user_2=dict(
                    timezones={1: 50, 2: 50},
                    classification="professional",
                    test_strategies=dict(osvvm=dict(commit_time=2 * 86400, repo="c/d")),
                ),
                user_3=dict(
                    classification="professional",
                    timezones={0: 10, 1: 70, 2: 20},
                    test_strategies=dict(
                        osvvm=dict(commit_time=3 * 86400, repo="e/f"),
                        vunit=dict(commit_time=1 * 86400, repo="g/h"),
                    ),
                ),
            )

            create_timezone_chart_mock.return_value = dict(user=dict())

            timezone_data = create_timezone_charts(
                user_data, image_path, reference_timezone_data=None
            )
            self.assertDictEqual(timezone_data, dict(user=dict()))

            expected_timezone_arg = dict(vunit={-8: 1, 1: 1}, osvvm={1: 2},)
            create_timezone_chart_mock.assert_called_once_with(
                expected_timezone_arg, "Framework", None
            )
            self.assertTrue((image_path / "framework_timezone_chart.svg").exists())

    @staticmethod
    @patch("visualize_users.create_timezone_charts")
    @patch("visualize_users.print_vunit_hotspot")
    @patch("visualize_users.create_euler_diagrams")
    @patch("visualize_users.create_plots")
    @patch("visualize_users.create_bar_races")
    def test_visualize(
        create_bar_races_mock,
        create_plots_mock,
        create_euler_diagrams_mock,
        print_vunit_hotspot_mock,
        create_timezone_charts_mock,
    ):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            std_users_stat_path = temp_path / "std_users_stat_path.json"
            sample_users_stat_path = temp_path / "sample_users_stat_path.json"
            image_path = temp_path / "image_path"
            video_path = temp_path / "video_path"

            with open(std_users_stat_path, "w") as json:
                dump("std_users_stat", json)

            with open(sample_users_stat_path, "w") as json:
                dump("sample_users_stat", json)

            create_timezone_charts_mock.side_effect = [
                dict(unknown="reference_timezone_data"),
                "timezone_data",
            ]
            create_plots_mock.return_value = "accumulated_histograms"

            visualize(
                std_users_stat_path, sample_users_stat_path, image_path, video_path
            )

            calls = [
                call("sample_users_stat", image_path),
                call("std_users_stat", image_path, "reference_timezone_data"),
            ]

            create_timezone_charts_mock.assert_has_calls(calls)
            print_vunit_hotspot_mock.assert_called_once_with("std_users_stat")
            create_euler_diagrams_mock.assert_called_once_with(
                "std_users_stat", image_path
            )
            create_plots_mock.assert_called_once_with("std_users_stat", image_path)
            create_bar_races_mock.assert_called_once_with(
                "accumulated_histograms", date(2011, 1, 1), image_path, video_path
            )

    @staticmethod
    @patch(
        "sys.argv",
        [
            "visualize_users.py",
            "path/to/users_stat.json",
            "path/to/sample_users_stat.json",
            "path/to/image",
            "path/to/video",
        ],
    )
    @patch("visualize_users.visualize")
    def test_cli(visualize_mock):
        main()
        visualize_mock.assert_called_once_with(
            Path("path") / "to" / "users_stat.json",
            Path("path") / "to" / "sample_users_stat.json",
            Path("path") / "to" / "image",
            Path("path") / "to" / "video",
        )
