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
from visualize_users import (
    create_euler_diagrams,
    create_plots,
    create_bar_races,
    create_timeline,
    main,
)
from visualize_test_strategy import date_range


class TestVisualizeUsers(TestCase):
    """Tests visualize_users.py."""

    user_data = dict(
        user_1=dict(
            classification="professional", test_strategies=dict(vunit=1 * 86400)
        ),
        user_2=dict(
            classification="professional", test_strategies=dict(osvvm=2 * 86400)
        ),
        user_3=dict(
            classification="professional",
            test_strategies=dict(osvvm=3 * 86400, vunit=1 * 86400),
        ),
        user_4=dict(
            classification="professional", test_strategies=dict(vunit=5 * 86400)
        ),
        user_5=dict(classification="academic", test_strategies=dict(vunit=6 * 86400)),
        user_6=dict(classification="unknown", test_strategies=dict(vunit=7 * 86400)),
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

    @staticmethod
    @patch(
        "sys.argv",
        [
            "visualize_users.py",
            "path/to/users_stat.json",
            "path/to/image",
            "path/to/video",
        ],
    )
    @patch("visualize_users.visualize")
    def test_cli(visualize_mock):
        main()
        visualize_mock.assert_called_once_with(
            Path("path") / "to" / "users_stat.json",
            Path("path") / "to" / "image",
            Path("path") / "to" / "video",
        )
