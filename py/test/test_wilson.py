# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Test visualize_users."""

# pylint: disable=too-many-lines
from unittest import TestCase
from tempfile import TemporaryDirectory
from pathlib import Path
from datetime import datetime
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from wilson import (
    get_wilson_data,
    calc_bounds,
    get_y_pos,
    extract_vhdl_users,
    get_similarity,
    get_region,
    to_graph,
    sum_graphs,
    to_distrinution,
    user_data_to_graph,
    user_data_to_distribution,
    get_total_ci,
    scale_distribution,
    add_month,
    get_active_users_last_2_years,
    generate_wilson_study_2018_2020,
    generate_wilson_study_vhdl_2018_2020,
    generate_github_wilson_comparison_2020,
    generate_github_wilson_combined_comparison_2020,
    get_temporal_distributions,
    get_temporal_bias_data,
    generate_temporal_bias_analysis_mid_2018_2020,
    generate_github_wilson_europe,
    get_region_bias_similarity,
    generate_region_bias_similarity,
    generate_github_academic_and_professional_comparison_2020,
    generate_github_academic_professional_comparison,
    generate_github_wilson_full_combined_comparison,
)

# from visualize_test_strategy import date_range


class TestWilson(TestCase):  # pylint: disable=too-many-public-methods
    """Test wilson.py."""

    # pylint: disable=missing-function-docstring
    def test_get_wilson_data(self):
        meta_data, graph = get_wilson_data()
        self.assertEqual(graph["all"][2016]["uvm"], 0.47 * 0.48 + 0.53 * 0.71)
        self.assertEqual(
            meta_data[2016]["total_fpga_frameworks"],
            0.48 + 0.23 + 0.09 + 0.18 + 0.04 + 0.03 + 0.02 + 0.17,
        )
        self.assertEqual(graph["fpga"][2016]["total_users"], round(0.47 * 1703))
        self.assertEqual(graph["asic"][2016]["total_users"], 1703 - round(0.47 * 1703))
        self.assertEqual(graph["all"][2016]["total_users"], 1703)
        self.assertEqual(
            graph["fpga"][2016]["total_frameworks"],
            0.48 + 0.23 + 0.09 + 0.18 + 0.04 + 0.03 + 0.02 + 0.17,
        )
        self.assertEqual(
            graph["asic"][2016]["total_frameworks"],
            0.71 + 0.18 + 0.05 + 0.11 + 0.04 + 0.08 + 0.04 + 0.1,
        )
        self.assertAlmostEqual(
            graph["all"][2016]["total_frameworks"],
            0.47 * (0.48 + 0.23 + 0.09 + 0.18 + 0.04 + 0.03 + 0.02 + 0.17)
            + 0.53 * (0.71 + 0.18 + 0.05 + 0.11 + 0.04 + 0.08 + 0.04 + 0.1),
        )

    def test_calc_bounds(self):  # pylint: disable=too-many-locals
        for scale in [1, 2]:
            graph = dict(all=dict(total_users=100, a=0.5, b=0.3, c=0.2))

            expected_n_users_1 = [50 / scale, 30 / scale, 20 / scale]
            (lower_bound_a, upper_bound_a) = proportion_confint(
                50, 100, 0.05, "binom_test"
            )
            (lower_bound_b, upper_bound_b) = proportion_confint(
                30, 100, 0.05, "binom_test"
            )
            (lower_bound_c, upper_bound_c) = proportion_confint(
                20, 100, 0.05, "binom_test"
            )
            expected_lower_bounds_1 = [
                (users - 100 * lower_bound) / scale
                for users, lower_bound in [
                    (50, lower_bound_a),
                    (30, lower_bound_b),
                    (20, lower_bound_c),
                ]
            ]
            expected_upper_bounds_1 = [
                (100 * upper_bound - users) / scale
                for users, upper_bound in [
                    (50, upper_bound_a),
                    (30, upper_bound_b),
                    (20, upper_bound_c),
                ]
            ]

            n_users, lower_bounds, upper_bounds = calc_bounds(
                graph, "all", None, ["a", "b", "c"], scale
            )
            self.assertListEqual(n_users, expected_n_users_1)
            self.assertListEqual(lower_bounds, expected_lower_bounds_1)
            self.assertListEqual(upper_bounds, expected_upper_bounds_1)

            expected_n_users_2 = [70 / scale, 30 / scale]
            (lower_bound_ac, upper_bound_ac) = proportion_confint(
                70, 100, 0.05, "binom_test"
            )
            (lower_bound_b, upper_bound_b) = proportion_confint(
                30, 100, 0.05, "binom_test"
            )
            expected_lower_bounds_2 = [
                (users - 100 * lower_bound) / scale
                for users, lower_bound in [
                    (70, lower_bound_ac),
                    (30, lower_bound_b),
                ]
            ]
            expected_upper_bounds_2 = [
                (100 * upper_bound - users) / scale
                for users, upper_bound in [
                    (70, upper_bound_ac),
                    (30, upper_bound_b),
                ]
            ]

            n_users, lower_bounds, upper_bounds = calc_bounds(
                graph, "all", None, ["a&c", "b"], scale
            )
            self.assertListEqual(n_users, expected_n_users_2)
            self.assertListEqual(lower_bounds, expected_lower_bounds_2)
            self.assertListEqual(upper_bounds, expected_upper_bounds_2)

            graph = dict(
                all={
                    2017: dict(total_users=100, a=0.5, b=0.3, c=0.2),
                    2016: dict(total_users=10, a=0.1, b=0.9),
                }
            )

            n_users, lower_bounds, upper_bounds = calc_bounds(
                graph, "all", 2017, ["a", "b", "c"], scale
            )
            self.assertListEqual(n_users, expected_n_users_1)
            self.assertListEqual(lower_bounds, expected_lower_bounds_1)
            self.assertListEqual(upper_bounds, expected_upper_bounds_1)

    def test_get_y_pos(self):
        graph = dict(
            all={
                2017: dict(total_users=100, a=0.5, b=0.3, c=0.2),
                2016: dict(total_users=10, a=0.1, b=0.9),
            }
        )
        framework_groups = ["a&c", "b"]
        year = 2017
        target = "all"
        normalize = False
        y_pos, scale = get_y_pos(framework_groups, graph, year, target, normalize)
        self.assertListEqual(y_pos, [70, 30])
        self.assertEqual(scale, 1)

        framework_groups = ["a", "b"]
        normalize = True
        y_pos, scale = get_y_pos(framework_groups, graph, year, target, normalize)
        self.assertListEqual(y_pos, [62.5, 37.5])
        self.assertEqual(scale, 0.8)

    def test_extract_vhdl_users(self):
        all_years = [2019, 2020]
        all_frameworks = ["osvvm", "uvvm", "a", "b"]
        graph = dict(
            fpga={
                2019: dict(
                    total_users=100,
                    total_frameworks=1.0,
                    osvvm=0.1,  # 10
                    uvvm=0.2,  # 20
                    a=0.6,  # 60
                    b=0.1,  # 10
                ),
                2020: dict(
                    total_users=200,
                    total_frameworks=1.2,
                    osvvm=0.15,
                    uvvm=0.25,
                    a=0.6,
                    b=0.2,
                ),
            },
            asic={
                2019: dict(
                    total_users=300,
                    total_frameworks=2,
                    osvvm=0.2,  # 60
                    uvvm=0.3,  # 90
                    a=0.75,  # 225
                    b=0.75,  # 225
                ),
                2020: dict(
                    total_users=400,
                    total_frameworks=1.5,
                    osvvm=0.3,
                    uvvm=0.4,
                    a=0.2,
                    b=0.6,
                ),
            },
            all={
                2019: dict(
                    total_users=400,
                    total_frameworks=(100 + 300 * 2) / 400,
                    osvvm=(0.1 * 100 + 0.2 * 300) / 400,
                    uvvm=(0.2 * 100 + 0.3 * 300) / 400,
                    a=(0.6 * 100 + 0.75 * 300) / 400,
                    b=(0.1 * 100 + 0.75 * 300) / 400,
                ),
                2020: dict(
                    total_users=600,
                    total_frameworks=(200 * 1.2 + 400 * 1.5) / 600,
                    osvvm=(0.15 * 200 + 0.3 * 400) / 600,
                    uvvm=(0.25 * 200 + 0.4 * 400) / 600,
                    a=(0.6 * 200 + 0.2 * 400) / 600,
                    b=(0.2 * 200 + 0.6 * 400) / 600,
                ),
            },
        )
        meta_data = {
            2019: dict(
                fpga_vhdl_portion=0.4, asic_vhdl_portion=0.75, fpga_portion=0.25
            ),
            2020: dict(
                fpga_vhdl_portion=0.6, asic_vhdl_portion=0.5, fpga_portion=1 / 3
            ),
        }

        vhdl_graph = extract_vhdl_users(graph, meta_data, all_years, all_frameworks)
        self.assertDictEqual(
            vhdl_graph["fpga"],
            {
                2019: dict(
                    total_users=40,
                    osvvm=0.25,
                    uvvm=0.5,
                    a=9 / 40,
                    b=1 / 40,
                ),
                2020: dict(
                    total_users=120,
                    osvvm=0.25,
                    uvvm=50 / 120,
                    a=0.4,
                    b=0.4 / 3,
                ),
            },
        )

        self.assertDictEqual(
            vhdl_graph["asic"],
            {
                2019: dict(
                    total_users=225,
                    osvvm=60 / 225,
                    uvvm=90 / 225,
                    a=150 / 225,
                    b=150 / 225,
                ),
                2020: dict(
                    total_users=200,
                    osvvm=0.6,
                    uvvm=0.8,
                    a=0.025,
                    b=0.075,
                ),
            },
        )

        self.assertEqual(vhdl_graph["all"][2019]["total_users"], 265)
        self.assertEqual(vhdl_graph["all"][2019]["osvvm"], 70 / 265)
        self.assertEqual(vhdl_graph["all"][2019]["uvvm"], 110 / 265)
        self.assertAlmostEqual(
            vhdl_graph["all"][2019]["a"], (310 * 285) / (265 * (285 + 235)), places=3
        )
        self.assertAlmostEqual(
            vhdl_graph["all"][2019]["b"], (310 * 235) / (265 * (285 + 235)), places=3
        )

        self.assertEqual(vhdl_graph["all"][2020]["total_users"], 320)
        self.assertEqual(vhdl_graph["all"][2020]["osvvm"], 150 / 320)
        self.assertEqual(vhdl_graph["all"][2020]["uvvm"], 210 / 320)
        self.assertAlmostEqual(
            vhdl_graph["all"][2020]["a"],
            ((144 + 300 - 360) * 200) / (320 * 480),
            places=3,
        )
        self.assertAlmostEqual(
            vhdl_graph["all"][2020]["b"],
            ((144 + 300 - 360) * 280) / (320 * 480),
            places=3,
        )

    def test_get_similarity(self):
        user_distributions = [[1, 1, 1], [1, 1, 1]]
        assumed_distribution = [1 / 3, 1 / 3, 1 / 3]
        similarities, most_probable_distribution = get_similarity(
            user_distributions, assumed_distribution
        )
        self.assertSequenceEqual(similarities, [1])
        self.assertSequenceEqual(most_probable_distribution, assumed_distribution)

        user_distributions = [[0, 0, 3], [0, 0, 3]]
        assumed_distribution = [0.5, 0.3, 0.2]
        similarities, most_probable_distribution = get_similarity(
            user_distributions, assumed_distribution
        )
        self.assertAlmostEqual(similarities[0], 0.2 ** 6)
        self.assertSequenceEqual(most_probable_distribution, assumed_distribution)

        user_distributions = [[0, 0, 3], [0, 0, 3]]
        expected_distribution = [0, 0, 1]
        similarities, most_probable_distribution = get_similarity(user_distributions)
        self.assertSequenceEqual(similarities, [1])
        self.assertSequenceEqual(most_probable_distribution, expected_distribution)

    def test_get_region(self):
        for timezone in range(-12, 15):
            region = get_region(timezone)
            if -10 <= timezone <= -2:
                self.assertEqual(region, "america")
            elif -1 <= timezone <= 3:
                self.assertEqual(region, "europe")
            else:
                self.assertEqual(region, "asia")

    def test_to_graph(self):
        graph = to_graph(dict(all=[5, 3, 2]), "test")
        expected_graph = dict(
            source="test",
            all={2020: dict(total_users=10, uvm=0.5, osvvm=0.3, uvvm=0.2)},
        )
        self.assertDictEqual(graph, expected_graph)

        graph = to_graph(dict(fpga=[5, 3, 2]), "test", 2019, "fpga")
        expected_graph = dict(
            source="test",
            fpga={2019: dict(total_users=10, uvm=0.5, osvvm=0.3, uvvm=0.2)},
        )
        self.assertDictEqual(graph, expected_graph)

    def test_sum_graphs(self):
        graph_1 = dict(fpga={2019: dict(uvm=0.5, osvvm=0.3, uvvm=0.2, total_users=10)})
        graph_2 = dict(fpga={2019: dict(uvm=0.1, osvvm=0.2, uvvm=0.7, total_users=30)})

        res = sum_graphs([graph_1, graph_2], "Test", "fpga", 2019)

        self.assertDictEqual(
            res,
            dict(
                source="Test",
                fpga={
                    2019: dict(uvm=8 / 40, osvvm=9 / 40, uvvm=23 / 40, total_users=40)
                },
            ),
        )

    def test_to_distribution(self):
        graph = dict(
            fpga={2019: dict(uvm=0.5, osvvm=0.3, uvvm=0.2, total_users=10)},
            asic={
                2019: dict(uvm=0.5, osvvm=0.3, uvvm=0.2, total_users=10),
                2020: dict(uvm=0.1, osvvm=0.2, uvvm=0.7, total_users=30),
            },
        )

        distribution = to_distrinution(graph, "asic", 2020)

        expected_distribution = [3, 6, 21]
        self.assertSequenceEqual(distribution, expected_distribution)

    def test_user_data_to_graph(self):
        user_data = dict(
            user_1=dict(test_strategies=dict(a=dict())),
            user_2=dict(test_strategies=dict(c=dict())),
            user_3=dict(test_strategies=dict(a=dict(), b=dict())),
            user_4=dict(test_strategies=dict(b=dict(), c=dict())),
            user_5=dict(test_strategies=dict(b=dict())),
        )

        graph = user_data_to_graph(user_data, "Test", ["a", "b"])

        expected_graph = dict(
            source="Test", all={2020: dict(a=0.4, b=0.6, total_users=5)}
        )
        self.assertDictEqual(graph, expected_graph)

    def test_user_data_to_distribution(self):
        user_data = dict(
            user_1=dict(
                classification="professional",
                timezones={0: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=0, last_commit_time=100)
                ),
            ),
            user_2=dict(
                classification="unknown",
                timezones={0: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=1, last_commit_time=100),
                    uvvm=dict(first_commit_time=2, last_commit_time=100),
                    osvvm=dict(first_commit_time=3, last_commit_time=100),
                ),
            ),
            user_3=dict(
                classification="professional",
                timezones={0: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=4, last_commit_time=100),
                    osvvm=dict(first_commit_time=5, last_commit_time=100),
                ),
            ),
            user_4=dict(
                classification="professional",
                timezones={-5: 1},
                test_strategies=dict(
                    osvvm=dict(first_commit_time=7, last_commit_time=100),
                    uvvm=dict(first_commit_time=6, last_commit_time=100),
                ),
            ),
            user_5=dict(
                classification="professional",
                timezones={0: 1},
                test_strategies=dict(
                    osvvm=dict(first_commit_time=8, last_commit_time=100),
                    vunit=dict(first_commit_time=9, last_commit_time=100),
                ),
            ),
        )

        distribution = user_data_to_distribution(user_data)
        self.assertSequenceEqual(distribution, [3, 4, 2])

        def time_filter(first_commit_time, _last_commit_time):
            return 3 <= first_commit_time <= 6

        distribution = user_data_to_distribution(user_data, time_filter)
        self.assertSequenceEqual(distribution, [1, 2, 1])

        def region_filter(region):
            return region != "america"

        distribution = user_data_to_distribution(
            user_data, time_filter, region_filter=region_filter
        )
        self.assertSequenceEqual(distribution, [1, 2, 0])

        def classification_filter(classification):
            return classification != "unknown"

        distribution = user_data_to_distribution(
            user_data, time_filter, classification_filter, region_filter
        )
        self.assertSequenceEqual(distribution, [1, 1, 0])

    def test_get_total_ci(self):
        lower_bound, _ = proportion_confint(
            40,
            100,
            0.05,
            "binom_test",
        )
        _, upper_bound = proportion_confint(
            60,
            100,
            0.05,
            "binom_test",
        )
        expected_total_ci = 100 * (upper_bound - lower_bound)

        total_ci = get_total_ci([40, 60])

        self.assertAlmostEqual(total_ci, expected_total_ci, places=6)

    def test_scale_distribution(self):
        distribution = [5, 3, 2]
        target = 12
        self.assertSequenceEqual(scale_distribution(target, distribution), [6, 4, 2])

        target = 17
        self.assertSequenceEqual(scale_distribution(target, distribution), [9, 5, 3])

        target = 19
        self.assertSequenceEqual(scale_distribution(target, distribution), [9, 6, 4])

    def test_add_month(self):
        test_date = datetime(2021, 7, 1)
        result_date = add_month(test_date)
        self.assertEqual(result_date.year, 2021)
        self.assertEqual(result_date.month, 8)
        self.assertEqual(result_date.day, 1)

        test_date = datetime(2021, 7, 31)
        result_date = add_month(test_date)
        self.assertEqual(result_date.year, 2021)
        self.assertEqual(result_date.month, 8)
        self.assertEqual(result_date.day, 31)

        test_date = datetime(2021, 12, 13)
        result_date = add_month(test_date)
        self.assertEqual(result_date.year, 2022)
        self.assertEqual(result_date.month, 1)
        self.assertEqual(result_date.day, 13)

    def test_get_active_users_last_2_years(self):
        user_data = dict(
            user_1=dict(
                classification="academic",
                test_strategies=dict(
                    uvm=dict(
                        first_commit_time=datetime(2018, 6, 1).timestamp(),
                        last_commit_time=datetime(2018, 6, 30).timestamp(),
                    )
                ),
            ),
            user_2=dict(
                classification="professional",
                test_strategies=dict(
                    uvm=dict(
                        first_commit_time=datetime(2018, 6, 1).timestamp(),
                        last_commit_time=datetime(2018, 7, 1).timestamp(),
                    )
                ),
            ),
            user_3=dict(
                classification="professional",
                test_strategies=dict(
                    uvm=dict(
                        first_commit_time=datetime(2018, 7, 1).timestamp(),
                        last_commit_time=datetime(2020, 6, 30).timestamp(),
                    )
                ),
            ),
            user_4=dict(
                classification="professional",
                test_strategies=dict(
                    uvm=dict(
                        first_commit_time=datetime(2020, 6, 30).timestamp(),
                        last_commit_time=datetime(2020, 7, 1).timestamp(),
                    )
                ),
            ),
            user_5=dict(
                classification="professional",
                test_strategies=dict(
                    uvm=dict(
                        first_commit_time=datetime(2020, 7, 1).timestamp(),
                        last_commit_time=datetime(2020, 7, 2).timestamp(),
                    )
                ),
            ),
            user_6=dict(
                classification="unknown",
                test_strategies=dict(
                    osvvm=dict(
                        first_commit_time=datetime(2018, 6, 1).timestamp(),
                        last_commit_time=datetime(2018, 6, 30).timestamp(),
                    )
                ),
            ),
        )

        active_users_last_2_years = get_active_users_last_2_years(user_data)

        self.assertEqual(
            active_users_last_2_years["uvm"]["professional"][datetime(2018, 5, 1)], 0
        )
        self.assertEqual(
            active_users_last_2_years["uvm"]["professional"][datetime(2018, 6, 1)], 1
        )
        self.assertEqual(
            active_users_last_2_years["uvm"]["professional"][datetime(2018, 7, 1)], 2
        )
        self.assertEqual(
            active_users_last_2_years["uvm"]["professional"][datetime(2020, 5, 1)], 2
        )
        self.assertEqual(
            active_users_last_2_years["uvm"]["professional"][datetime(2020, 6, 1)], 3
        )
        self.assertEqual(
            active_users_last_2_years["uvm"]["academic"][datetime(2020, 6, 1)], 0
        )
        self.assertEqual(
            active_users_last_2_years["uvm"]["academic"][datetime(2020, 5, 1)], 1
        )
        self.assertEqual(
            active_users_last_2_years["uvm"]["academic"][datetime(2018, 6, 1)], 1
        )
        self.assertEqual(
            active_users_last_2_years["uvm"]["academic"][datetime(2018, 5, 1)], 0
        )
        self.assertEqual(
            active_users_last_2_years["osvvm"]["unknown"][datetime(2020, 6, 1)], 0
        )
        self.assertEqual(
            active_users_last_2_years["osvvm"]["unknown"][datetime(2020, 5, 1)], 1
        )
        self.assertEqual(
            active_users_last_2_years["osvvm"]["unknown"][datetime(2018, 6, 1)], 1
        )
        self.assertEqual(
            active_users_last_2_years["osvvm"]["unknown"][datetime(2018, 5, 1)], 0
        )

    def test_generate_wilson_study_2018_2020(self):
        with TemporaryDirectory() as img_dir:
            img_path = Path(img_dir)
            generate_wilson_study_2018_2020(img_path, ["uvm", "osvvm", "uvvm"])
            self.assertTrue((img_path / "wilson_study_2018_2020.svg").exists())

    def test_generate_wilson_study_vhdl_2018_2020(self):
        with TemporaryDirectory() as img_dir:
            img_path = Path(img_dir)
            meta_data, wilson_graph = get_wilson_data()
            generate_wilson_study_vhdl_2018_2020(
                img_path, wilson_graph, meta_data, ["uvm", "osvvm", "uvvm"]
            )
            self.assertTrue((img_path / "wilson_study_vhdl_2018_2020.svg").exists())

    def test_github_wilson_comparison_2020(self):
        with TemporaryDirectory() as img_dir:
            img_path = Path(img_dir)
            user_data = dict(
                user_1=dict(
                    classification="professional",
                    test_strategies=dict(
                        uvm=dict(),
                        osvvm=dict(),
                        uvvm=dict(),
                    ),
                )
            )
            vhdl_graph = dict(
                source="x",
                all={2020: dict(total_users=100, osvvm=0.1, uvvm=0.1, uvm=0.1)},
            )
            generate_github_wilson_comparison_2020(img_path, user_data, vhdl_graph)
            self.assertTrue((img_path / "github_wilson_comparison_2020.svg").exists())

    def test_generate_github_wilson_combined_comparison_2020(self):
        with TemporaryDirectory() as img_dir:
            img_path = Path(img_dir)
            vhdl_graph = dict(
                source="x",
                all={2020: dict(total_users=100, osvvm=0.1, uvvm=0.1, uvm=0.1)},
            )
            github_graph = vhdl_graph
            generate_github_wilson_combined_comparison_2020(
                img_path, vhdl_graph, github_graph
            )
            self.assertTrue(
                (img_path / "github_wilson_combined_comparison_2020.svg").exists()
            )

    def test_get_temporal_distributions(self):
        def time_filter(first_commit_time, _last_commit_time):
            return first_commit_time == 0

        user_data = dict(
            user_1=dict(
                classification="professional",
                timezones={0: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=0, last_commit_time=1),
                ),
            ),
            user_2=dict(
                classification="academic",
                timezones={-8: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=0, last_commit_time=1),
                    osvvm=dict(first_commit_time=0, last_commit_time=1),
                ),
            ),
            user_3=dict(
                classification="unknown",
                timezones={8: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=0, last_commit_time=1),
                    osvvm=dict(first_commit_time=0, last_commit_time=1),
                    uvvm=dict(first_commit_time=0, last_commit_time=1),
                ),
            ),
            user_4=dict(
                classification="professional",
                timezones={0: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=0, last_commit_time=1),
                ),
            ),
            user_5=dict(
                classification="professional",
                timezones={0: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=0, last_commit_time=1),
                ),
            ),
            user_6=dict(
                classification="academic",
                timezones={-8: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=0, last_commit_time=1),
                ),
            ),
            user_7=dict(
                classification="academic",
                timezones={-8: 1},
                test_strategies=dict(
                    uvm=dict(first_commit_time=1, last_commit_time=1),
                ),
            ),
        )

        distributions = get_temporal_distributions(user_data, time_filter)
        self.assertSequenceEqual(distributions["all"], [6, 2, 1])
        self.assertSequenceEqual(distributions["professional"], [3, 0, 0])
        self.assertSequenceEqual(distributions["academic"], [2, 1, 0])
        self.assertSequenceEqual(distributions["unknown"], [1, 1, 1])
        self.assertSequenceEqual(distributions["known"], [5, 1, 0])
        self.assertSequenceEqual(distributions["america"], [2, 1, 0])
        self.assertSequenceEqual(distributions["europe"], [3, 0, 0])
        self.assertSequenceEqual(distributions["asia"], [1, 1, 1])

    def test_get_temporal_bias_data(self):
        wilson_user_distribution = {2020: [5, 3, 2]}
        github_user_distribution = {2020: [2, 2, 6]}
        study_year = 2020
        first_date = datetime(2020, 6, 1)

        user_data = dict(
            user_1=dict(
                classification="academic",
                timezones={0: 1},
                test_strategies=dict(
                    uvm=dict(
                        first_commit_time=datetime(2020, 6, 1).timestamp(),
                        last_commit_time=datetime(2020, 6, 1).timestamp(),
                    )
                ),
            ),
            user_2=dict(
                classification="academic",
                timezones={0: 1},
                test_strategies=dict(
                    osvvm=dict(
                        first_commit_time=datetime(2020, 7, 1).timestamp(),
                        last_commit_time=datetime(2020, 7, 1).timestamp(),
                    )
                ),
            ),
        )
        temporal_bias_data = get_temporal_bias_data(
            user_data,
            wilson_user_distribution,
            github_user_distribution,
            study_year,
            first_date,
        )

        wilson_similarity, most_probable_distribution = get_similarity(
            [wilson_user_distribution[2020], [0, 1, 0]]
        )
        self.assertEqual(
            temporal_bias_data[str(datetime(2020, 7, 1).timestamp())][
                "wilson_similarity"
            ],
            100 * wilson_similarity[0],
        )

        self.assertEqual(
            temporal_bias_data[str(datetime(2020, 7, 1).timestamp())][
                "early_late_similarity"
            ],
            100
            * get_similarity([[1, 0, 0], [0, 1, 0]], most_probable_distribution)[0][0],
        )

        self.assertEqual(
            temporal_bias_data[str(datetime(2020, 7, 1).timestamp())]["total_ci"],
            get_total_ci([0, 1, 0]),
        )

        self.assertEqual(
            temporal_bias_data[str(datetime(2020, 7, 1).timestamp())]["europe"],
            100,
        )

        self.assertEqual(
            temporal_bias_data[str(datetime(2020, 7, 1).timestamp())]["academic"],
            100,
        )

    def test_generate_temporal_bias_analysis_mid_2018(self):
        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            wilson_user_distribution = {2018: [5, 3, 2], 2020: [5, 3, 2]}
            github_user_distribution = {2018: [2, 2, 6], 2020: [2, 2, 6]}

            user_data = dict(
                user_1=dict(
                    classification="academic",
                    timezones={0: 1},
                    test_strategies=dict(
                        uvm=dict(
                            first_commit_time=datetime(2020, 6, 1).timestamp(),
                            last_commit_time=datetime(2020, 6, 1).timestamp(),
                        )
                    ),
                ),
                user_2=dict(
                    classification="academic",
                    timezones={-8: 1},
                    test_strategies=dict(
                        osvvm=dict(
                            first_commit_time=datetime(2020, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 7, 1).timestamp(),
                        )
                    ),
                ),
                user_3=dict(
                    classification="academic",
                    timezones={0: 1},
                    test_strategies=dict(
                        osvvm=dict(
                            first_commit_time=datetime(2020, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 7, 1).timestamp(),
                        )
                    ),
                ),
                user_4=dict(
                    classification="academic",
                    timezones={8: 1},
                    test_strategies=dict(
                        osvvm=dict(
                            first_commit_time=datetime(2020, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 7, 1).timestamp(),
                        )
                    ),
                ),
                user_5=dict(
                    classification="academic",
                    timezones={-8: 1},
                    test_strategies=dict(
                        osvvm=dict(
                            first_commit_time=datetime(2018, 7, 1).timestamp(),
                            last_commit_time=datetime(2018, 7, 1).timestamp(),
                        )
                    ),
                ),
                user_6=dict(
                    classification="academic",
                    timezones={0: 1},
                    test_strategies=dict(
                        osvvm=dict(
                            first_commit_time=datetime(2018, 7, 1).timestamp(),
                            last_commit_time=datetime(2018, 7, 1).timestamp(),
                        )
                    ),
                ),
                user_7=dict(
                    classification="academic",
                    timezones={8: 1},
                    test_strategies=dict(
                        osvvm=dict(
                            first_commit_time=datetime(2018, 7, 1).timestamp(),
                            last_commit_time=datetime(2018, 7, 1).timestamp(),
                        )
                    ),
                ),
            )

            generate_temporal_bias_analysis_mid_2018_2020(
                output_path,
                output_path,
                user_data,
                wilson_user_distribution,
                github_user_distribution,
                to_graph(
                    {"all": wilson_user_distribution[2020]},
                    "Wilson",
                    year=2020,
                    target="all",
                ),
                to_graph(
                    {"all": github_user_distribution[2020]},
                    "GitHub",
                    year=2020,
                    target="all",
                ),
            )

            for year in [2018, 2020]:
                self.assertTrue(
                    (output_path / f"temporal_bias_analysis_{year}.svg").exists()
                )
                self.assertTrue((output_path / f"temporal_bias_{year}.json").exists())

            self.assertTrue(
                (output_path / "temporal_bias_analysis_mid_2018.svg").exists()
            )

    def test_generate_github_wilson_europe(self):
        with TemporaryDirectory() as img_dir:
            img_path = Path(img_dir)
            vhdl_graph = dict(fpga={2020: dict(total_users=100)})
            split_date = datetime(2018, 7, 1)
            temporal_bias_data = {
                2020: {
                    str(split_date.timestamp()): dict(
                        github_user_distribution_late=dict(europe=[5, 3, 2])
                    )
                }
            }
            generate_github_wilson_europe(
                img_path, vhdl_graph, temporal_bias_data, split_date
            )
            self.assertTrue((img_path / "github_wilson_europe.svg").exists())

    def test_get_region_bias_similarity(self):
        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            github_recent_user_distribution = dict(
                all=[1, 1, 1], america=[1, 0, 0], europe=[0, 1, 0], asia=[0, 0, 1]
            )
            wilson_user_distribution = {2020: [1, 1, 1]}

            _x, _y, z = get_region_bias_similarity(  # pylint: disable=invalid-name
                output_path, github_recent_user_distribution, wilson_user_distribution
            )
            unique, counts = np.unique(
                [round(value, ndigits=7) for value in z], return_counts=True
            )
            stat = dict(zip(unique, counts))
            similarity_3_0_0 = round(
                100 * get_similarity([[3, 0, 0], [1, 1, 1]])[0][0], ndigits=7
            )
            self.assertEqual(stat[similarity_3_0_0], 3)
            similarity_2_1_0 = round(
                100 * get_similarity([[2, 1, 0], [1, 1, 1]])[0][0], ndigits=7
            )
            self.assertEqual(stat[similarity_2_1_0], 6)
            similarity_1_1_1 = 100
            self.assertEqual(stat[similarity_1_1_1], 1)
            self.assertEqual(stat[-1], 6)

            self.assertTrue((output_path / "region_bias_similarity.json").exists())

    def test_generate_region_bias_similarity(self):
        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            github_recent_user_distribution = dict(
                all=[1, 1, 1], america=[1, 0, 0], europe=[0, 1, 0], asia=[0, 0, 1]
            )
            wilson_user_distribution = {2020: [1, 1, 1]}
            temporal_bias_data = {
                2020: {
                    str(datetime(2018, 7, 1).timestamp()): dict(wilson_similarity=17)
                }
            }

            generate_region_bias_similarity(
                output_path,
                output_path,
                github_recent_user_distribution,
                wilson_user_distribution,
                temporal_bias_data,
                datetime(2018, 7, 1),
            )

            self.assertTrue((output_path / "region_bias_similarity.svg").exists())

    def test_generate_github_academic_and_professional_comparison_2020(self):
        with TemporaryDirectory() as img_dir:
            img_path = Path(img_dir)
            github_recent_user_distribution = dict(
                academic=[1, 2, 3], professional=[5, 3, 2]
            )
            vhdl_graph = to_graph(dict(all=[7, 5, 11]), "Wilson", 2020)

            generate_github_academic_and_professional_comparison_2020(
                img_path, github_recent_user_distribution, vhdl_graph
            )
            self.assertTrue(
                (
                    img_path / "github_academic_and_professional_comparison_2020.svg"
                ).exists()
            )

    def test_generate_github_academic_professional_comparison(self):
        with TemporaryDirectory() as img_dir:
            img_path = Path(img_dir)
            user_data = dict(
                user_1=dict(
                    classification="academic",
                    test_strategies=dict(
                        uvm=dict(
                            first_commit_time=datetime(2018, 6, 1).timestamp(),
                            last_commit_time=datetime(2018, 6, 30).timestamp(),
                        )
                    ),
                ),
                user_2=dict(
                    classification="professional",
                    test_strategies=dict(
                        uvm=dict(
                            first_commit_time=datetime(2018, 6, 1).timestamp(),
                            last_commit_time=datetime(2018, 7, 1).timestamp(),
                        )
                    ),
                ),
                user_3=dict(
                    classification="professional",
                    test_strategies=dict(
                        uvm=dict(
                            first_commit_time=datetime(2018, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 6, 30).timestamp(),
                        )
                    ),
                ),
                user_4=dict(
                    classification="professional",
                    test_strategies=dict(
                        uvm=dict(
                            first_commit_time=datetime(2020, 6, 30).timestamp(),
                            last_commit_time=datetime(2020, 7, 1).timestamp(),
                        )
                    ),
                ),
                user_5=dict(
                    classification="professional",
                    test_strategies=dict(
                        uvm=dict(
                            first_commit_time=datetime(2020, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 7, 2).timestamp(),
                        )
                    ),
                ),
            )

            generate_github_academic_professional_comparison(
                img_path, user_data, datetime(2020, 6, 1)
            )

            self.assertTrue(
                (img_path / "github_academic_professional_comparison.svg").exists()
            )

    def test_generate_github_wilson_full_combined_comparison(self):
        with TemporaryDirectory() as img_dir:
            img_path = Path(img_dir)
            user_data = dict(
                user_1=dict(
                    timezones={0: 1},
                    test_strategies=dict(
                        vunit=dict(
                            first_commit_time=datetime(2018, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 6, 30).timestamp(),
                        )
                    ),
                ),
                user_2=dict(
                    timezones={0: 1},
                    test_strategies=dict(
                        cocotb=dict(
                            first_commit_time=datetime(2018, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 6, 30).timestamp(),
                        )
                    ),
                ),
                user_3=dict(
                    timezones={0: 1},
                    test_strategies=dict(
                        uvm=dict(
                            first_commit_time=datetime(2018, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 6, 30).timestamp(),
                        )
                    ),
                ),
                user_4=dict(
                    timezones={0: 1},
                    test_strategies=dict(
                        osvvm=dict(
                            first_commit_time=datetime(2018, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 6, 30).timestamp(),
                        )
                    ),
                ),
                user_5=dict(
                    timezones={0: 1},
                    test_strategies=dict(
                        uvvm=dict(
                            first_commit_time=datetime(2018, 7, 1).timestamp(),
                            last_commit_time=datetime(2020, 6, 30).timestamp(),
                        )
                    ),
                ),
            )
            combined_graph = to_graph(dict(all=[1, 1, 1]), "Test", year=2020)

            generate_github_wilson_full_combined_comparison(
                img_path, user_data, combined_graph
            )

            self.assertTrue(
                (img_path / "github_wilson_full_combined_comparison.svg").exists()
            )

        print("Done")
