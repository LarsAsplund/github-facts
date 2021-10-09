# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021, Lars Asplund lars.anders.asplund@gmail.com

"""Script for processing the data from the Wilson study."""

# pylint: disable=too-many-lines
import argparse
from pathlib import Path
from json import load, dump
from itertools import combinations
from time import mktime, strptime
from datetime import datetime
from math import floor
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import numpy as np
from scipy.stats import multinomial
from statsmodels.stats.proportion import proportion_confint
from visualize_test_strategy import fix_framework_name_casing


def get_wilson_data():
    """Return data provided by the Wilson studies."""
    graph = dict(
        source="Wilson",
        fpga={
            2016: dict(
                uvm=0.48,
                pss=0.00,
                ovm=0.23,
                avm=0.09,
                vvm=0.18,
                rvm=0.04,
                erm=0.03,
                urm=0.02,
                osvvm=0.00,
                uvvm=0.00,
                python=0.00,
                other=0.17,
            ),
            2017: dict(
                uvm=0.54,
                pss=0.00,
                ovm=0.21,
                avm=0.12,
                vvm=0.15,
                rvm=0.03,
                erm=0.03,
                urm=0.01,
                osvvm=0.00,
                uvvm=0.00,
                python=0.00,
                other=0.15,
            ),
            2018: dict(
                uvm=0.48,
                pss=0.05,
                ovm=0.16,
                avm=0.14,
                vvm=0,
                rvm=0,
                erm=0,
                urm=0,
                osvvm=0.17,
                uvvm=0.10,
                python=0.00,
                other=0.05,
            ),
            2019: dict(
                uvm=0.52,
                pss=0.05,
                ovm=0.14,
                avm=0.14,
                vvm=0,
                rvm=0,
                erm=0,
                urm=0,
                osvvm=0.14,
                uvvm=0.09,
                python=0.00,
                other=0.06,
            ),
            2020: dict(
                uvm=0.48,
                pss=0.01,
                ovm=0.15,
                avm=0.10,
                vvm=0.09,
                rvm=0.07,
                erm=0.02,
                urm=0.02,
                osvvm=0.18,
                uvvm=0.20,
                python=0.17,
                other=0.12,
            ),
        },
        asic={
            2016: dict(
                uvm=0.71,
                pss=0.00,
                ovm=0.18,
                avm=0.05,
                vvm=0.11,
                rvm=0.04,
                erm=0.08,
                urm=0.04,
                osvvm=0.00,
                uvvm=0.00,
                python=0.00,
                other=0.10,
            ),
            2017: dict(
                uvm=0.75,
                pss=0.00,
                ovm=0.16,
                avm=0.05,
                vvm=0.08,
                rvm=0.02,
                erm=0.07,
                urm=0.04,
                osvvm=0.00,
                uvvm=0.00,
                python=0.00,
                other=0.085,  # Not provided. Set to average between 2016 and 2018.
            ),
            2018: dict(
                uvm=0.71,
                pss=0.02,
                ovm=0.13,
                avm=0.07,
                vvm=0.10,
                rvm=0.04,
                erm=0.06,
                urm=0.06,
                osvvm=0.06,
                uvvm=0.06,
                python=0.00,
                other=0.07,
            ),
            2019: dict(
                uvm=0.74,
                pss=0.04,
                ovm=0.14,
                avm=0.08,
                vvm=0.07,
                rvm=0.03,
                erm=0.07,
                urm=0.05,
                osvvm=0.05,
                uvvm=0.07,
                python=0.00,
                other=0.05,
            ),
            2020: dict(
                uvm=0.74,
                pss=0.03,
                ovm=0.06,
                avm=0.04,
                vvm=0.07,
                rvm=0.04,
                erm=0.05,
                urm=0.05,
                osvvm=0.04,
                uvvm=0.04,
                python=0.11,
                other=0.07,
            ),
        },
    )

    meta_data = {
        2016: dict(
            n=1703, fpga_portion=0.47, fpga_vhdl_portion=0.62, asic_vhdl_portion=0.21
        ),
        2017: dict(
            n=1703, fpga_portion=0.47, fpga_vhdl_portion=0.58, asic_vhdl_portion=0.20
        ),
        2018: dict(
            n=1205, fpga_portion=0.46, fpga_vhdl_portion=0.61, asic_vhdl_portion=0.26
        ),
        2019: dict(
            n=1205, fpga_portion=0.46, fpga_vhdl_portion=0.55, asic_vhdl_portion=0.20
        ),
        2020: dict(
            n=1492, fpga_portion=0.47, fpga_vhdl_portion=0.65, asic_vhdl_portion=0.25
        ),
    }

    years = meta_data.keys()
    frameworks = graph["fpga"][2016].keys()

    # "all" is the FPGA and ASIC data combined
    graph["all"] = dict()
    for year in years:
        graph["all"][year] = dict()
        for framework in frameworks:
            graph["all"][year][framework] = graph["fpga"][year][framework] * meta_data[
                year
            ]["fpga_portion"] + graph["asic"][year][framework] * (
                1 - meta_data[year]["fpga_portion"]
            )

    targets = ["fpga", "asic", "all"]

    # The data for framework usage doesn't sum to one since some study participants use
    # more than one framework
    for target in targets:
        for year in years:
            meta_data[year][f"total_{target}_frameworks"] = sum(
                graph[target][year].values()
            )

    for year in years:
        graph["fpga"][year]["total_users"] = round(
            meta_data[year]["n"] * meta_data[year]["fpga_portion"]
        )
        graph["asic"][year]["total_users"] = (
            meta_data[year]["n"] - graph["fpga"][year]["total_users"]
        )
        graph["all"][year]["total_users"] = meta_data[year]["n"]

        graph["fpga"][year]["total_frameworks"] = meta_data[year][
            "total_fpga_frameworks"
        ]
        graph["asic"][year]["total_frameworks"] = meta_data[year][
            "total_asic_frameworks"
        ]
        graph["all"][year]["total_frameworks"] = meta_data[year]["total_all_frameworks"]

    return meta_data, graph


def calc_bounds(graph, target, year, framework_groups, scale):
    """Calculate bounds of confidence interval."""
    if not year:
        total_users = graph[target]["total_users"]
    else:
        total_users = graph[target][year]["total_users"]
    lower_bounds = []
    upper_bounds = []
    n_users = []
    for framework_group in framework_groups:
        frameworks = framework_group.split("&")
        n_users_in_groups = 0
        for framework in frameworks:
            if not year:
                n_users_in_groups += round(total_users * graph[target][framework])
            else:
                n_users_in_groups += round(total_users * graph[target][year][framework])

        (lower_bound, upper_bound) = proportion_confint(
            n_users_in_groups,
            total_users,
            0.05,
            "binom_test",
        )
        if (total_users * scale) > 0:
            lower_bounds.append(
                100
                * (n_users_in_groups - total_users * lower_bound)
                / (total_users * scale)
            )
            upper_bounds.append(
                100
                * (total_users * upper_bound - n_users_in_groups)
                / (total_users * scale)
            )
            n_users.append(100 * n_users_in_groups / (total_users * scale))
        else:
            lower_bounds.append(0)
            upper_bounds.append(0)
            n_users.append(0)

    return n_users, lower_bounds, upper_bounds


def get_y_pos(framework_groups, graph, year, target, normalize):
    """Get bar graph y positions."""
    y_pos = []
    for framework_group in framework_groups:
        frameworks = framework_group.split("&")
        framework_sum = 0
        for framework in frameworks:
            if not year:
                framework_sum += graph[target][framework] * 100
            else:
                framework_sum += graph[target][year][framework] * 100
        y_pos.append(framework_sum)

    if normalize:
        scale = sum(y_pos) / 100
    else:
        scale = 1

    y_pos = [y / scale if scale > 0 else y for y in y_pos]

    return y_pos, scale


def make_bar_graph(  # pylint: disable=too-many-arguments, too-many-locals
    graphs,
    first_year,
    last_year,
    year_step,
    title,
    targets,
    framework_groups,
    show_confidence=True,
    normalize=False,
    output_path=None,
):
    """Make single bar graph of a set of graphs.

    The plot is optionally saved to file.
    """
    if None in (first_year, last_year, year_step):
        years = range(1)
    else:
        years = range(first_year, last_year + 1, year_step)

    width = 1 / (len(years) * len(graphs) + 1)

    for target in targets:
        _, axis = plt.subplots()
        for year_number, year in enumerate(years):
            for graph_number, graph in enumerate(graphs):
                x_pos = [
                    i
                    - ((len(years) * len(graphs) - 1) / 2) * width
                    + (year_number * len(graphs) + graph_number) * width
                    for i in range(len(framework_groups))
                ]

                y_pos, scale = get_y_pos(
                    framework_groups, graph, year, target, normalize
                )

                if year != 0:
                    label = f"{graph['source']} {year}"
                else:
                    label = f"{graph['source']}"

                axis.bar(x_pos, y_pos, width=width, label=label)

                _, lower_bounds, upper_bounds = calc_bounds(
                    graph, target, year, framework_groups, scale
                )
                for i, _y in enumerate(y_pos):
                    if show_confidence and y_pos[i]:
                        axis.errorbar(
                            x_pos[i],
                            y_pos[i],
                            lower_bounds[i],
                            capsize=3,
                            color="black",
                            uplims=True,
                        )
                        axis.errorbar(
                            x_pos[i],
                            y_pos[i],
                            upper_bounds[i],
                            capsize=3,
                            color="black",
                            lolims=True,
                        )

        axis.set_ylabel(f"Percentage of {target.upper()} users")
        axis.set_title(title)
        axis.set_xticks(range(len(framework_groups)))
        axis.set_xticklabels(fix_framework_name_casing(framework_groups), rotation=45)
        axis.legend()

    if output_path:
        plt.savefig(output_path, format=output_path.suffix[1:])


def extract_vhdl_users(graph, meta_data, all_years, all_frameworks):
    """Extract VHDL users from Wilson study data."""
    vhdl_graph = dict(source="Wilson", fpga=dict(), asic=dict(), all=dict())
    n_remaining_fw_users = dict()

    for year in all_years:
        vhdl_graph["fpga"][year] = dict(
            total_users=round(
                graph["fpga"][year]["total_users"]
                * meta_data[year]["fpga_vhdl_portion"]
            )
        )
        vhdl_graph["asic"][year] = dict(
            total_users=round(
                graph["asic"][year]["total_users"]
                * meta_data[year]["asic_vhdl_portion"]
            )
        )

        vhdl_graph["all"][year] = dict(
            total_users=vhdl_graph["fpga"][year]["total_users"]
            + vhdl_graph["asic"][year]["total_users"]
        )

        n_remaining_fw_users["fpga"] = (
            graph["fpga"][year]["total_frameworks"]
            * vhdl_graph["fpga"][year]["total_users"]
        )
        n_remaining_fw_users["asic"] = (
            graph["asic"][year]["total_frameworks"]
            * vhdl_graph["asic"][year]["total_users"]
        )
        n_remaining_fw_users["all"] = (
            n_remaining_fw_users["fpga"] + n_remaining_fw_users["asic"]
        )

        for framework in ["osvvm", "uvvm"]:
            vhdl_graph["fpga"][year][framework] = round(
                graph["fpga"][year]["total_users"] * graph["fpga"][year][framework]
            )
            vhdl_graph["all"][year][framework] = vhdl_graph["fpga"][year][framework]
            n_remaining_fw_users["fpga"] -= vhdl_graph["fpga"][year][framework]
            vhdl_graph["fpga"][year][framework] /= vhdl_graph["fpga"][year][
                "total_users"
            ]

            vhdl_graph["asic"][year][framework] = round(
                graph["asic"][year]["total_users"] * graph["asic"][year][framework]
            )
            vhdl_graph["all"][year][framework] += vhdl_graph["asic"][year][framework]
            n_remaining_fw_users["asic"] -= vhdl_graph["asic"][year][framework]
            vhdl_graph["asic"][year][framework] /= vhdl_graph["asic"][year][
                "total_users"
            ]

            n_remaining_fw_users["all"] -= vhdl_graph["all"][year][framework]
            vhdl_graph["all"][year][framework] /= vhdl_graph["all"][year]["total_users"]

        for framework in set(all_frameworks) - set(["osvvm", "uvvm"]):
            vhdl_graph["fpga"][year][framework] = (
                round(
                    n_remaining_fw_users["fpga"]
                    * graph["fpga"][year][framework]
                    / (
                        graph["fpga"][year]["total_frameworks"]
                        - graph["fpga"][year]["osvvm"]
                        - graph["fpga"][year]["uvvm"]
                    )
                )
                / vhdl_graph["fpga"][year]["total_users"]
            )

            vhdl_graph["asic"][year][framework] = (
                round(
                    n_remaining_fw_users["asic"]
                    * graph["asic"][year][framework]
                    / (
                        graph["asic"][year]["total_frameworks"]
                        - graph["asic"][year]["osvvm"]
                        - graph["asic"][year]["uvvm"]
                    )
                )
                / vhdl_graph["asic"][year]["total_users"]
            )

            vhdl_graph["all"][year][framework] = (
                round(
                    n_remaining_fw_users["all"]
                    * graph["all"][year][framework]
                    / (
                        graph["all"][year]["total_frameworks"]
                        - graph["all"][year]["osvvm"]
                        - graph["all"][year]["uvvm"]
                    )
                )
                / vhdl_graph["all"][year]["total_users"]
            )

    return vhdl_graph


def get_similarity(
    user_distributions, assumed_distribution=None
):  # pylint: disable=too-many-locals, too-many-branches
    """Get similarity between graphs."""
    n_users = []
    for user_distribution in user_distributions:
        n_users.append(sum(user_distribution))

    if not assumed_distribution:
        most_probable_distribution = [0, 0, 0]
        for idx, _ in enumerate(most_probable_distribution):
            for user_distribution in user_distributions:
                most_probable_distribution[idx] += user_distribution[idx]
            most_probable_distribution[idx] /= sum(n_users)
    else:
        most_probable_distribution = assumed_distribution

    rnd = []
    for idx in range(len(user_distributions)):
        rnd.append(multinomial(n_users[idx], most_probable_distribution))

    similarities = []
    for pair in combinations(range(len(user_distributions)), 2):
        p_combined = rnd[pair[0]].pmf(user_distributions[pair[0]]) * rnd[pair[1]].pmf(
            user_distributions[pair[1]]
        )

        p_sample = []
        for study in pair:
            study_sample_size = n_users[study]
            prob = []
            for n_uvm in range(study_sample_size + 1):
                for n_osvvm in range(study_sample_size + 1 - n_uvm):
                    n_uvvm = study_sample_size - n_uvm - n_osvvm
                    prob.append(rnd[study].pmf([n_uvm, n_osvvm, n_uvvm]))

            prob.sort(reverse=True)
            p_sample.append(prob)

        idx_0 = 0
        idx_1 = 0
        idx_0_stop = len(p_sample[0]) - 1
        idx_1_stop = len(p_sample[1]) - 1
        p_more_probable = 0

        while idx_0 <= idx_0_stop:
            while idx_1 <= idx_1_stop:
                prob = p_sample[0][idx_0] * p_sample[1][idx_1]
                if prob > p_combined:
                    p_more_probable += prob
                else:
                    idx_1_stop = idx_1 - 1
                idx_1 += 1
            idx_1 = 0
            idx_0 += 1

        similarities.append(1 - p_more_probable)

    return similarities, most_probable_distribution


def get_region(timezone):
    """Convert timezone to region name."""
    if -10 <= int(timezone) < -1:
        return "america"

    if -1 <= int(timezone) < 4:
        return "europe"

    return "asia"


def to_graph(distribution, name, year=None, target="all"):
    """Convert distribution to graph."""
    if not year:
        year = 2020

    graph = {"source": name, target: {year: dict()}}
    total_users = sum(distribution[target])
    graph[target][year]["total_users"] = total_users

    for idx, framework in enumerate(["uvm", "osvvm", "uvvm"]):
        if total_users != 0:
            graph[target][year][framework] = distribution[target][idx] / total_users
        else:
            graph[target][year][framework] = 0

    return graph


def sum_graphs(graphs, name, target, year):
    """Sum a set of graphs."""
    summed_graph = dict()
    summed_graph["source"] = name
    summed_graph[target] = dict()
    summed_graph[target][year] = dict()
    summed_graph[target][year]["total_users"] = 0
    for graph in graphs:
        summed_graph[target][year]["total_users"] += graph[target][year]["total_users"]

    for framework in ["uvm", "osvvm", "uvvm"]:
        summed_graph[target][year][framework] = 0
        for graph in graphs:
            summed_graph[target][year][framework] += round(
                graph[target][year]["total_users"] * graph[target][year][framework]
            )

        summed_graph[target][year][framework] /= summed_graph[target][year][
            "total_users"
        ]

    return summed_graph


def to_distrinution(graph, target, year):
    """Convert graph to distribution."""
    distribution = [
        round(graph[target][year][framework] * graph[target][year]["total_users"])
        for framework in ["uvm", "osvvm", "uvvm"]
    ]

    return distribution


def user_data_to_graph(user_data, name, frameworks):
    """Convert user statistics to graph."""
    graph = dict(source=name, all={2020: {framework: 0 for framework in frameworks}})
    for data in user_data.values():
        # Note that "strategy" and "framework" are used interchangeably in the code
        for strategy in data["test_strategies"]:
            if strategy in frameworks:
                graph["all"][2020][strategy] += 1

    graph["all"][2020]["total_users"] = sum(graph["all"][2020].values())
    for strategy in frameworks:
        graph["all"][2020][strategy] /= graph["all"][2020]["total_users"]

    return graph


def user_data_to_distribution(
    user_data,
    time_filter=None,
    classification_filter=None,
    region_filter=None,
    frameworks=None,
):
    """Convert user statistics to distribution."""
    frameworks = ["uvm", "osvvm", "uvvm"] if frameworks is None else frameworks
    distribution = [0] * len(frameworks)

    for data in user_data.values():
        timezone = int(max(data["timezones"], key=data["timezones"].get))
        if region_filter and not region_filter(get_region(timezone)):
            continue

        if classification_filter and not classification_filter(data["classification"]):
            continue

        for strategy in data["test_strategies"]:
            if strategy not in frameworks:
                continue

            if time_filter and not time_filter(
                data["test_strategies"][strategy]["first_commit_time"],
                data["test_strategies"][strategy]["last_commit_time"],
            ):
                continue

            distribution[frameworks.index(strategy)] += 1

    return distribution


def get_total_ci(user_distribution):
    """Get sum of all confidence intervals."""
    total_ci = 0
    for n_users in user_distribution:
        (lower_bound, upper_bound) = proportion_confint(
            n_users,
            sum(user_distribution),
            0.05,
            "binom_test",
        )
        total_ci += 100 * (upper_bound - lower_bound)

    return total_ci


def scale_distribution(target, distribution):
    """Scale distribution to get target sum."""
    factor = target / sum(distribution)
    new_distribution = [
        round(distribution[idx] * factor) for idx in range(len(distribution))
    ]
    decimals = [
        distribution[idx] * factor - floor(distribution[idx] * factor)
        for idx in range(len(distribution))
    ]
    if sum(new_distribution) == target:
        pass
    elif (sum(new_distribution) - target) == 1:
        lowest_decimal_to_be_rounded_up = 1
        for decimal in decimals:
            if (round(decimal) == 1) or (decimal == 0.5):
                if decimal < lowest_decimal_to_be_rounded_up:
                    lowest_decimal_to_be_rounded_up = decimal
        try:
            new_distribution[decimals.index(lowest_decimal_to_be_rounded_up)] -= 1
        except Exception as exc:
            raise RuntimeError(f"{target} {distribution}") from exc
    elif (sum(new_distribution) - target) == -1:
        highest_decimal_to_be_rounded_down = 0
        for decimal in decimals:
            if round(decimal) == 0:
                if decimal > highest_decimal_to_be_rounded_down:
                    highest_decimal_to_be_rounded_down = decimal
        new_distribution[decimals.index(highest_decimal_to_be_rounded_down)] += 1
    else:
        raise RuntimeError("Didn't expect this")

    return new_distribution


def add_month(current_date):
    """Add month to date."""
    if current_date.month == 12:
        return datetime(current_date.year + 1, 1, current_date.day)

    return datetime(current_date.year, current_date.month + 1, current_date.day)


def get_active_users_last_2_years(user_data):
    """Get graph with users active during the last two years."""
    active_users_per_month = dict()
    for framework in ["uvm", "osvvm", "uvvm"]:
        active_users_per_month[framework] = dict()
        for classification in ["professional", "academic", "unknown"]:
            active_users_per_month[framework][classification] = dict()
            current_month = datetime(2013, 1, 1)
            while current_month < datetime(2020, 7, 1):
                active_users_per_month[framework][classification][current_month] = set()
                current_month = add_month(current_month)

    for user, data in user_data.items():
        for strategy in data["test_strategies"]:
            if strategy in ["uvm", "osvvm", "uvvm"]:
                commit_date = datetime.fromtimestamp(
                    data["test_strategies"][strategy]["first_commit_time"]
                )
                current_month = max(
                    datetime(commit_date.year, commit_date.month, 1),
                    datetime(2013, 1, 1),
                )
                commit_date = datetime.fromtimestamp(
                    data["test_strategies"][strategy]["last_commit_time"]
                )
                last_month = min(
                    datetime(commit_date.year, commit_date.month, 1),
                    datetime(2020, 6, 1),
                )

                while current_month <= last_month:
                    active_users_per_month[strategy][data["classification"]][
                        current_month
                    ].add(user)
                    current_month = add_month(current_month)

    active_users_last_2_years = dict()
    for framework in ["uvm", "osvvm", "uvvm"]:
        active_users_last_2_years[framework] = dict()
        for classification in ["professional", "academic", "unknown"]:
            active_users_last_2_years[framework][classification] = dict()
            month = datetime(2015, 1, 1)
            while month <= datetime(2020, 6, 1):
                current_month = datetime(month.year - 2, month.month, month.day)
                current_month = add_month(current_month)
                active_users_last_2_years[framework][classification][month] = set()
                while current_month <= month:
                    active_users_last_2_years[framework][classification][month].update(
                        active_users_per_month[framework][classification][current_month]
                    )

                    current_month = add_month(current_month)

                active_users_last_2_years[framework][classification][month] = len(
                    active_users_last_2_years[framework][classification][month]
                )
                month = add_month(month)

    return active_users_last_2_years


def generate_wilson_study_2018_2020(img_path, all_frameworks):
    """Generate bar graph over Wilson study data between 2018 and 2020."""
    meta_data, wilson_graph = get_wilson_data()

    make_bar_graph(
        [wilson_graph],
        first_year=2018,
        last_year=2020,
        year_step=1,
        title="Framework usage",
        targets=["all"],
        framework_groups=all_frameworks,
        show_confidence=True,
        output_path=img_path / "wilson_study_2018_2020.svg",
    )

    return meta_data, wilson_graph


def generate_wilson_study_vhdl_2018_2020(
    img_path, wilson_graph, meta_data, all_frameworks
):
    """Generate bar graph over Wilson study VHDL data between 2018 and 2020."""
    vhdl_graph = extract_vhdl_users(
        wilson_graph, meta_data, range(2018, 2020 + 1), all_frameworks
    )

    make_bar_graph(
        [vhdl_graph],
        first_year=2018,
        last_year=2020,
        year_step=1,
        title="Framework usage for VHDL designs",
        targets=["all"],
        framework_groups=["uvm", "osvvm", "uvvm"],
        show_confidence=True,
        normalize=True,
        output_path=img_path / "wilson_study_vhdl_2018_2020.svg",
    )

    return vhdl_graph


def generate_github_wilson_comparison_2020(img_path, user_data, vhdl_graph):
    """Generate bar graph over GitHub and Wilson data from 2020."""
    github_graph = user_data_to_graph(user_data, "GitHub", ["uvm", "osvvm", "uvvm"])

    make_bar_graph(
        [vhdl_graph, github_graph],
        first_year=2020,
        last_year=2020,
        year_step=1,
        title="Framework usage for VHDL designs",
        targets=["all"],
        framework_groups=["uvm", "osvvm", "uvvm"],
        show_confidence=True,
        normalize=True,
        output_path=img_path / "github_wilson_comparison_2020.svg",
    )

    return github_graph


def generate_github_wilson_combined_comparison_2020(img_path, vhdl_graph, github_graph):
    """Generate bar graph over GitHub and Wilson data from 2020 compared with the combined study."""
    combined_graph = sum_graphs(
        [vhdl_graph, github_graph], "GitHub + Wilson", "all", 2020
    )

    make_bar_graph(
        [vhdl_graph, combined_graph, github_graph],
        first_year=2020,
        last_year=2020,
        year_step=1,
        title="Framework usage for VHDL designs",
        targets=["all"],
        framework_groups=["uvm", "osvvm", "uvvm"],
        show_confidence=True,
        normalize=True,
        output_path=img_path / "github_wilson_combined_comparison_2020.svg",
    )

    return combined_graph


def make_region_filter(regions):
    """Make filter passing specified regions."""

    def region_filter(region):
        return region in regions

    return region_filter


def make_classification_filter(classifications):
    """Make filter passing specified classifications."""

    def classification_filter(classification):
        return classification in classifications

    return classification_filter


def get_temporal_distributions(user_data, time_filter):
    """Get total, classification, and regional distributions for a time period."""
    return dict(
        all=user_data_to_distribution(user_data, time_filter),
        known=user_data_to_distribution(
            user_data,
            time_filter,
            classification_filter=make_classification_filter(
                ["professional", "academic"]
            ),
        ),
        unknown=user_data_to_distribution(
            user_data,
            time_filter,
            classification_filter=make_classification_filter(["unknown"]),
        ),
        professional=user_data_to_distribution(
            user_data,
            time_filter,
            classification_filter=make_classification_filter(["professional"]),
        ),
        academic=user_data_to_distribution(
            user_data,
            time_filter,
            classification_filter=make_classification_filter(["academic"]),
        ),
        america=user_data_to_distribution(
            user_data,
            time_filter,
            region_filter=make_region_filter(["america"]),
        ),
        europe=user_data_to_distribution(
            user_data,
            time_filter,
            region_filter=make_region_filter(["europe"]),
        ),
        asia=user_data_to_distribution(
            user_data,
            time_filter,
            region_filter=make_region_filter(["asia"]),
        ),
    )


def get_temporal_bias_data(  # pylint: disable=too-many-locals
    user_data,
    wilson_user_distribution,
    github_user_distribution,
    study_year,
    first_date=datetime(2013, 1, 1),
):
    """Get total, classification, and regional distributions for early and late data.

    Distributions are calculated for a range of dates splitting early from late.
    """
    current_date = first_date
    github_user_distribution_early = dict()
    github_user_distribution_late = dict()
    temporal_bias_data = dict()
    last_date = datetime(study_year, 7, 1)

    while current_date <= last_date:
        current_timestamp = current_date.timestamp()

        def early_time_filter(_first_commit_time, last_commit_time):
            return last_commit_time < current_timestamp

        def late_time_filter(first_commit_time, last_commit_time):
            return (last_commit_time >= current_timestamp) and not (
                first_commit_time > last_date.timestamp()
            )

        github_user_distribution_early[current_timestamp] = get_temporal_distributions(
            user_data, early_time_filter
        )
        github_user_distribution_late[current_timestamp] = get_temporal_distributions(
            user_data, late_time_filter
        )
        (
            similarities_github_wilson,
            most_probably_distribution_github_wilson,
        ) = get_similarity(
            [
                github_user_distribution_late[current_timestamp]["all"],
                wilson_user_distribution[study_year],
            ]
        )

        similarities_github_early_late, _ = get_similarity(
            [
                github_user_distribution_early[current_timestamp]["all"],
                github_user_distribution_late[current_timestamp]["all"],
            ],
            most_probably_distribution_github_wilson,
        )

        total_ci = get_total_ci(github_user_distribution_late[current_timestamp]["all"])

        print(
            "Early/late similarity in %s = %.1f%%. Wilson similarity = %.1f%%. "
            "Percentage of users %.1f%%, Total error = %.1f%%."
            % (
                current_date,
                100 * similarities_github_early_late[0],
                100 * similarities_github_wilson[0],
                100
                * sum(github_user_distribution_late[current_timestamp]["all"])
                / sum(github_user_distribution[study_year]),
                total_ci,
            )
        )

        temporal_bias_data[str(current_timestamp)] = dict(
            early_late_similarity=100 * similarities_github_early_late[0],
            wilson_similarity=100 * similarities_github_wilson[0],
            total_ci=total_ci,
            github_user_distribution_early=github_user_distribution_early[
                current_timestamp
            ],
            github_user_distribution_late=github_user_distribution_late[
                current_timestamp
            ],
            america=100
            * (
                sum(github_user_distribution_late[current_timestamp]["america"])
                / sum(github_user_distribution_late[current_timestamp]["all"])
            ),
            europe=100
            * (
                sum(github_user_distribution_late[current_timestamp]["europe"])
                / sum(github_user_distribution_late[current_timestamp]["all"])
            ),
            asia=100
            * (
                sum(github_user_distribution_late[current_timestamp]["asia"])
                / sum(github_user_distribution_late[current_timestamp]["all"])
            ),
            professional=100
            * (
                sum(github_user_distribution_late[current_timestamp]["professional"])
                / sum(github_user_distribution_late[current_timestamp]["known"])
            ),
            academic=100
            * (
                sum(github_user_distribution_late[current_timestamp]["academic"])
                / sum(github_user_distribution_late[current_timestamp]["known"])
            ),
        )
        current_date = add_month(current_date)

    return temporal_bias_data


def generate_temporal_bias_analysis_mid_2018_2020(  # pylint: disable=too-many-arguments, too-many-locals
    data_path,
    img_path,
    user_data,
    wilson_user_distribution,
    github_user_distribution,
    vhdl_graph,
    github_graph,
):
    """Generate graphs with similarity between Wilson data and early/late GitHub data."""
    temporal_bias_data = dict()
    for study_year in [2018, 2020]:
        print(f"Study year = {study_year}")

        result_path = data_path / f"temporal_bias_{study_year}.json"
        if not result_path.exists():
            temporal_bias_data[study_year] = get_temporal_bias_data(
                user_data,
                wilson_user_distribution,
                github_user_distribution,
                study_year,
            )
            with open(result_path, "w") as fptr:
                dump(temporal_bias_data[study_year], fptr)
        else:
            with open(result_path) as fptr:
                temporal_bias_data[study_year] = load(fptr)

        _, axis = plt.subplots()
        dates = [
            datetime.fromtimestamp(float(timestamp))
            for timestamp in temporal_bias_data[study_year]
        ]
        axis.plot(
            dates,
            [
                temporal_bias_data[study_year][timestamp]["early_late_similarity"]
                for timestamp in temporal_bias_data[study_year]
            ],
            label="GitHub recent/past similarity",
        )
        axis.plot(
            dates,
            [
                temporal_bias_data[study_year][timestamp]["wilson_similarity"]
                for timestamp in temporal_bias_data[study_year]
            ],
            label="Wilson/GitHub recent similarity",
        )
        axis.plot(
            dates,
            [
                temporal_bias_data[study_year][timestamp]["total_ci"]
                for timestamp in temporal_bias_data[study_year]
            ],
            label="Uncertainty",
        )

        axis.set_xlabel("Separation date")
        axis.set_ylabel("[%]")
        axis.set_title(
            f"Similarity and Uncertainty for Past and Recent GitHub Data {study_year}"
        )
        axis.legend()

        split_date = datetime(study_year - 2, 7, 1)
        split_date_as_str = split_date.strftime("%d/%m")

        x_ticks = [
            mdates.date2num(datetime(year, 1, 1))
            for year in range(2013, study_year + 1)
        ]
        x_ticks.append(mdates.date2num(split_date))
        x_tick_labels = [str(year) for year in range(2013, study_year + 1)]
        x_tick_labels.append(split_date_as_str)
        plt.xticks(x_ticks, x_tick_labels, rotation=90)
        plt.axhline(y=5, linestyle="--", color="black")
        plt.axhline(y=100, linestyle="--", color="black")
        plt.axvline(
            x=mdates.date2num(datetime(study_year - 2, 7, 1)),
            linestyle=":",
            color="black",
        )

        plt.savefig(img_path / f"temporal_bias_analysis_{study_year}.svg", format="svg")

    timestamp = str(split_date.timestamp())
    print(
        "Recent Github/Wilson similarity: "
        f"{temporal_bias_data[2020][timestamp]['wilson_similarity']}"
    )
    print(
        "Past/recent Github similarity: "
        f"{temporal_bias_data[2020][timestamp]['early_late_similarity']}"
    )

    print(f"Total CI for Github data {get_total_ci(github_user_distribution[2020])}%")

    total_ci = get_total_ci(
        temporal_bias_data[2020][timestamp]["github_user_distribution_late"]["all"]
    )
    print("Total CI for recent Github data " f"{total_ci}%")

    make_bar_graph(
        [
            vhdl_graph,
            github_graph,
            to_graph(
                temporal_bias_data[2020][str(split_date.timestamp())][
                    "github_user_distribution_late"
                ],
                f"GitHub after {split_date_as_str}",
            ),
            to_graph(
                temporal_bias_data[2020][str(split_date.timestamp())][
                    "github_user_distribution_early"
                ],
                f"GitHub before {split_date_as_str}",
            ),
        ],
        2020,
        2020,
        1,
        "Temporal Bias Analysis",
        ["all"],
        ["uvm", "osvvm", "uvvm"],
        True,
        True,
        output_path=img_path / "temporal_bias_analysis_mid_2018.svg",
    )

    return temporal_bias_data


def generate_github_wilson_europe(img_path, vhdl_graph, temporal_bias_data, split_date):
    """Generate bar graph comparing Wilson and GitHub data for Europe."""
    vhdl_europe_fpga = dict(
        uvm=0.27,
        pss=0.00,
        ovm=0.17,
        avm=0.09,
        vvm=0.03,
        rvm=0.01,
        erm=0.03,
        urm=0.01,
        osvvm=0.34,
        uvvm=0.38,
        python=0.32,
        other=0.10,
    )

    wilson_european_participant_portion = 0.26
    wilson_europe_fpga_user_distribution = dict(
        europe=[
            round(
                vhdl_europe_fpga[framework]
                * vhdl_graph["fpga"][2020]["total_users"]
                * wilson_european_participant_portion
            )
            for framework in ["uvm", "osvvm", "uvvm"]
        ]
    )

    wilson_graph_europe = to_graph(
        wilson_europe_fpga_user_distribution,
        "Wilson FPGA Europe",
        2020,
        target="europe",
    )

    github_recent_user_distribution = temporal_bias_data[2020][
        str(split_date.timestamp())
    ]["github_user_distribution_late"]

    github_recent_graph_europe = to_graph(
        github_recent_user_distribution, "GitHub Europe", 2020, target="europe"
    )
    make_bar_graph(
        [wilson_graph_europe, github_recent_graph_europe],
        2020,
        2020,
        1,
        "Framework usage for European VHDL designs",
        ["europe"],
        ["uvm", "osvvm", "uvvm"],
        True,
        True,
        output_path=img_path / "github_wilson_europe.svg",
    )

    similarities_europe, _ = get_similarity(
        [
            wilson_europe_fpga_user_distribution["europe"],
            github_recent_user_distribution["europe"],
        ]
    )
    print(
        f"Wilson FPGA Europe/GitHub Europe similarity = {100 * similarities_europe[0]}%"
    )

    return github_recent_user_distribution


def get_region_bias_similarity(  # pylint: disable=too-many-locals
    data_path, github_recent_user_distribution, wilson_user_distribution
):
    """Get similarity between Wilson and recent GitHub data for different regional biases."""
    result_path = data_path / "region_bias_similarity.json"
    step = 1
    x, y = np.meshgrid(  # pylint: disable=invalid-name
        np.arange(0, sum(github_recent_user_distribution["all"]) + 1, step),
        np.arange(0, sum(github_recent_user_distribution["all"]) + 1, step),
    )

    if not result_path.exists():
        z = np.full(x.shape, -1.0)  # pylint: disable=invalid-name

        cache = dict()
        for n_america_users in range(
            0, sum(github_recent_user_distribution["all"]) + 1, step
        ):
            for n_europe_users in range(
                0,
                sum(github_recent_user_distribution["all"]) + 1 - n_america_users,
                step,
            ):
                n_asia_users = (
                    sum(github_recent_user_distribution["all"])
                    - n_america_users
                    - n_europe_users
                )

                america_users = scale_distribution(
                    n_america_users, github_recent_user_distribution["america"]
                )
                europe_users = scale_distribution(
                    n_europe_users, github_recent_user_distribution["europe"]
                )
                asia_users = scale_distribution(
                    n_asia_users, github_recent_user_distribution["asia"]
                )

                github_mix = [
                    america_users[idx] + europe_users[idx] + asia_users[idx]
                    for idx in range(3)
                ]
                similarity = cache.get(tuple(github_mix))
                if not similarity:
                    similarities, _ = get_similarity(
                        [wilson_user_distribution[2020], github_mix]
                    )
                    similarity = similarities[0]
                    cache[tuple(github_mix)] = similarity
                else:
                    print(f"Found {tuple(github_mix)} in cache: {similarity}")

                z[n_europe_users // step][n_america_users // step] = similarity * 100

                print(
                    f"{github_mix} ({america_users}/{europe_users}/{asia_users}, "
                    f"{sum(america_users)}/{sum(europe_users)}/{sum(asia_users)}),  : "
                    f"{100 * similarity}%"
                )

        with open(result_path, "w") as fptr:
            dump(z.tolist(), fptr)
    else:
        with open(result_path) as fptr:
            z = np.array(load(fptr))  # pylint: disable=invalid-name

    return x.ravel(), y.ravel(), z.ravel()


def generate_region_bias_similarity(  # pylint: disable=too-many-arguments, too-many-locals
    data_path,
    img_path,
    github_recent_user_distribution,
    wilson_user_distribution,
    temporal_bias_data,
    split_date,
):
    """Generate plot for similarity between Wilson and recent GitHub data with regional biases."""
    x, y, z = get_region_bias_similarity(  # pylint: disable=invalid-name
        data_path, github_recent_user_distribution, wilson_user_distribution
    )

    x_category = dict(less_than_5=[], more_than_5=[], original=[])
    y_category = dict(less_than_5=[], more_than_5=[], original=[])
    z_category = dict(less_than_5=[], more_than_5=[], original=[])
    colors = dict(less_than_5=[], more_than_5=[], original=[])
    sizes = dict(less_than_5=[], more_than_5=[], original=[])

    z_max = max(z)

    for idx, _ in enumerate(z):
        if z[idx] < 0:
            continue

        category = "less_than_5" if z[idx] <= 5 else "more_than_5"

        x_category[category].append(
            100 * x[idx] / sum(github_recent_user_distribution["all"])
        )
        y_category[category].append(
            100 * y[idx] / sum(github_recent_user_distribution["all"])
        )
        z_category[category].append(z[idx])
        sizes[category].append(2)
        if category == "less_than_5":
            colors[category].append((1 - 0.5 * z[idx] / 5, 0, 0))
        else:
            colors[category].append((0, 0.5 + 0.5 * z[idx] / z_max, 0))

    x_category["original"].append(
        100
        * sum(github_recent_user_distribution["america"])
        / sum(github_recent_user_distribution["all"])
    )
    y_category["original"].append(
        100
        * sum(github_recent_user_distribution["europe"])
        / sum(github_recent_user_distribution["all"])
    )

    wilson_github_recent_similarity = temporal_bias_data[2020][
        str(split_date.timestamp())
    ]["wilson_similarity"]
    z_category["original"].append(wilson_github_recent_similarity)
    colors["original"].append((0, 0, 1))
    sizes["original"].append(25)

    fig = plt.figure()
    axis = fig.add_subplot(projection="3d")
    axis.view_init(azim=-25)

    labels = dict(
        less_than_5="Similarity <= 5%",
        more_than_5="Similarity > 5%",
        original="Original GitHub study\nSimilarity = %.1f%%"
        % (wilson_github_recent_similarity),
    )
    for category in x_category:
        axis.scatter(
            x_category[category],
            y_category[category],
            z_category[category],
            c=colors[category],
            label=labels[category],
            s=sizes[category],
        )

    axis.set_xlabel("Users in North and South America [%]")
    axis.set_ylabel("Users in Europe and Africa [%]")
    axis.set_zlabel("Similarity with Wilson Study [%]")
    axis.set_title("GitHub/Wilson Similarity with Regional Bias")
    axis.legend(bbox_to_anchor=(-0.1, 0.8), loc="upper left")

    plt.savefig(img_path / "region_bias_similarity.svg", format="svg")


def generate_github_academic_and_professional_comparison_2020(
    img_path, github_recent_user_distribution, vhdl_graph
):
    """Generate bar graph comparing Wilson data and GitHub professional & academic data for 2020."""
    github_recent_graph_professional = to_graph(
        github_recent_user_distribution,
        "GitHub Professional",
        2020,
        target="professional",
    )
    github_recent_graph_professional["all"] = github_recent_graph_professional[
        "professional"
    ].copy()

    github_recent_graph_academic = to_graph(
        github_recent_user_distribution, "GitHub Academic", 2020, target="academic"
    )
    github_recent_graph_academic["all"] = github_recent_graph_academic[
        "academic"
    ].copy()

    make_bar_graph(
        [
            vhdl_graph,
            github_recent_graph_academic,
            github_recent_graph_professional,
        ],
        2020,
        2020,
        1,
        "Framework usage for VHDL designs",
        ["all"],
        ["uvm", "osvvm", "uvvm"],
        True,
        True,
        output_path=img_path / "github_academic_and_professional_comparison_2020.svg",
    )


def generate_github_academic_professional_comparison(
    img_path, user_data, start_date=None
):
    """Generate graph comparing GitHub professional and academic data over time."""
    active_users_last_2_years = get_active_users_last_2_years(user_data)
    current_month = datetime(2015, 1, 1) if start_date is None else start_date
    professional_academic_similarity = dict()
    while current_month <= datetime(2020, 6, 1):
        professional_distribution = [
            active_users_last_2_years[framework]["professional"][current_month]
            for framework in ["uvm", "osvvm", "uvvm"]
        ]
        academic_distribution = [
            active_users_last_2_years[framework]["academic"][current_month]
            for framework in ["uvm", "osvvm", "uvvm"]
        ]

        similarities, _ = get_similarity(
            [
                professional_distribution,
                academic_distribution,
            ]
        )

        professional_academic_similarity[current_month] = 100 * similarities[0]

        print(f"{current_month} : {100 * similarities[0]}")

        current_month = add_month(current_month)

    _, axis = plt.subplots()
    axis.plot(
        list(professional_academic_similarity.keys()),
        list(professional_academic_similarity.values()),
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Similarity [%]")
    axis.set_title(
        "GitHub academic and professional data similarity. Two-year average."
    )
    output_path = img_path / "github_academic_professional_comparison.svg"
    plt.savefig(output_path, format=output_path.suffix[1:])


def generate_github_wilson_full_combined_comparison(  # pylint: disable=too-many-locals
    img_path, user_data, combined_graph
):
    """Generate bar graph comparing all studied frameworks using data from both studies."""

    def last_2_years_filter(first_commit_time, last_commit_time):
        return (first_commit_time < datetime(2020, 7, 1).timestamp()) and (
            last_commit_time >= datetime(2018, 7, 1).timestamp()
        )

    frameworks = ["vunit", "cocotb", "uvm", "osvvm", "uvvm"]
    github_last_two_years = user_data_to_distribution(
        user_data,
        last_2_years_filter,
        frameworks=frameworks,
    )
    total = sum(github_last_two_years)
    combined_distribution = dict()
    lower_bound = dict()
    upper_bound = dict()

    for framework in ["vunit", "cocotb"]:
        combined_distribution[framework] = (
            100 * github_last_two_years[frameworks.index(framework)] / total
        )
        (lower_bound[framework], upper_bound[framework]) = proportion_confint(
            github_last_two_years[frameworks.index(framework)],
            total,
            0.05,
            "binom_test",
        )
        lower_bound[framework] = (
            combined_distribution[framework] - 100 * lower_bound[framework]
        )
        upper_bound[framework] = (
            100 * upper_bound[framework] - combined_distribution[framework]
        )
    portion_other = 1 - sum(combined_distribution.values()) / 100

    for framework in ["uvm", "osvvm", "uvvm"]:
        combined_distribution[framework] = 100 * (
            combined_graph["all"][2020][framework] * portion_other
        )
        (lower_bound[framework], upper_bound[framework]) = proportion_confint(
            round(
                combined_graph["all"][2020][framework]
                * combined_graph["all"][2020]["total_users"]
            ),
            combined_graph["all"][2020]["total_users"],
            0.05,
            "binom_test",
        )
        lower_bound[framework] *= portion_other
        upper_bound[framework] *= portion_other
        lower_bound[framework] = (
            combined_distribution[framework] - 100 * lower_bound[framework]
        )
        upper_bound[framework] = (
            100 * upper_bound[framework] - combined_distribution[framework]
        )

    _, axis = plt.subplots()
    axis.bar(
        list(range(len(combined_distribution))),
        combined_distribution.values(),
        label="Wilson and GitHub studies combined",
    )
    for idx, (key, value) in enumerate(combined_distribution.items()):
        axis.errorbar(
            idx,
            value,
            lower_bound[key],
            capsize=3,
            color="black",
            uplims=True,
        )
        axis.errorbar(
            idx,
            value,
            upper_bound[key],
            capsize=3,
            color="black",
            lolims=True,
        )
    axis.set_ylabel("Percentage of users")
    axis.set_title("Framework usage for VHDL designs")
    axis.set_xticks(range(len(combined_distribution)))
    axis.set_xticklabels(
        fix_framework_name_casing(combined_distribution.keys()), rotation=45
    )
    axis.legend()
    plt.savefig(img_path / "github_wilson_full_combined_comparison.svg", format="svg")


def main():  # pylint: disable=too-many-locals
    """Parse command line arguments and generate statistics."""
    parser = argparse.ArgumentParser(
        description="Provides Statistics Relating the GitHub Study Results "
        "to the Results of the Wilson Study"
    )

    parser.parse_args()

    all_frameworks = [
        "uvm",
        "pss",
        "ovm",
        "avm",
        "vvm",
        "rvm",
        "erm",
        "urm",
        "osvvm",
        "uvvm",
        "python",
        "other",
    ]

    root = Path(__file__).parent
    img_path = root / ".." / "img"
    data_path = root / ".."
    with open(data_path / "user_stat.json") as json:
        user_data = load(json)

    meta_data, wilson_graph = generate_wilson_study_2018_2020(img_path, all_frameworks)
    vhdl_graph = generate_wilson_study_vhdl_2018_2020(
        img_path, wilson_graph, meta_data, all_frameworks
    )
    github_graph = generate_github_wilson_comparison_2020(
        img_path, user_data, vhdl_graph
    )
    combined_graph = generate_github_wilson_combined_comparison_2020(
        img_path, vhdl_graph, github_graph
    )

    ###########################################

    wilson_user_distribution = dict()
    for year in range(2018, 2021):
        wilson_user_distribution[year] = to_distrinution(vhdl_graph, "all", year)

    github_user_distribution = dict()
    for year in range(2018, 2021):

        def time_filter(first_commit_time, _last_commit_time):
            if year == 2019:
                return first_commit_time < mktime(strptime("2020-01-01", "%Y-%m-%d"))

            return first_commit_time < mktime(strptime(f"{year}-07-01", "%Y-%m-%d"))

        github_user_distribution[year] = user_data_to_distribution(
            user_data, time_filter
        )

    (
        similarities_github_wilson,
        _most_probably_distribution_github_wilson,
    ) = get_similarity(
        [
            github_user_distribution[2020],
            wilson_user_distribution[2020],
        ]
    )

    print(f"Github/Wilson similarity 2020: {similarities_github_wilson[0] * 100}")

    ##############################

    temporal_bias_data = generate_temporal_bias_analysis_mid_2018_2020(
        data_path,
        img_path,
        user_data,
        wilson_user_distribution,
        github_user_distribution,
        vhdl_graph,
        github_graph,
    )

    split_date = datetime(2018, 7, 1)
    github_recent_user_distribution = generate_github_wilson_europe(
        img_path, vhdl_graph, temporal_bias_data, split_date
    )

    generate_region_bias_similarity(
        data_path,
        img_path,
        github_recent_user_distribution,
        wilson_user_distribution,
        temporal_bias_data,
        split_date,
    )
    generate_github_academic_and_professional_comparison_2020(
        img_path, github_recent_user_distribution, vhdl_graph
    )
    generate_github_academic_professional_comparison(img_path, user_data)
    generate_github_wilson_full_combined_comparison(img_path, user_data, combined_graph)

    plt.show()


if __name__ == "__main__":
    main()
