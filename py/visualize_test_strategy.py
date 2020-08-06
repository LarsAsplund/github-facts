# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Script for visualizing the derived GitHub statistics."""

import argparse
from json import load
from datetime import date, timedelta
from pathlib import Path
from subprocess import call
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter


def date_range(first_date, last_date):
    """Return dates from first_data to, and including, last_date."""
    current_date = first_date
    while current_date <= last_date:
        yield current_date
        current_date += timedelta(days=1)


def integrate_over_dates(date_dict, first_date, last_date):
    integrated_dict = {first_date - timedelta(days=1): 0}
    for day in date_range(first_date, last_date):
        integrated_dict[day] = integrated_dict[day - timedelta(days=1)] + date_dict[day]

    return integrated_dict


def create_accumulated_histogram_from_dict_with_dates(date_dict, date_function):
    """
    Create an accumulated histogram from a dict with dates.
    
    The dates are founds in the values by applying the provided date function. If
    the date function returns None that value doesn't contribute to the
    accumulated histogram.
    """

    histogram = dict()
    first_date = date(2070, 1, 1)
    last_date = date(1970, 1, 1)
    for value in date_dict.values():
        days = date_function(value)
        if not days:
            continue
        else:
            for day in days:
                if day in histogram:
                    histogram[day] += 1
                else:
                    histogram[day] = 1

        first_date = min(first_date, min(days))
        last_date = max(last_date, max(days))

    accumulated_histogram = []
    for day in date_range(first_date, last_date):
        if day == first_date:
            accumulated_histogram.append(histogram[day])
        elif day in histogram:
            accumulated_histogram.append(accumulated_histogram[-1] + histogram[day])
        else:
            accumulated_histogram.append(accumulated_histogram[-1])

    return accumulated_histogram, first_date, last_date


def extend_accumulated_histogram(
    accumulated_histogram,
    first_date,
    last_date,
    extended_first_date,
    extended_last_date,
):
    if first_date > last_date:
        return [0] * ((extended_last_date - extended_first_date).days + 1)

    result = [0] * (first_date - extended_first_date).days
    result.extend(accumulated_histogram)
    result.extend([accumulated_histogram[-1]] * (extended_last_date - last_date).days)

    return result


def get_percentage_repos_with_tests(
    repos_stat, min_number_of_repos_with_std_test_strategy
):
    """
    Return percentage of repos providing tests and percentage of those using a standard framework.

    Percentages are expressed as yearly averages.
    """
    first_date = date.fromtimestamp(min(repos_stat["created_at"].values()))
    last_date = date.fromtimestamp(max(repos_stat["created_at"].values()))

    if first_date + timedelta(days=364) > last_date:
        return [], [], []

    created_repos = {day: 0 for day in date_range(first_date, last_date)}
    created_repos_with_test = {day: 0 for day in date_range(first_date, last_date)}
    created_repos_with_std_test_strategy = {
        day: 0 for day in date_range(first_date, last_date)
    }

    for repo, data in repos_stat["repo_stat"].items():
        creation_date = date.fromtimestamp(repos_stat["created_at"][repo])
        created_repos[creation_date] += 1
        if data["has_tests"] or data["test_strategies"]:
            created_repos_with_test[creation_date] += 1
        if data["test_strategies"]:
            created_repos_with_std_test_strategy[creation_date] += 1

    total_repos = integrate_over_dates(created_repos, first_date, last_date)
    total_repos_with_test = integrate_over_dates(
        created_repos_with_test, first_date, last_date
    )
    total_repos_with_std_test_strategy = integrate_over_dates(
        created_repos_with_std_test_strategy, first_date, last_date
    )

    first_date_to_plot = first_date + timedelta(days=364)
    for day in date_range(first_date + timedelta(days=364), last_date):
        if (
            total_repos_with_std_test_strategy[day]
            >= min_number_of_repos_with_std_test_strategy
        ):
            first_date_to_plot = day
            break

    def calc_last_year_percentage(subset, total):
        percentage = []

        for current_date in date_range(first_date_to_plot, last_date):
            total_last_year = (
                total[current_date] - total[current_date - timedelta(days=365)]
            )
            subset_last_year = (
                subset[current_date] - subset[current_date - timedelta(days=365)]
            )
            percentage.append(100 * subset_last_year / total_last_year)

        return percentage

    percentage_repos_with_tests = calc_last_year_percentage(
        total_repos_with_test, total_repos
    )
    percentage_repos_with_std_test_strategy = calc_last_year_percentage(
        total_repos_with_std_test_strategy, total_repos_with_test
    )

    timeline = list(date_range(first_date_to_plot, last_date))

    return (
        timeline,
        percentage_repos_with_tests,
        percentage_repos_with_std_test_strategy,
    )


def make_graph_over_time(time, lines, ylabel, title, output_path, show_legend=True):
    _, ax = plt.subplots()
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(MonthLocator())

    max_value = 0
    time_as_list = list(time)
    for framework, values in lines.items():
        ax.plot(
            time_as_list, values, label=fix_framework_name_casing(framework),
        )

        max_value = max(max_value, max(values))

    ax.set_ylim(0, max_value / 0.9)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend()

    if output_path.exists():
        output_path.unlink()
    plt.savefig(output_path, format=output_path.suffix[1:])


def fix_framework_name_casing(lowercase_names):
    """Return proper casing for a standard framework expressed in all lowercase letters."""

    def fix_case(name):
        if name == "vunit":
            return "VUnit"
        elif name == "osvvm":
            return "OSVVM"
        elif name == "uvvm":
            return "UVVM"
        elif name == "uvm":
            return "UVM"
        else:
            return name

    if isinstance(lowercase_names, str):
        return fix_case(lowercase_names)

    uppercase_names = []
    for name in lowercase_names:
        uppercase_names.append(fix_case(name))

    return uppercase_names


def get_repo_framework_distribution(repos_stat, repo_classification):
    """
    Get the number of repositories using each framework.
    
    Returns a dict with keys are framework(s) joined with "&" (for example "VUnit&OSVVM")
    and the values the number of repositories using the combination of frameworks as indicated
    by the key.
    """

    distribution = dict(
        total=dict(), professional=dict(), academic=dict(), unknown=dict()
    )

    for repo, data in repos_stat["repo_stat"].items():
        test_strategies = data["test_strategies"]
        if test_strategies:
            if not repo in repo_classification:
                raise RuntimeError(f"No classification for {repo}.")
            elif repo_classification[repo] not in [
                "professional",
                "academic",
                "unknown",
            ]:
                raise RuntimeError(
                    f"Unknown classification {repo_classification[repo]} for {repo}."
                )

            test_strategies.sort()
            test_strategies = fix_framework_name_casing(test_strategies)
            test_strategies = "&".join(test_strategies)

            for classification in ["total", repo_classification[repo]]:
                if test_strategies in distribution[classification]:
                    distribution[classification][test_strategies] += 1
                else:
                    distribution[classification][test_strategies] = 1

    return distribution


def plot_euler_diagram(distribution, output_path):
    """
    Visualize distribution as an euler diagram and save it to output_path.
    
    This function uses the eulerr package in R.
    """

    group_list = ", ".join(
        [f"'{group}' = {number}" for group, number in distribution.items()]
    )

    fixed_output_path = str(output_path).replace("\\", "/")
    script = f"""
library(eulerr)
set.seed(0)
diagram <- euler(c({group_list}), shape = 'ellipse')
residuals = resid(diagram)
svg('{fixed_output_path}')
if (summary(residuals)[6] >= 1) {{
  legend_text = c("Missing Groups:")
  for (idx in 1:length(residuals)) {{
    if (residuals[idx] >= 1) {{
      legend_text = append(legend_text, paste("-", names(residuals)[idx], ":", residuals[idx]))
    }}
  }}
  plot(diagram, labels = rownames(diagram$ellipses), legend = list(labels = legend_text, pch="", side="bottom"), quantities = TRUE)
}} else {{
  plot(diagram, quantities = TRUE)
}}
d = dev.off()"""

    script_path = output_path.parent / "script.r"
    script_path.write_text(script)

    if output_path.exists():
        output_path.unlink()

    if call(["Rscript", str(script_path)]) != 0:
        raise RuntimeError("Failed to execute R script")

    script_path.unlink()


def visualize_repo_framework_distribution(repos_stat, repo_classification, output_path):
    """
    Visualize the number of repositories using each framework as an Euler diagrams.

    The input classification determines what framework(s) to include in the diagram.
    """

    distributions = get_repo_framework_distribution(repos_stat, repo_classification)
    for classification, distribution in distributions.items():
        plot_euler_diagram(
            distribution,
            output_path / f"{classification}_repo_framework_distribution.svg",
        )


def visualize(
    repos_stat_path,
    repo_classification_path,
    output_path,
    min_number_of_repos_with_std_test_strategy=10,
):
    """Visualize statistics."""
    with open(repos_stat_path) as json:
        repos_stat = load(json)

    with open(repo_classification_path) as json:
        repo_classification = load(json)

    (
        timeline,
        percentage_repos_with_tests,
        percentage_repos_with_std_test_strategy,
    ) = get_percentage_repos_with_tests(
        repos_stat, min_number_of_repos_with_std_test_strategy
    )

    make_graph_over_time(
        timeline,
        dict(data=percentage_repos_with_tests),
        "Percentage",
        "Repositories Providing Tests (1 Year Average)",
        output_path / "repositories_providing_tests.png",
        False,
    )

    make_graph_over_time(
        timeline,
        dict(data=percentage_repos_with_std_test_strategy),
        "Percentage",
        "Repositories with Tests Using a Standard Framework (1 Year Average)",
        output_path / "repositories_using_std_framework.png",
        False,
    )

    visualize_repo_framework_distribution(repos_stat, repo_classification, output_path)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualises Test Strategy Statistics")

    parser.add_argument(
        "repos_stat_path",
        help="JSON file containing test strategy data for all repositories",
        type=Path,
    )

    parser.add_argument(
        "repo_classification_path",
        help="JSON file classifying all repositories with a standard test strategy",
        type=Path,
    )

    parser.add_argument(
        "output_path", help="Directory where plots are saved", type=Path
    )

    args = parser.parse_args()

    visualize(args.repos_stat_path, args.repo_classification_path, args.output_path)


if __name__ == "__main__":
    main()
