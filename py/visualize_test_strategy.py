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

    def date_range(first_date, last_date):
        current_date = first_date
        while current_date <= last_date:
            yield current_date
            current_date += timedelta(days=1)

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

    def integrate(created_repos):
        total_repos = {first_date - timedelta(days=1): 0}
        for day in date_range(first_date, last_date):
            total_repos[day] = total_repos[day - timedelta(days=1)] + created_repos[day]

        return total_repos

    total_repos = integrate(created_repos)
    total_repos_with_test = integrate(created_repos_with_test)
    total_repos_with_std_test_strategy = integrate(created_repos_with_std_test_strategy)

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


def make_graph_over_time(time, values, ylabel, title, output_path):
    """
    Plot the provided values as a function of time.

    The plot is saved as an image on a format based on the extension on the provided output_path.
    """
    _fig, axes = plt.subplots()
    axes.xaxis.set_major_locator(YearLocator())
    axes.xaxis.set_major_formatter(DateFormatter("%Y"))
    axes.xaxis.set_minor_locator(MonthLocator())

    axes.plot(time, values)

    axes.set_ylim(0, max(values) / 0.9)
    axes.set_xlabel("Date")
    axes.set_ylabel(ylabel)
    axes.set_title(title)

    plt.savefig(output_path, format=output_path.suffix[1:])


def get_repo_framework_distribution(repos_stat, repo_classification):
    """
    Get the number of repositories using each framework.
    
    Returns a dict with keys are framework(s) joined with "&" (for example "VUnit&OSVVM")
    and the values the number of repositories using the combination of frameworks as indicated
    by the key.
    """

    def fix_framework_name_casing(lowercase_names):
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

        uppercase_names = []
        for name in lowercase_names:
            uppercase_names.append(fix_case(name))

        return uppercase_names

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
        percentage_repos_with_tests,
        "Percentage",
        "Repositories Providing Tests (1 Year Average)",
        output_path / "repositories_providing_tests.png",
    )

    make_graph_over_time(
        timeline,
        percentage_repos_with_std_test_strategy,
        "Percentage",
        "Repositories with Tests Using a Standard Framework (1 Year Average)",
        output_path / "repositories_using_std_framework.png",
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
