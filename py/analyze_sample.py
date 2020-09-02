# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Script for analyzing users of repos using standard frameworks."""

import argparse
from os import listdir
from json import dump, load
from pathlib import Path
from git import Repo
from analyze_users import get_source_files, USER_JSON_VERSION, update_timezones, batch


def get_repo_user_timezones(repo, source_files):
    """Return active user timezones derived from commits to the repo."""
    user_data = dict()
    if not source_files:
        return user_data

    for paths in batch(source_files):
        for commit in repo.iter_commits(paths=paths):
            author = commit.author
            user = f"{author.name} <{author.email}>"
            timezones = dict()
            timezones[commit.author_tz_offset // -3600] = 1
            update_timezones(user_data, user, timezones)

    return user_data


def find_all_repos(repos_root):
    """Find all Git repositories in repos_root."""

    def visit(directory):
        if (directory / ".git").is_dir():
            return [f"{directory.parent.name}/{directory.name}"]

        repo_list = []
        for sub_directory in listdir(directory):
            repo_list += visit(directory / sub_directory)

        return repo_list

    return visit(repos_root)


def get_user_timezones(repos_root, redo):
    """Get user data from all repositories under repos_root."""
    user_data = dict()
    repo_list = find_all_repos(repos_root)
    for iteration, full_repo_name in enumerate(repo_list):
        print(f"{iteration + 1}. Analyzing {full_repo_name}")
        user_name = full_repo_name.split("/")[0]
        repo_name = full_repo_name.split("/")[1]
        user_data_analysis = (
            repos_root / user_name / f"{repo_name}.user.{USER_JSON_VERSION}.json"
        )
        if user_data_analysis.exists() and not redo:
            print("  Already analyzed")
            with open(user_data_analysis) as json:
                repo_user_data = load(json)
        else:
            repo = Repo(str(repos_root / full_repo_name))
            repo_user_data = get_repo_user_timezones(
                repo, get_source_files(repo, [".vhd", ".vhdl"])
            )

        with open(user_data_analysis, "w") as json:
            dump(repo_user_data, json)

        for user, data in repo_user_data.items():
            update_timezones(user_data, user, data["timezones"])

    return user_data


def analyze(repos_root, output_path, redo):
    """Analyze all repos under the given root directory."""
    user_data = get_user_timezones(repos_root, redo,)
    print(f"Total number of users: {len(user_data)}")

    with open(output_path, "w") as json:
        dump(user_data, json)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze users of repositories using a standard framework"
    )
    parser.add_argument(
        "repos_root",
        help="Directory where the cloned repositories are located",
        type=Path,
    )
    parser.add_argument(
        "output_path",
        help="Path to JSON file listing the analysis result for all repositories",
        type=Path,
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        default=False,
        help="Redo user analysis even if previous results exists",
    )

    args = parser.parse_args()

    analyze(
        args.repos_root, args.output_path, args.redo,
    )


if __name__ == "__main__":
    main()
