# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Script for analyzing users of repos using standard frameworks."""

import argparse
from json import dump, load
from pathlib import Path
from datetime import datetime
from git import Repo
from fuzzywuzzy import fuzz
from github_clone import clone
from analyze_test_strategy import analyze_test_strategy

USER_JSON_VERSION = 1


def get_repos_using_a_std_framework(repos_stat):
    """Return repos using a standard framework."""
    repo_list = []
    for repo, data in repos_stat["repo_stat"].items():
        if data["test_strategies"]:
            repo_list.append(repo)
    return repo_list


def get_source_files(repo):
    """Return current VHDL, SystemVerilog and Python source files."""
    extensions = [".vhd", ".vhdl", ".py", ".sv"]
    source_files = []
    try:
        head = repo.head.ref.commit
    except ValueError:  # No head
        return []
    for item in head.tree.traverse():
        if item.type == "blob":
            if Path(item.path).suffix.lower() in extensions:
                source_files.append(item.path)

    return source_files


def get_current_frameworks(repo, source_files):
    """Return the standard frameworks found in the HEAD of the repo."""
    head = repo.head.ref.commit
    current_frameworks = set()
    for source_file in source_files:
        code = (head.tree / source_file).data_stream.read().decode(encoding="latin-1")
        test_strategies = analyze_test_strategy(code, Path(source_file).suffix)
        current_frameworks.update(test_strategies)

    return current_frameworks


def resolve_classification(classification_1, classification_2):
    """Return the classification of a user when there are two different classifications."""
    if (classification_1 == "professional") or (classification_2 == "professional"):
        return "professional"

    if (classification_1 == "academic") or (classification_2 == "academic"):
        return "academic"

    return "unknown"


def update_data(user_data, user, strategy, commit_time, classification):
    """Update collected user data with new information."""
    if user not in user_data:
        user_data[user] = dict(test_strategies={strategy: commit_time})

    elif strategy not in user_data[user]["test_strategies"]:
        user_data[user]["test_strategies"][strategy] = commit_time

    else:
        user_data[user]["test_strategies"][strategy] = min(
            user_data[user]["test_strategies"][strategy], commit_time,
        )

    if "classification" not in user_data[user]:
        user_data[user]["classification"] = classification
    else:
        user_data[user]["classification"] = resolve_classification(
            user_data[user]["classification"], classification
        )


def get_repo_user_data(repo, source_files, classification):
    """
    Return the user data derived from commits to the repo.

    User data includes classification and when the user started to work with standard frameworks.
    The latter is derived from the earliest commit to a file using such a framework.
    """
    user_data = dict()
    dropped_test_strategies = set()
    if not source_files:
        return user_data, dropped_test_strategies

    current_frameworks = get_current_frameworks(repo, source_files)

    # GitPython can crash if handling large batches of source files. 100 seems to be working¨
    # fine.
    def batch(full_list):
        """Yield successive 100-sized batches."""
        for idx in range(0, len(full_list), 100):
            yield full_list[idx : idx + 100]

    for batch in batch(source_files):
        commit_no = 0
        for commit in repo.iter_commits(paths=batch):
            commit_no += 1
            author = commit.author
            user = f"{author.name} <{author.email}>"
            for source_file in commit.stats.files.keys():
                if Path(source_file).suffix not in [".vhd", ".vhdl", ".py", ".sv"]:
                    continue
                try:
                    code = (
                        (commit.tree / source_file)
                        .data_stream.read()
                        .decode(encoding="latin-1")
                    )
                except KeyError:  # File was deleted
                    continue

                test_strategies = analyze_test_strategy(code, Path(source_file).suffix)

                for strategy in test_strategies:
                    if strategy not in current_frameworks:
                        dropped_test_strategies.update([strategy])
                    else:
                        update_data(
                            user_data,
                            user,
                            strategy,
                            datetime.timestamp(commit.authored_datetime),
                            classification,
                        )

    return user_data, dropped_test_strategies


def get_potential_aliases(user_experience):
    """
    Return potential aliases for the same user.

    A potential alias is one that has the same email as another user or if two user names
    are sufficiently similar.
    """
    potential_aliases = []
    for i, user_1 in enumerate(user_experience):
        for j, user_2 in enumerate(user_experience):
            if i >= j:
                continue
            name_1 = user_1[: user_1.rfind("<") - 1]
            name_2 = user_2[: user_2.rfind("<") - 1]
            email_1 = user_1[user_1.rfind("<") + 1 : -1]
            email_2 = user_2[user_2.rfind("<") + 1 : -1]
            if (fuzz.ratio(ascii(name_1), ascii(name_2)) > 70) or (email_1 == email_2):
                potential_aliases.append((user_1, user_2))

    return potential_aliases


def remove_aliases(user_experience, user_aliases):
    """Remove information about aliases by moving it to the original user name."""
    result = dict()
    for user, data in user_experience.items():
        fixed_user = user_aliases.get(user, user)
        if fixed_user in result:
            for strategy, commit_time in data["test_strategies"].items():
                if strategy in result[fixed_user]["test_strategies"]:
                    result[fixed_user]["test_strategies"][strategy] = min(
                        result[fixed_user]["test_strategies"][strategy], commit_time
                    )
                else:
                    result[fixed_user]["test_strategies"][strategy] = commit_time
        else:
            result[fixed_user] = data

    return result


def get_user_data(repos_root, repo_list, repo_classification, user_aliases, redo):
    """Get user data from all repositories in the list."""
    user_data = dict()
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
            repo = Repo(repos_root / full_repo_name)
            classification = repo_classification[full_repo_name]
            repo_user_data, dropped_test_strategies = get_repo_user_data(
                repo, get_source_files(repo), classification
            )
            if dropped_test_strategies:
                print(
                    f"{full_repo_name} has dropped {', '.join(dropped_test_strategies)}"
                )

        repo_user_data = remove_aliases(repo_user_data, user_aliases)

        with open(user_data_analysis, "w") as json:
            dump(repo_user_data, json)

        new_users = []
        for user in repo_user_data:
            if user not in user_data:
                new_users.append(user)
        if new_users:
            print("  New users: %s" % ascii("|".join(new_users))[1:-1])

        for user, data in repo_user_data.items():
            for strategy, commit_time in data["test_strategies"].items():
                update_data(
                    user_data, user, strategy, commit_time, data["classification"],
                )

    return user_data


def analyze(
    repos_root,
    repos_stat_path,
    repo_classification_path,
    user_aliases_path,
    output_path,
    redo,
    github_user,
    github_access_token,
):
    """Analyze all repos under the given root directory."""
    with open(repos_stat_path) as json:
        repo_list = get_repos_using_a_std_framework(load(json))

    clone(
        repo_list,
        repos_root,
        github_user,
        github_access_token,
        retry=redo,
        no_zip=True,
    )

    with open(repo_classification_path) as repo_classification_json, open(
        user_aliases_path
    ) as user_aliases_json:
        user_data = get_user_data(
            repos_root,
            repo_list,
            load(repo_classification_json),
            load(user_aliases_json),
            redo,
        )
    print(f"Total number of users: {len(user_data)}")

    print("Potential aliases:")
    for user_1, user_2 in get_potential_aliases(user_data):
        print(f"{ascii(user_1)[1:-1]} is similar to {ascii(user_2)[1:-1]}")

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
        "user_aliases_path",
        help="JSON file containing user alias to true name mapping",
        type=Path,
    )

    parser.add_argument(
        "output_path",
        help="Path to JSON file listing the analysis result for all repositories",
        type=Path,
    )
    parser.add_argument(
        "github_user", help="Github user name to use with the Github API"
    )
    parser.add_argument(
        "github_access_token", help="Github access token to use with the Github API"
    )

    parser.add_argument(
        "--redo",
        action="store_true",
        default=False,
        help="Redo user analysis even if previous results exists",
    )
    args = parser.parse_args()

    analyze(
        args.repos_root,
        args.repos_stat_path,
        args.repo_classification_path,
        args.user_aliases_path,
        args.output_path,
        args.redo,
        args.github_user,
        args.github_access_token,
    )


if __name__ == "__main__":
    main()
