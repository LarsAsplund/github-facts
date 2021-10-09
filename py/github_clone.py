# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Script for cloning Gthub repositories."""

import shutil
import stat
import zipfile
import argparse
from pathlib import Path
from os import chmod, walk
from subprocess import call
from json import load, dump
from github_search import GithubSearch

BASIC_JSON_VERSION = 1


def rmtree(path):
    """
    Remove the given path recursively.

    :note: we use shutil rmtree but adjust its behaviour to see whether files that
        couldn't be deleted are read-only. Windows will not remove them in that case.
    """

    def onerror(func, path, exc_info):
        # Is the error an access error ?
        chmod(path, stat.S_IWUSR)
        try:
            func(path)  # Will scream if still not possible to delete.
        except Exception as ex:
            print(ex)

    try:
        return shutil.rmtree(path, False, onerror)
    except Exception as ex:
        print(ex)


def clone_repo(repo_dir, repo_url, sparse=True):
    """
    Make an optionally sparse clone of a repo just containing VHDL, SystemVerilog and Python files.

    Didn't find a way to do that with the Github module.
    """
    errors = False
    errors |= call(["git", "init"], cwd=repo_dir) != 0
    errors |= (
        call(["git", "remote", "add", "-f", "origin", "%s" % repo_url], cwd=repo_dir)
        != 0
    )

    if sparse:
        errors |= (
            call(["git", "config", "core.sparseCheckout", "true"], cwd=repo_dir) != 0
        )

        with open(repo_dir / ".git" / "info" / "sparse-checkout", "w") as fptr:
            fptr.write("*.vhd\n")
            fptr.write("*.vhdl\n")
            fptr.write("*.py\n")
            fptr.write("*.sv\n")

    errors |= call(["git", "pull", "origin", "main"], cwd=repo_dir) != 0

    return not errors


def get_basic_data(
    github_repo_full_name, basic_data_path, github_user, github_access_token
):
    """Get basic data about a Github repository."""
    search = GithubSearch(github_user, github_access_token, n_attempts=2)
    update_basic_data = True
    basic_data = None
    if basic_data_path.exists():
        with open(basic_data_path) as json:
            basic_data = load(json)
            if basic_data:
                update_basic_data = False

    if update_basic_data:
        response = search.request(
            "https://api.github.com/repos/%s" % github_repo_full_name,
            page_size=None,
        )
        if not response.ok:
            basic_data = None
            print("Failed getting basic data")
        else:
            basic_data = response.json()

    return basic_data


def zip_repo(repo_dir):
    """Zip a repository after removing .git."""
    rmtree(repo_dir / ".git")

    user_dir = repo_dir.parent.resolve()
    github_repo_name = repo_dir.name
    ziph = zipfile.ZipFile(
        user_dir / f"{github_repo_name}.zip",
        "w",
        zipfile.ZIP_DEFLATED,
    )
    for root, _, files in walk(repo_dir):
        for file in files:
            ziph.write(Path(root) / file)

    ziph.close()


def clone(
    repos,
    repos_root,
    github_user,
    github_access_token,
    retry,
    no_zip=False,
):
    """Clones all repos in the provided list."""

    if not repos_root.exists():
        repos_root.mkdir()

    n_repos = 0
    for github_repo_full_name in repos:
        n_repos += 1

        exception_reason = None
        try:
            github_repo_user = github_repo_full_name.split("/")[0]
            github_repo_name = github_repo_full_name.split("/")[1]

            user_dir = repos_root / github_repo_user

            if not user_dir.exists():
                user_dir.mkdir()

            if (user_dir / f"{github_repo_name}.failed").exists() and not retry:
                print("%d. %s already failed" % (n_repos, github_repo_full_name))
                continue

            repo_dir = user_dir / github_repo_name
            cloned_repo = False
            updated_repo = False
            already_cloned = False
            if not no_zip and (user_dir / f"{github_repo_name}.zip").exists():
                print("%d. %s already cloned" % (n_repos, github_repo_full_name))
                already_cloned = True
            elif (repo_dir / ".git").exists() and retry:
                print("%d. Updating %s" % (n_repos, github_repo_full_name))
                updated_repo = (
                    call(["git", "pull", "origin", "master"], cwd=repo_dir) == 0
                )
            else:
                if not repo_dir.exists():
                    repo_dir.mkdir()

                print("%d. Cloning %s" % (n_repos, github_repo_full_name))
                clone_url = f"https://github.com/{github_repo_full_name}.git"
                cloned_repo = clone_repo(repo_dir, clone_url)

        except Exception as ex:
            exception_reason = str(ex)
            print(exception_reason)
            print("Cloning failed")

        if cloned_repo or updated_repo or already_cloned:
            basic_data_path = (
                user_dir / f"{github_repo_name}.basic.{BASIC_JSON_VERSION}.json"
            )

            basic_data = get_basic_data(
                github_repo_full_name,
                basic_data_path,
                github_user,
                github_access_token,
            )

            if basic_data:
                with open(basic_data_path, "w") as json:
                    dump(basic_data, json)

        if not no_zip:
            if cloned_repo:
                zip_repo(repo_dir)
            elif not already_cloned:
                with open(user_dir / f"{github_repo_name}.failed", "w") as fptr:
                    if exception_reason:
                        fptr.write(exception_reason)

            if repo_dir.exists():
                rmtree(str(repo_dir))


def clone_from_file_based_list(
    repo_list_path,
    repos_root,
    github_user,
    github_access_token,
    retry,
    no_zip=False,
):
    """Clones all repos in the provided file-based list."""

    with open(repo_list_path) as repo_list:
        repos = [line[:-1] for line in repo_list]
        clone(repos, repos_root, github_user, github_access_token, retry, no_zip)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clone repos from a list. Only checkout VHDL, SystemVerilog, and Python files."
        "Optionally zip to save space (default)."
    )
    parser.add_argument("repo_list", help="File containing a list of repos", type=Path)
    parser.add_argument(
        "repos_root", help="Directory where the repos will be cloned", type=Path
    )
    parser.add_argument(
        "github_user", help="Github user name to use with the Github API"
    )
    parser.add_argument(
        "github_access_token", help="Github access token to use with the Github API"
    )
    parser.add_argument(
        "--retry",
        help="Retry cloning repos that previously failed",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--no-zip", help="Don't zip the clone", action="store_true", default=False
    )

    args = parser.parse_args()

    clone_from_file_based_list(
        args.repo_list,
        args.repos_root,
        args.github_user,
        args.github_access_token,
        args.retry,
        args.no_zip,
    )


if __name__ == "__main__":
    main()
