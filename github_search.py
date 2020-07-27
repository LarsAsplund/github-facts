# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Script for searching Github."""

from time import sleep, time

from os.path import splitext
from os import walk
from pathlib import Path
import argparse
from requests import get


class GithubSearch:
    """Manage the search for all relevant repositories."""

    def __init__(self, github_user, github_access_token, n_attempts=1e9):
        self._auth = (github_user, github_access_token)
        self._first_request_at = None
        self._n_attempts = n_attempts
        self._last_range_size = 1

    def request(self, url, page_size=100):
        """Perform a search towards the Github API."""
        if not self._first_request_at:
            self._first_request_at = time()

        n_attempts = 0
        valid_response = False
        response = None
        full_url = ""
        while not valid_response and (n_attempts < self._n_attempts):
            n_attempts += 1

            # There is a rate limit for the number of searches that can be performed every minute
            # Wait and retry if that limit is hit
            # Have some margin to the 60 seconds and make sure we never hit negative times
            if n_attempts > 1:
                wait_for = max(65 - int(time() - self._first_request_at), 5)
                print(f"Waiting {wait_for} seconds for access.")
                sleep(wait_for)
                self._first_request_at = time()

            try:
                if page_size:
                    full_url = url + f"&per_page={page_size}"
                else:
                    full_url = url

                response = get(full_url, auth=self._auth)
                valid_response = response.ok
            except:
                valid_response = False

        return response

    def _get_file_set(self, url, min_size):
        """
        Get a search query returning a file set not limited by Github.

        Github has a maximum of 1000 search results for every search. The number of GitHub API
        interactions per minute is also limited and that makes large GitHub searches slow.
        There are two basic types of interactions:

        1. Making a search, for example a search for files in the 1000 to 1010 bytes range, to see
        if the search result has less than 1000 files.
        2. Downloading the search result takes one interaction for every 100 files in the search
        result

        It's better two have few large file sets than many small since that results in fewer
        download interactions. Downloading 4 file sets with 250 files in each takes 12 interactions
        while downloading a single file set with 1000 files takes 10 interactions. However, it
        takes more searches to find a large file set. For example, if you know that the a - b range
        contains 250 files you can decide to download that with three interactions or try to see if
        a wider range a - c, where c is estimated, results in a larger file set. The risk is
        that a - c contains more than 1000 files such that the extra search was a waste.
        This function uses a heuristic to make this decision.
        """
        # Github has a limit on max 1000 results for a search query
        max_num_of_results = 1000

        # Github doesn't allow searches for files larger than 324 kB
        max_file_size = 384 * 1024

        def files_in_range(start, stop):
            resp = self.request(url + f"+size:{start}..{stop}")

            return resp.json()["total_count"], resp

        def good_enough_range(stop, n_searches):
            return stop["n_files"] >= max_num_of_results / (1 + 1 / n_searches)

        def get_new_stop(stop1, stop2):
            if stop1["n_files"] == stop2["n_files"]:
                if stop1["n_files"] <= max_num_of_results:
                    stop = max(stop1["stop"], stop2["stop"]) + 2 * abs(
                        stop1["stop"] - stop2["stop"]
                    )
                else:
                    stop = min(stop1["stop"], stop2["stop"]) - 2 * abs(
                        stop1["stop"] - stop2["stop"]
                    )

            else:
                stop = round(
                    (
                        stop1["stop"] * (max_num_of_results - stop2["n_files"])
                        - stop2["stop"] * (max_num_of_results - stop1["n_files"])
                    )
                    / (stop1["n_files"] - stop2["n_files"])
                )

            return min(max(min_size, stop), max_file_size)

        if min_size >= max_file_size:
            return (None, -1, url + "")

        start = min_size
        history = [dict(stop=start - 1, n_files=0, resp=None)]
        stop = history[-1]["stop"] + max(1, self._last_range_size - 1)
        n_searches = 1
        n_files, resp = files_in_range(min_size, stop)
        history.append(dict(stop=stop, n_files=n_files, resp=resp))

        while not good_enough_range(history[-1], n_searches) and (
            history[-1]["n_files"] <= max_num_of_results
        ):
            stop = get_new_stop(history[-1], history[-2])
            if stop == history[-1]:
                stop += history[-1] - history[-2]

            n_files, resp = files_in_range(min_size, stop)
            n_searches += 1
            history.append(dict(stop=stop, n_files=n_files, resp=resp))

        if history[-1]["n_files"] > max_num_of_results:
            best_below = history[-2]
            best_above = history[-1]

            improving = True
            while not good_enough_range(best_below, n_searches) and improving:
                stop = get_new_stop(best_below, best_above)
                n_files, resp = files_in_range(min_size, stop)
                n_searches += 1

                improving = False
                if best_below["n_files"] < n_files <= max_num_of_results:
                    best_below = dict(stop=stop, n_files=n_files, resp=resp)
                    improving = True

                if max_num_of_results < n_files < best_above["n_files"]:
                    best_above = dict(stop=stop, n_files=n_files, resp=resp)
                    improving = True

            first_candidate = best_below
            second_candidate = best_above
        else:
            first_candidate = history[-1]
            second_candidate = history[-2]

        if (first_candidate["n_files"] <= max_num_of_results) and (
            first_candidate["stop"] != start - 1
        ):
            stop = first_candidate
        else:
            stop = second_candidate

        print(f"Range {start} - {stop['stop']} bytes contains {stop['n_files']} files")

        self._last_range_size = stop["stop"] - start + 1

        return (
            stop["resp"],
            stop["stop"] + 1,
            url + f"+size:{start}..{stop['stop']}",
        )

    def _get_repos(self, language, query, report_path, repo_backup, min_file_size=0):
        """Get repos by adding to backed up repos and starting the search from a min file size."""
        n_files = 0
        old_min_file_size = min_file_size
        base_url = f"https://api.github.com/search/code?q={query}+language:{language}"
        (response, min_file_size, url) = self._get_file_set(base_url, min_file_size,)
        while response:
            while url:
                files = response.json()["items"]

                n_files += len(files)
                for file in files:
                    repo_backup.add(file["repository"]["full_name"])

                print(f"Files = {n_files}, repos = {len(repo_backup)}")

                url = (
                    None
                    if not response.links.get("next")
                    else response.links.get("next").get("url")
                )
                if url:
                    response = self.request(url)

            with open(
                report_path / f"{language}_{old_min_file_size}_{min_file_size - 1}.txt",
                "w",
            ) as fptr:
                for repo in repo_backup:
                    fptr.write(f"{repo}\n")

            old_min_file_size = min_file_size
            (response, min_file_size, url) = self._get_file_set(base_url, min_file_size)

        return repo_backup

    @staticmethod
    def _get_backup(report_path):
        """
        Get the latest file of backuped search results (found repos).

        The function also returns the maximum file size in these repos which
        indicates the next file size range to search for doing additional searches
        """
        max_file_size = 0
        latest_backup = None
        backuped_repos = set()
        for root, _, files in walk(report_path):
            for file in files:
                upper_limit = int(splitext(file)[0].split("_")[-1])
                if upper_limit > max_file_size:
                    max_file_size = upper_limit
                    latest_backup = Path(root) / file

        if latest_backup:
            with open(latest_backup) as fptr:
                for line in fptr:
                    backuped_repos.add(line[:-1])

        return backuped_repos, max_file_size

    def search(self, language, query, report_path):
        """Search for repositories containing files in the languages specified."""
        if not report_path.exists():
            report_path.mkdir()

        backuped_repos, max_file_size = self._get_backup(report_path)
        self._get_repos(language, query, report_path, backuped_repos, max_file_size + 1)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search for Github repos containing specific file types"
    )
    parser.add_argument(
        "language", help="Language accepted by Github's API, for example vhdl"
    )
    parser.add_argument(
        "query", help="Code phrase to search for, for example vunit_lib"
    )
    parser.add_argument(
        "report_path",
        help="Directory where backups and final result will be stored",
        type=Path,
    )
    parser.add_argument(
        "github_user", help="Github user name to use with the Github API"
    )
    parser.add_argument(
        "github_access_token", help="Github access token to use with the Github API"
    )

    args = parser.parse_args()

    GithubSearch(args.github_user, args.github_access_token).search(
        args.language, args.query, args.report_path
    )


if __name__ == "__main__":
    main()
