# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""
Test github_search.
"""
from unittest import TestCase
from unittest.mock import patch, call
from tempfile import TemporaryDirectory
from random import randint, seed
import re
from pathlib import Path
from github_search import GithubSearch, main


class Response:
    def __init__(self, respond_ok, next_link=True):
        self.ok = respond_ok
        self._data = dict(
            items=[
                dict(repository=dict(full_name="foo/bar")),
                dict(repository=dict(full_name="spam/eggs")),
            ]
        )
        if next_link:
            self.links = dict(next=dict(url="some_ur_lsize:100..120"))
        else:
            self.links = dict()

    def set_total_count(self, total_count):
        self._data["total_count"] = total_count

    def json(self):
        return self._data


class TestGithubSearch(TestCase):
    """Tests the GithubSearch class."""

    def setUp(self):
        seed(173862)

    class SideEffect:
        def __init__(self):
            self.n_files_by_size = [0]
            for _size in range(1, 10000):
                self.n_files_by_size.append(randint(0, 10))

            self.n_files_by_size[9990] = 1013
            self._range_re = re.compile(r"size:(?P<start>\d+)\.\.(?P<stop>\d+)")
            self._call_count = 0

        def __call__(self, url, auth):
            self._call_count += 1
            match = self._range_re.search(url)
            start = int(match.group("start"))
            stop = min(9999, int(match.group("stop")))
            total_count = sum(self.n_files_by_size[start : stop + 1])

            resp = Response(True, next_link=self._call_count <= 1)
            resp.set_total_count(total_count)

            return resp

    @patch("github_search.get")
    def test_passing_request(self, get_mock):
        get_mock.return_value.ok = True
        search = GithubSearch("user", "access_token")
        self.assertTrue(search.request("url").ok)
        get_mock.assert_called_once_with(
            "url&per_page=100", auth=("user", "access_token")
        )

    @patch("github_search.get")
    def test_default_page_size(self, get_mock):
        get_mock.return_value.ok = True
        search = GithubSearch("user", "access_token")
        self.assertTrue(search.request("url", None).ok)
        get_mock.assert_called_once_with("url", auth=("user", "access_token"))

    @patch("github_search.get")
    def test_failing_request(self, get_mock):
        get_mock.return_value.ok = False
        search = GithubSearch("user", "access_token", n_attempts=1)
        self.assertFalse(search.request("url").ok)
        get_mock.assert_called_once_with(
            "url&per_page=100", auth=("user", "access_token")
        )

    @patch("github_search.get")
    def test_failing_with_exception(self, get_mock):
        get_mock.side_effect = Exception()
        search = GithubSearch("user", "access_token", n_attempts=1)
        self.assertIsNone(search.request("url"))
        get_mock.assert_called_once_with(
            "url&per_page=100", auth=("user", "access_token")
        )

    @patch("github_search.sleep")
    @patch("github_search.time")
    @patch("github_search.get")
    def test_passing_after_retry(self, get_mock, time_mock, sleep_mock):
        get_mock.side_effect = [Response(False), Response(True)]
        time_mock.side_effect = [10, 17, 75]
        search = GithubSearch("user", "access_token", n_attempts=2)
        resp = search.request("url", 50)
        self.assertTrue(resp.ok)
        sleep_mock.assert_called_once_with(58)
        get_mock.assert_has_calls(
            [call("url&per_page=50", auth=("user", "access_token"))] * 2
        )

    @patch("github_search.sleep")
    @patch("github_search.time")
    @patch("github_search.get")
    def test_failing_after_retry(self, get_mock, time_mock, sleep_mock):
        get_mock.side_effect = [Response(False), Response(False)]
        time_mock.side_effect = [10, 17, 75]
        search = GithubSearch("user", "access_token", n_attempts=2)
        resp = search.request("url")
        self.assertFalse(resp.ok)
        sleep_mock.assert_called_once_with(58)
        get_mock.assert_has_calls(
            [call("url&per_page=100", auth=("user", "access_token"))] * 2
        )

    @patch("github_search.sleep")
    @patch("github_search.get")
    def test_finding_a_search_subset_returning_a_non_cropped_search_result(
        self, get_mock, _sleep_mock
    ):

        for _ in range(5):
            side_effect = self.SideEffect()
            get_mock.side_effect = side_effect

            search = GithubSearch("user", "access_token")
            min_size = randint(1, 10000)
            resp, new_min_size, search_url = search._get_file_set("url", min_size)

            total_count = sum(side_effect.n_files_by_size[min_size:new_min_size])
            self.assertEqual(resp.json()["total_count"], total_count)
            self.assertTrue(total_count <= 1000)
            self.assertEqual(search_url, f"url+size:{min_size}..{new_min_size - 1}")

    def test_that_max_searchable_file_size_is_handled(self):
        search = GithubSearch("user", "access_token")
        resp, _, _ = search._get_file_set("url", 384 * 1024)
        self.assertIsNone(resp)

    @patch("github_search.sleep")
    @patch("github_search.get")
    def test_that_a_file_size_with_more_than_maximum_search_hits_is_accepted(
        self, get_mock, _sleep_mock
    ):
        side_effect = self.SideEffect()
        get_mock.side_effect = side_effect
        search = GithubSearch("user", "access_token")
        resp, _, _ = search._get_file_set("url", 9990)
        self.assertEqual(resp.json()["total_count"], 1013)

    def test_reading_backup(self):
        with TemporaryDirectory() as report_path:
            search = GithubSearch("user", "access_token")
            backuped_repos, max_file_size = search._get_backup(report_path)
            self.assertFalse(backuped_repos)
            self.assertFalse(max_file_size)

            with open(Path(report_path) / "vhdl_0_97.txt", "w") as vhdl_0_97, open(
                Path(report_path) / "vhdl_98_1002.txt", "w"
            ) as vhdl_98_1002:
                vhdl_0_97.write("repo_a\nrepo_b\nrepo_c\n")
                vhdl_98_1002.write("repo_a\nrepo_b\nrepo_c\nrepo_d\n")

            backuped_repos, max_file_size = search._get_backup(report_path)
            self.assertIn("repo_a", backuped_repos)
            self.assertIn("repo_b", backuped_repos)
            self.assertIn("repo_c", backuped_repos)
            self.assertIn("repo_d", backuped_repos)
            self.assertEqual(max_file_size, 1002)

    @patch("github_search.sleep")
    @patch("github_search.get")
    @patch("github_search.GithubSearch._get_file_set")
    def test_end_to_end_searching(self, get_file_set_mock, get_mock, _sleep_mock):
        get_file_set_mock.side_effect = [
            (Response(True), 121, "some_ur_lsize:1..120"),
            (Response(True), 141, "some_ur_lsize:121..140"),
            (None, -1, ""),
        ]
        side_effect = self.SideEffect()
        get_mock.side_effect = side_effect
        search = GithubSearch("user", "access_token")
        with TemporaryDirectory() as report_path:
            search.search("vhdl", "end", Path(report_path) / "report")
            for file_name in ["vhdl_1_120.txt", "vhdl_121_140.txt"]:
                self.assertTrue((Path(report_path) / "report" / file_name).exists())
                with open(Path(report_path) / "report" / file_name) as fptr:
                    txt = fptr.read()
                    self.assertEqual(txt.count("\n"), 2)
                    self.assertIn("spam/eggs\n", txt)
                    self.assertIn("foo/bar\n", txt)

    @staticmethod
    @patch(
        "sys.argv",
        ["github_search.py", "vhdl", "end", "report_path", "UserName", "access_token",],
    )
    @patch("github_search.GithubSearch", autospec=True)
    def test_cli(github_search_mock):
        main()
        github_search_mock.assert_called_once_with("UserName", "access_token")
        github_search_mock.return_value.search.assert_called_once_with(
            "vhdl", "end", Path("report_path")
        )
