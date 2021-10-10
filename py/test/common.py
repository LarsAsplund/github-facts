# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com

"""Test github_clone."""

import subprocess
from pathlib import Path


def create_git_repo(file_names, repo_dir):
    subprocess.call(["git", "init"], cwd=repo_dir)
    subprocess.call(["git", "branch", "--move", "main"], cwd=repo_dir)
    subprocess.call(["git", "config", "--local", "user.name", "Your Name"], cwd=repo_dir)
    subprocess.call(["git", "config", "--local", "user.email", "you@example.com"], cwd=repo_dir)
    for file_name in file_names:
        path = Path(repo_dir) / file_name
        parent = path.parent
        if parent != Path(repo_dir):
            parent.mkdir(parents=True)
        path.touch()
        subprocess.call(["git", "add", str(path)], cwd=repo_dir)
    subprocess.call(["git", "commit", "-m", "test_commit"], cwd=repo_dir)
