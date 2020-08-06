# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2020, Lars Asplund lars.anders.asplund@gmail.com
"""Script for visualizing the derived GitHub statistics."""

import argparse
from json import load
from datetime import datetime, date, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from IPython.display import HTML
from moviepy.editor import VideoFileClip
from moviepy.video.fx.resize import resize
from visualize_test_strategy import (
    plot_euler_diagram,
    fix_framework_name_casing,
    date_range,
    create_accumulated_histogram_from_dict_with_dates,
    extend_accumulated_histogram,
    make_graph_over_time,
)


def create_euler_diagrams(user_data, output_dir):
    """Create an Euler diagram for the user data and store in the provided output directory."""
    for classification in ["total", "professional", "academic", "unknown"]:
        print(f"Generating Euler diagram for {classification} users")
        distribution = dict()
        for data in user_data.values():
            if classification not in ("total", data["classification"]):
                continue
            strategy_names = fix_framework_name_casing(
                list(data["test_strategies"].keys())
            )
            strategy_names.sort()
            group_name = "&".join(strategy_names)

            if group_name not in distribution:
                distribution[group_name] = 1
            else:
                distribution[group_name] += 1

        plot_euler_diagram(distribution, output_dir / f"{classification}_user.svg")


def create_plots(user_data, output_dir):
    """Create a plot over time for the user data and store in the provided output directory."""

    def make_get_date(classification, framework):
        def get_date(data):
            if classification not in ("total", data["classification"]):
                return None

            if (framework not in data["test_strategies"]) and (framework != "std"):
                return None

            if framework == "std":
                dates = [
                    date.fromtimestamp(timestamp)
                    for timestamp in data["test_strategies"].values()
                ]
            else:
                dates = [date.fromtimestamp(data["test_strategies"][framework])]

            return dates

        return get_date

    accumulated_histograms = dict()
    for classification in ["professional", "academic", "unknown", "total"]:
        print(
            f"Generating plot for number of {classification} framework users over time"
        )

        accumulated_histogram = dict()
        first_date = dict()
        last_date = dict()
        for framework in ["vunit", "osvvm", "uvvm", "uvm", "cocotb", "std"]:

            (
                accumulated_histogram[framework],
                first_date[framework],
                last_date[framework],
            ) = create_accumulated_histogram_from_dict_with_dates(
                user_data, make_get_date(classification, framework)
            )

        global_first_date = min(first_date.values()) - timedelta(days=1)
        global_last_date = max(last_date.values())
        for framework in accumulated_histogram:
            accumulated_histogram[framework] = extend_accumulated_histogram(
                accumulated_histogram[framework],
                first_date[framework],
                last_date[framework],
                global_first_date,
                global_last_date,
            )

        title = (
            f"Number of {classification} users over time"
            if classification != "total"
            else "Number of users over time"
        )

        accumulated_histogram_to_plot = {
            framework: data
            for framework, data in accumulated_histogram.items()
            if framework != "std"
        }

        make_graph_over_time(
            date_range(global_first_date, global_last_date),
            accumulated_histogram_to_plot,
            "Number of users",
            title,
            output_dir / f"{classification}_users_over_time.svg",
        )

        accumulated_histograms[classification] = dict(
            accumulated_histogram=accumulated_histogram,
            first_date=global_first_date,
            last_date=global_last_date,
        )

    return accumulated_histograms


def create_timeline(timeline_start_date, last_date, framework_introduction):
    """Create a sequence of dates with pauses at interesting dates."""
    frame_dates = []
    pause_dates = {value["date"] for value in framework_introduction.values()}
    pause_dates.add(last_date)

    for current_date in date_range(timeline_start_date, last_date):
        if current_date in pause_dates:
            frame_dates += [current_date] * 90
        elif (current_date - timeline_start_date).days % 5 == 0:
            frame_dates.append(current_date)

    return frame_dates


def create_bar_race(
    accumulated_histogram,
    first_date,
    last_date,
    classification,
    timeline_start_date,
    image_path,
    video_path,
):
    """Create a bar race animation for the accumulated histogram of users."""

    class BarHandler:
        """This class maintains the state and functionality used to create the bar race."""

        def __init__(
            self,
            accumulated_histogram,
            first_date,
            last_date,
            classification,
            image_path,
            video_path,
            frame_dates,
            introduction,
        ):
            self._accumulated_histogram = accumulated_histogram
            self._accumulated_histogram_to_plot = {
                framework: data
                for framework, data in accumulated_histogram.items()
                if framework != "std"
            }
            self._first_date = first_date
            self._last_date = last_date
            self._classification = classification

            self._image_path = image_path
            self._video_path = video_path
            self._introduction = introduction
            self._logo = dict()
            for framework in self._accumulated_histogram_to_plot:
                self._logo[framework] = mpimg.imread(
                    str(self._image_path / f"{framework}_logo.png")
                )

            self._colors = {
                "VUnit": "#adb0ff",
                "OSVVM": "#90d595",
                "UVVM": "#e48381",
                "UVM": "#f7bb5f",
                "cocotb": "#eafb50",
                "": "#eafb50",
            }

            self._fig, self._ax = plt.subplots(figsize=(15, 8))

            self._ax.get_yaxis().set_ticklabels([])
            for spine in ["left", "right", "bottom"]:
                self._ax.spines[spine].set_visible(False)
            self._bbox = self._ax.get_position()
            self._data_to_figure_transform = (
                self._ax.transData + self._fig.transFigure.inverted()
            )

            self._previous_y_pos = dict()
            # Animate calls an extra time for some reason
            self._frame_dates = [frame_dates[0]] + frame_dates
            self._last_date = self._frame_dates[-1]
            self._idx = 0
            n_max = 0
            for num in self._accumulated_histogram_to_plot.values():
                n_max = max(n_max, max(num))
            self._xlim_high = n_max / 0.8

        @property
        def fig(self):
            """Return the figure used for the animation."""
            return self._fig

        def create_image(self, frame_idx):
            """Create a PNG image for the given frame."""
            self._idx = frame_idx
            self._create_frame(self._frame_dates[frame_idx], show_play_button=True)
            file_name = self._image_path / f"{self._classification}_user_bar_race.png"
            if file_name.exists():
                file_name.unlink()

            plt.savefig(file_name, format="png")

        def create(self, _):
            """
            Create a frame.
            
            This is a callback used by matplotlib when creating a frame in the animation.
            """
            if self._idx >= len(self._frame_dates):
                frame_date = self._frame_dates[-1]
            else:
                frame_date = self._frame_dates[self._idx]
            self._idx += 1

            self._create_frame(frame_date)

        def _create_frame(self, frame_date, show_play_button=False):
            def to_offset(frame_date):
                if frame_date < self._first_date:
                    return 0

                if frame_date <= self._last_date:
                    return (frame_date - self._first_date).days

                return -1

            def weight(item):
                number_of_users = item[1]
                framework = item[0]
                days_since_introduction = (
                    datetime.now().date()
                    - self._introduction[fix_framework_name_casing(framework)]["date"]
                ).days

                return number_of_users + days_since_introduction / 100000

            def get_n_users():
                n_users = dict()
                for framework, num in self._accumulated_histogram_to_plot.items():
                    n_users[fix_framework_name_casing(framework)] = num[
                        to_offset(frame_date)
                    ]

                n_users = {
                    framework: num
                    for framework, num in sorted(n_users.items(), key=weight)
                }

                return n_users

            def get_bar_positions():
                next_date = self._frame_dates[
                    min(self._idx + 1, len(self._frame_dates) - 1)
                ]

                next_n_users = dict()
                for framework, num in self._accumulated_histogram_to_plot.items():
                    next_n_users[fix_framework_name_casing(framework)] = num[
                        to_offset(next_date)
                    ]

                next_n_users = {
                    fw: num for fw, num in sorted(next_n_users.items(), key=weight)
                }

                next_y = {fw: idx for idx, fw in enumerate(next_n_users)}

                y_pos = dict()

                for i, framework in enumerate(n_users):
                    if framework in self._previous_y_pos:
                        y_pos[framework] = self._previous_y_pos[framework]
                    else:
                        y_pos[framework] = i

                    if abs(y_pos[framework] - next_y[framework]) < 0.02:
                        y_pos[framework] = next_y[framework]
                    elif y_pos[framework] < next_y[framework]:
                        y_pos[framework] += 0.02
                    elif y_pos[framework] > next_y[framework]:
                        y_pos[framework] -= 0.02

                    self._previous_y_pos[framework] = y_pos[framework]

                return y_pos

            def add_bar_text(n_users, total_n_users, y_pos):
                for framework, num in n_users.items():
                    if frame_date < self._introduction[framework]["date"]:
                        continue
                    if frame_date == self._introduction[framework]["date"]:
                        self._ax.text(
                            num,
                            y_pos[framework],
                            "   %s: %s"
                            % (
                                frame_date.strftime("%b %Y"),
                                self._introduction[framework]["text"],
                            ),
                            size=24,
                            weight=600,
                            ha="left",
                            va="center",
                        )
                    elif num > self._xlim_high / 10:
                        self._ax.text(
                            num,
                            y_pos[framework],
                            "%s  " % framework,
                            size=14,
                            weight=600,
                            ha="right",
                            va="center",
                        )
                    if (num > 0) and (
                        frame_date != self._introduction[framework]["date"]
                    ):
                        self._ax.text(
                            num,
                            y_pos[framework],
                            "  %d (%.1f%%)"
                            % (num, (num * 100) / max(1, total_n_users),),
                            size=14,
                            ha="left",
                            va="center",
                        )

            def add_user_counter(total_n_users):
                self._ax.text(
                    1,
                    0.05,
                    "%s\n%s"
                    % (frame_date.strftime("%b %Y"), "Users: %d" % total_n_users,),
                    transform=self._ax.transAxes,
                    color="grey",
                    size=46,
                    ha="right",
                )

            def configure_plot():
                self._ax.clear()
                self._ax.set_xlim(0, self._xlim_high)
                self._ax.set_ylim(-0.5, 4.5)
                self._ax.xaxis.tick_top()
                self._ax.xaxis.set_label_position("top")
                if self._classification:
                    title = "Number of %s Users" % self._classification
                else:
                    title = "Number of Users"
                self._ax.set_title(title, {"fontsize": 18})
                self._ax.get_yaxis().set_ticklabels([])
                for spine in ["left", "right", "bottom"]:
                    self._ax.spines[spine].set_visible(False)
                plt.grid(True, "major", "x", color="lightgrey")

            def add_play_button():
                play_button = mpimg.imread(str(self._image_path / "play.png"))
                self._ax.imshow(
                    play_button,
                    extent=[self._xlim_high * 0.4, self._xlim_high * 0.6, 1, 3,],
                    aspect=self._xlim_high / 10,
                    zorder=5,
                    alpha=0.9,
                )
                self._bbox = self._ax.get_position()
                self._data_to_figure_transform = (
                    self._ax.transData + self._fig.transFigure.inverted()
                )

            def add_bars(n_users, y_pos):
                container = self._ax.barh(
                    list(n_users.keys()),
                    n_users.values(),
                    color=[self._colors[framework] for framework in n_users],
                )

                size = 0.1
                for idx, framework in enumerate(n_users.keys()):
                    if frame_date < self._introduction[framework]["date"]:
                        continue

                    container[idx].set_y(
                        y_pos[framework] - container[idx].get_height() / 2
                    )
                    y_pos_transformed = (
                        self._data_to_figure_transform.transform((0, y_pos[framework]))[
                            1
                        ]
                        - 0.5 * size
                    )

                    for axis in self._fig.axes:
                        if axis.get_label() == framework.lower():
                            if axis.get_position().min[1] != y_pos_transformed:
                                axis.set_position(
                                    [
                                        self._bbox.x0 - size,
                                        y_pos_transformed,
                                        size,
                                        size,
                                    ]
                                )
                            break
                    else:
                        axis = self._fig.add_axes(
                            [self._bbox.x0 - size, y_pos_transformed, size, size],
                            label=framework.lower(),
                        )
                        axis.axison = False
                        axis.imshow(
                            self._logo[framework.lower()], interpolation="bilinear"
                        )

            configure_plot()
            n_users = get_n_users()
            y_pos = get_bar_positions()

            total_n_users = self._accumulated_histogram["std"][to_offset(frame_date)]

            add_bar_text(n_users, total_n_users, y_pos)
            add_user_counter(total_n_users)

            if show_play_button:
                add_play_button()

            add_bars(n_users, y_pos)

    def create_animation(frame_dates, framework_introduction, image_path, video_path):
        bar_handler = BarHandler(
            accumulated_histogram,
            first_date,
            last_date,
            classification,
            image_path,
            video_path,
            frame_dates,
            framework_introduction,
        )

        animator = animation.FuncAnimation(
            bar_handler.fig, bar_handler.create, frames=frame_dates
        )

        # Generate animation
        mp4_file_name = video_path / f"{classification}_user_bar_race.mp4"
        if mp4_file_name.exists():
            mp4_file_name.unlink()

        HTML(animator.save(mp4_file_name, fps=30,))

        mp4 = VideoFileClip(str(mp4_file_name), fps_source="fps")
        mp4_resized = resize(mp4, 0.5)
        gif_file_name = video_path / f"{classification}_user_bar_race.gif"
        if gif_file_name.exists():
            gif_file_name.unlink()

        mp4_resized.write_gif(gif_file_name, fps=15)

    def create_cover_image(frame_dates, framework_introduction, image_path, video_path):
        bar_handler = BarHandler(
            accumulated_histogram,
            first_date,
            last_date,
            classification,
            image_path,
            video_path,
            frame_dates,
            framework_introduction,
        )
        bar_handler.create_image(len(frame_dates) - 1)

    framework_introduction = dict(
        VUnit=dict(date=date(2014, 11, 25), text="VUnit was open sourced"),
        OSVVM=dict(date=date(2013, 5, 1), text="OSVVM was open sourced"),
        UVVM=dict(
            date=date(2013, 9, 10), text="BVUL, now part of UVVM, was open sourced"
        ),
        UVM=dict(date=date(2011, 2, 28), text="UVM 1.0 was released"),
        cocotb=dict(date=date(2013, 6, 12), text="cocotb was open sourced"),
    )

    frame_dates = create_timeline(
        timeline_start_date, last_date, framework_introduction
    )
    create_animation(frame_dates, framework_introduction, image_path, video_path)
    create_cover_image(frame_dates, framework_introduction, image_path, video_path)


def create_bar_races(
    accumulated_histograms, timeline_start_date, image_path, video_path
):
    """Create a bar race for each class of users."""
    for classification, data in accumulated_histograms.items():
        print(
            f"Generating bar race for number of {classification} framework users over time"
        )
        create_bar_race(
            data["accumulated_histogram"],
            data["first_date"],
            data["last_date"],
            classification,
            timeline_start_date,
            image_path,
            video_path,
        )


def visualize(users_stat_path, image_path, video_path):
    """Create Euler diagrams, standard plots, and bar races for the user data."""
    with open(users_stat_path) as json:
        user_data = load(json)

    create_euler_diagrams(user_data, image_path)
    accumulated_histograms = create_plots(user_data, image_path)
    create_bar_races(accumulated_histograms, date(2011, 1, 1), image_path, video_path)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualises User Test Strategy Statistics"
    )

    parser.add_argument(
        "users_stat_path",
        help="JSON file containing user data for all users",
        type=Path,
    )

    parser.add_argument("image_path", help="Directory where plots are saved", type=Path)

    parser.add_argument(
        "video_path", help="Directory where animations are saved", type=Path
    )

    args = parser.parse_args()

    visualize(args.users_stat_path, args.image_path, args.video_path)


if __name__ == "__main__":
    main()
