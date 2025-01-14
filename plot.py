""" Module made to plot data. """

import os
import shutil
import argparse
from datetime import timedelta

import src.loader as loader
import src.artifact as artifact
import src.analysis as analysis

if os.path.exists("Analysis"):
    shutil.rmtree("Analysis")
os.mkdir("Analysis")

loader_ = loader.DataLoader()
manager_ = artifact.ArtifactManager()

parser = argparse.ArgumentParser(description="App made to plot data.")

parser.add_argument("-s", "--start", type=float, \
    default=2.5,
    help="Start shift in seconds of event to be analysed. Default to 2.5 seconds.")

parser.add_argument("-e", "--end", type=float, \
    default=2.5,
    help="End shift in seconds of event to be analysed. Default to 2.5 seconds.")

parser.add_argument("-ba", "--background", type=float, \
    default=0.5, \
    help="Background shift in seconds of event to be analysed.")

parser.add_argument("-bu", "--buoys", type=int, nargs="*", \
    default=manager_.get_buoys(),
    help="Buoys to be analysed: 1, 2, 3, 4, 5. Default to plot all")

parser.add_argument("-p", "--plots", type=str, nargs="*", \
    default=['fft', 'psd', 'lofar', 'time'],
    help="Plots to be made: 'fft', 'psd', 'lofar', 'time'. Default to plot all.")

parser.add_argument("-t", "--type", type=str, nargs="*", \
    default=manager_.get_types(),
    help=f"Artifact types to be ploted: {manager_.get_types()}. Default to plot all.")

parser.add_argument("-i", "--id", type=int, nargs="*", \
    default=None,
    help="Específic artifact to be ploted. Default to None.")

args = parser.parse_args()

artifacts = []
for type_ in args.type:
    for id_ in manager_.id_from_type(type_):
        artifacts.append(id_)

    if args.id:
        for id_ in args.id:
            if not id_ in artifacts:
                artifacts.append(id_)

for artifact_id in artifacts:
    print(f"Artifact id: {artifact_id}")
    for buoy_id in manager_[artifact_id]:
        time = manager_.get_time(artifact_id, buoy_id)
        if buoy_id in args.buoys:
            print(f"\tBuoy {buoy_id}:")

            start = time-timedelta(seconds=args.start)
            end = time+timedelta(seconds=args.end)

            for plot in args.plots:

                print("\t\tArtifact: ", end="")
                analysis.plot(buoy_id, artifact_id, plot, \
                    start, end, time, "Analysis/Artifact")

            if args.background:

                background = time-timedelta(seconds=args.background)

                for plot in args.plots:

                    print("\t\tBackground: ", end="")
                    analysis.plot(buoy_id, artifact_id, plot, \
                        start, background, time, "Analysis/Background")

                for plot in args.plots:

                    if plot == "lofar":
                        continue

                    if plot == "time":
                        continue

                    print("\t\tArtifact x Background: ", end="")
                    analysis.plot_artifact_vs_bkg(buoy_id, artifact_id, plot,\
                        start, background, end, time, "Analysis/Artifact x Background")
