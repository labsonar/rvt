"""
Evaluate script equivalent to CLI but run all files
without export and compiling results by ammunition type.
"""
import argparse
import numpy as np

import lps_rvt.types as rvt
import lps_rvt.dataloader as rvt_loader
import lps_rvt.pipeline as rvt_pipeline
import lps_rvt.preprocessing as rvt_preprocessing
import lps_rvt.detector as rvt_detector


def add_pipeline_options(parser: argparse.ArgumentParser) -> None:
    """
    Adds pipeline configuration options to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the options will be added.
    """
    pipeline_group = parser.add_argument_group("Pipeline Configuration",
                                               "Define the pipeline settings.")

    pipeline_group.add_argument("--sample-step", type=int, default=20,
                                help="Step size for analysis (samples).")
    pipeline_group.add_argument("--tolerance-before", type=int, default=160,
                                help="Tolerance before event (samples).")
    pipeline_group.add_argument("--tolerance-after", type=int, default=320,
                                help="Tolerance after event (samples).")
    pipeline_group.add_argument("--debounce-steps", type=int, default=50,
                                help="Debounce steps (steps).")

def main():
    """ Main CLI function"""

    parser = argparse.ArgumentParser(
        description="Test DataLoader with filters and processing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--only_artifacts", action="store_true",
                            help="Enable logging for debugging purposes.")

    add_pipeline_options(parser)
    rvt_preprocessing.add_preprocessing_options(parser)
    rvt_detector.add_detector_options(parser)

    args = parser.parse_args()

    loader=rvt_loader.ArtifactLoader() if args.only_artifacts else rvt_loader.DataLoader()
    selected_files = loader.get_files()

    preprocessors = rvt_preprocessing.get_preprocessors(args)
    detectors = rvt_detector.get_detectors(args)

    pipeline = rvt_pipeline.Pipeline(
        preprocessors=preprocessors,
        detectors=detectors,
        sample_step=args.sample_step,
        tolerance_before=args.tolerance_before,
        tolerance_after=args.tolerance_after,
        debounce_steps=args.debounce_steps,
        loader=loader)

    result_dict = pipeline.apply(selected_files)

    for file_id, result in result_dict.items():
        print(file_id, ": ", str(result))

    for ammu in rvt.Ammunition:

        tns, fps, fns, tps = [], [], [], []

        selected_files = loader.get_files(ammunition_types=[ammu])
        for file in selected_files:
            tn, fp, fn, tp = result_dict[file].get_cm().ravel()
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
            tps.append(tp)

        tns = np.sum(tns)
        fps = np.sum(fps)
        fns = np.sum(fns)
        tps = np.sum(tps)

        print(ammu, ": ", tps, "/", fns + tps, "  -> ", fps)

if __name__ == "__main__":
    main()
