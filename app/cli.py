"""Command line interface (CLI) equivalent to Streamlit homepage to test pipeline."""
import typing
import argparse

import lps_rvt.types as rvt
import lps_rvt.dataloader as rvt_loader
import lps_rvt.pipeline as rvt_pipeline
import lps_rvt.preprocessing as rvt_preprocessing
import lps_rvt.detector as rvt_detector

def add_file_options(parser: argparse.ArgumentParser):
    """ Add file selections options to argparse """

    file_group = parser.add_argument_group("File Selection",
                                           "Filter which files will be processed.")

    file_group.add_argument("-a", "--ammunition", nargs="*",
                             choices=[e.name for e in rvt.Ammunition],
                             default=[rvt.Ammunition.EXSUP.name],
                             help="Filter by ammunition types.")

    file_group.add_argument("-b", "--buoys", nargs="*", type=int,
                            help="Filter by buoy IDs.")

    file_group.add_argument("-s", "--subsets", nargs="*",
                             choices=[e.name for e in rvt.Subset],
                             default=[rvt.Subset.TRAIN.name],
                             help="Filter by subset types.")

def get_selected_files(args: argparse.Namespace) -> typing.List[int]:
    """Get selected files based on parse of arguments

    Returns:
        List[int]: File id list
    """
    # Convert arguments to appropriate types
    ammunition_types = [rvt.Ammunition[t] for t in args.ammunition] if args.ammunition else None
    buoys = args.buoys if args.buoys else None
    subsets = [rvt.Subset[s] for s in args.subsets] if args.subsets else None

    # Load files with filters
    loader = rvt_loader.DataLoader()
    return loader.get_files(ammunition_types, buoys, subsets)

def add_pipeline_options(parser: argparse.ArgumentParser) -> None:
    """
    Adds pipeline configuration options to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the options will be added.
    """
    pipeline_group = parser.add_argument_group("Pipeline Configuration", "Define the pipeline settings.")

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
    parser.add_argument("--output_dir", type=str, default="./results/cli",
                        help="Step size for analysis (samples).")
    parser.add_argument("--extensive_plot", action="store_true",
                            help="Enable logging for debugging purposes.")
    parser.add_argument("--no_plot", action="store_true",
                            help="Enable logging for debugging purposes.")


    add_pipeline_options(parser)
    add_file_options(parser)
    rvt_preprocessing.add_preprocessing_options(parser)
    rvt_detector.add_detector_options(parser)

    args = parser.parse_args()

    selected_files = get_selected_files(args)
    if not selected_files:
        print("No files selected.")
        return

    print(f"Selected files: {selected_files}")

    preprocessors = rvt_preprocessing.get_preprocessors(args)
    detectors = rvt_detector.get_detectors(args)

    pipeline = rvt_pipeline.Pipeline(preprocessors=preprocessors,
                                               detectors=detectors,
                                               sample_step=args.sample_step,
                                               tolerance_before=args.tolerance_before,
                                               tolerance_after=args.tolerance_after,
                                               debounce_steps=args.debounce_steps)
    

    if args.no_plot:
        result_dict = pipeline.apply(selected_files)

        for file_id, result in result_dict.items():
            print(file_id, ": ", str(result))

    else:
        pipeline.export(selected_files, args.output_dir, not args.extensive_plot)

if __name__ == "__main__":
    main()
