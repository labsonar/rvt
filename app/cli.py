"""Command line interface (CLI) equivalent to Streamlit homepage to test pipeline."""
import os
import typing
import argparse

import lps_rvt.types as rvt
import lps_rvt.dataloader as rvt_loader
import lps_rvt.pipeline as rvt_pipeline
import lps_rvt.preprocessing as rvt_preprocessing

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

def main():
    """ Main CLI function"""

    parser = argparse.ArgumentParser(
        description="Test DataLoader with filters and processing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_file_options(parser)
    rvt_preprocessing.add_preprocessing_options(parser)

    args = parser.parse_args()

    selected_files = get_selected_files(args)
    if not selected_files:
        print("No files selected.")
        return

    print(f"Selected files: {selected_files}")

    preprocessors = rvt_preprocessing.get_preprocessors(args)

    pipeline = rvt_pipeline.ProcessingPipeline(preprocessors, [])
    result_dict = pipeline.apply(selected_files)

    output_path = "./results/cli"
    os.makedirs(output_path, exist_ok=True)

    for file_id, result in result_dict.items():
        result.final_plot(os.path.join(output_path, f"{file_id}.png"))


if __name__ == "__main__":
    main()
