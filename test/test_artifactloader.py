"Simple test file for dataloader"
import argparse

import lps_rvt.types
import lps_rvt.dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DataLoader with filters.")
    parser.add_argument("-a","--ammunition", nargs="*",
                        choices=[e.name for e in lps_rvt.types.Ammunition],
                        help="Filter by data types.")
    parser.add_argument("-b","--buoys", nargs="*", type=int, help="Filter by buoy IDs.")
    parser.add_argument("-s","--subsets", nargs="*",
                        choices=[e.name for e in lps_rvt.types.Subset],
                        help="Filter by subset types.")
    args = parser.parse_args()

    file_types = [lps_rvt.types.Ammunition[t] for t in args.ammunition] \
                        if args.ammunition else None
    buoys = args.buoys if args.buoys else None
    subsets = [lps_rvt.types.Subset[s] for s in args.subsets] \
                        if args.subsets else None

    loader = lps_rvt.dataloader.ArtifactLoader()
    selected_files = loader.get_files(file_types, buoys, subsets)
    print("Selected files:", selected_files)
