"""Simple test to open each of the test files to check for problems.
"""
import lps_rvt.dataloader as rvt

def main() -> None:
    """Main function to load and display test files. """
    loader = rvt.DataLoader()

    for file in loader.get_files():
        print(f"Processing file: {file}")
        fs, data = loader.get_data(file)
        print(f"Sampling Frequency (fs): {fs}, Data Shape: {data.shape}")

if __name__ == "__main__":
    main()
