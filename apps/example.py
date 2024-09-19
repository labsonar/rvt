"""
Example Application
===================

This file provides an example application demonstrating the usage of the `example` module
from the `template_library` package.
"""

import os
import argparse
import project_name.example as lib

def main():
    """
    Main function to demonstrate the usage of the `sample_function` and sleep functionality.

    Args:
        sleep (float): The duration to sleep in seconds.
    """
    print(lib.sample_function())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'{os.path.basename(__file__)} application')
    args = parser.parse_args()
    main()
