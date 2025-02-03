"Script for running experiments on Z-Score anomaly detector."

import subprocess
from itertools import product

window_sizes = [2000]
overlaps = [0.2]
thresholds = [0.3]
wavelets = ["db4"]
levels = [1]

# Definir os arquivos utilizados (1 a 26, exceto 9, 10 e 26 - sao arquivos para teste,
#   dois EX-SUP e um GAE)
all_files = list(range(1, 27))
excluded_files = [9, 10, 26]
test_files = [f for f in all_files if f not in excluded_files]
# test_files = [21, 22, 23, 24, 25]

test_script = "test/detector_tests.py"

param_combinations = [
    (window_size, overlap, threshold, wavelet, level)
    for window_size, overlap, threshold, wavelet, level in product(window_sizes, overlaps, thresholds, wavelets, levels)
]

# if step <= window_size  # So mantem combinacoes com passo menor que a janela

for window_size, overlap, threshold, wavelet, level in param_combinations:

    cmd = [
        "python", test_script,
        "-a", "EX-SUP",
        "-b", "1", "2", "3", "4", "5",
        "-d", "3",
        "--params", str(window_size), str(overlap), str(threshold), str(wavelet), str(level),
        "--test"
    ]

    subprocess.run(cmd, check=True)
