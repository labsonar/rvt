"Script for running experiments on Z-Score anomaly detector."

import subprocess
from itertools import product

window_sizes = [4000]
steps = [0.7]
thresholds = [0.2]

# Definir os arquivos utilizados (1 a 26, exceto 9, 10 e 26 - sao arquivos para teste,
#   dois EX-SUP e um GAE)
all_files = list(range(1, 27))
excluded_files = [9, 10, 26]
test_files = [f for f in all_files if f not in excluded_files]
# test_files = [21, 22, 23, 24, 25]

test_script = "test/detector_tests.py"

param_combinations = [
    (window_size, step, threshold)
    for window_size, step, threshold in product(window_sizes, steps, thresholds)
    if step <= window_size  # So mantem combinacoes com passo menor que a janela
]

for window_size, step, threshold in param_combinations:

    cmd = [
        "python", test_script,
        "-f", *map(str, test_files),
        "-a", "EX-SUP",
        "-b", "1", "2", "3", "4", "5",
        "-d", "2",
        "--params", str(window_size), str(step), str(threshold),
        "--test"
    ]

    subprocess.run(cmd, check=True)
