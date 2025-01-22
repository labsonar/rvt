"Script for running experiments on Z-Score anomaly detector."

import subprocess
from itertools import product

window_sizes = [800]
steps = [20]
thresholds = [3.0]

# Definir os arquivos utilizados (1 a 26, exceto 9, 10 e 26 - sao arquivos para teste,
#   dois EX-SUP e um GAE)
all_files = list(range(1, 27))
excluded_files = [9, 10, 26]
test_files = [f for f in all_files if f not in excluded_files]

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
        "-d", "1",
        "-p", "0",
        "--params", str(window_size), str(step), str(threshold)
    ]

    subprocess.run(cmd, check=True)
