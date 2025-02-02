"""Simple Pipeline test"""
import numpy as np
import lps_rvt.pipeline as rvt

if __name__ == "__main__":
    # Casos de teste
    test_cases = [
        {
            "expected_detections": [100, 200, 300],
            "detect_samples": [95, 105, 205, 400],
            "tolerance_before": 5,
            "tolerance_after": 5,
            "expected_result": [2, 1, 1],  # 2 corretas (100, 200), 1 falso positivo (400), 1 falso negativo (300)
        },
        {
            "expected_detections": [50, 150, 250],
            "detect_samples": [45, 159, 260],
            "tolerance_before": 5,
            "tolerance_after": 10,
            "expected_result": [3, 0, 0],  # 3 corretas, 0 falsos positivos, 0 falsos negativos
        },
        {
            "expected_detections": [500, 600, 700],
            "detect_samples": [400, 800, 900],
            "tolerance_before": 10,
            "tolerance_after": 10,
            "expected_result": [0, 3, 3],  # Nenhuma correta, 3 falsos positivos, 3 falsos negativos
        }
    ]

    # Executando os testes
    for i, case in enumerate(test_cases):
        result = rvt.Detector.evaluate(
            case["expected_detections"],
            range(100),
            case["detect_samples"],
            case["tolerance_before"],
            case["tolerance_after"]
        )
        assert np.array_equal(result, np.array(case["expected_result"])), f"Teste {i+1} falhou! {result} != {case['expected_result']}"
        print(f"Teste {i+1} passou! Resultado: {result}")
