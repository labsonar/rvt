"""Simple test of evaluating function.
"""
import numpy as np
import lps_rvt.pipeline as rvt

def main() -> None:
    """Main function to test of evaluating function. """
    # Casos de teste
    test_cases = [
        {
            "expected_detections": [100, 200, 300],
            "expected_rebounds": [230],
            "detect_samples": [95, 105, 205, 400],
            "tolerance_before": 5,
            "tolerance_after": 5,
            "expected_result": [1, 1, 2],
                # 1 false positive (400)
                # 1 false negative (300)
                # 2 true positives (100, 200)
        },
        {
            "expected_detections": [50, 150, 250],
            "expected_rebounds": [180],
            "detect_samples": [45, 159, 260],
            "tolerance_before": 5,
            "tolerance_after": 10,
            "expected_result": [0, 0, 3],
                # 0 false positives
                # 0 false negatives
                # 3 true positives
        },
        {
            "expected_detections": [500, 600, 700],
            "expected_rebounds": [790],
            "detect_samples": [400, 800, 900],
            "tolerance_before": 10,
            "tolerance_after": 10,
            "expected_result": [2, 3, 0],
                # 2 false positives
                # 3 false negatives
                # 0 true positives
        }
    ]

    # run and check the tests
    for i, case in enumerate(test_cases):
        result = rvt.Detector.evaluate(
                case["expected_detections"],
                case["expected_rebounds"],
                range(100),
                case["detect_samples"],
                case["tolerance_before"],
                case["tolerance_after"]
            ).ravel().tolist()
        result.pop(0)
        assert np.array_equal(result, np.array(case["expected_result"])), \
            f"Test {i+1}: failed. Result[{result}] != Expected Result[{case['expected_result']}]"
        print(f"Test {i+1}: ok. Result[{result}]")

if __name__ == "__main__":
    main()
