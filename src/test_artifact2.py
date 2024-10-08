from artifact import ArtifactManager, Artifact

def test_print_detection_times():

    artifact_manager = ArtifactManager("../data/artifacts.csv")

    for artifact_id in artifact_manager.data["Artifact ID"]:
        artifact = Artifact(artifact_id)

        print(f"Artifact ID: {artifact.id}")

        for buoy_id, detection_time in artifact.times.items():
            print(f"    {buoy_id}: {detection_time}")

if __name__ == "__main__":
    test_print_detection_times()
