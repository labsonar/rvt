from artifact import ArtifactManager

def artifact_test():

    artifact_manager = ArtifactManager(base_path="/home/gabriel.lisboa/Workspace/RVT/rvt/data/artifacts.csv")

    for artifact in artifact_manager:
        for buoy_id_, time in artifact:
            print(f"[{buoy_id_}: {time}]")

artifact_test()