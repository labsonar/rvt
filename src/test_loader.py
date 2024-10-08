import datetime
from loader import DataLoader

def test_print_audio_dict():
    
    base_path = "/home/gabriel.lisboa/Workspace/RVT/Data/RVT/raw_data"
    data_loader = DataLoader(base_path)

    for buoy_id, file_list in data_loader.file_dict.items():
        print(f"Boia ID: {buoy_id}")
        for date, file in file_list:
            print(f"\tData: {date}, Arquivo: {file}")


test_print_audio_dict()
