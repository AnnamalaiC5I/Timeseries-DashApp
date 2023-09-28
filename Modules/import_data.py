import pandas as pd

class ImportData:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_csv(self) -> pd.DataFrame:
        
        assert self.file_path.endswith('.csv'), "Invalid file format"
        
        try:
            df = pd.read_csv(self.file_path)
            return df
        except Exception as e:
            print(f"Error reading CSV file: {e}")

    def read_excel(self, sheet_name: str =None) -> pd.DataFrame:
        
        assert self.file_path.endswith('.xlsx'), "Invalid file format"
        
        try:
            df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            return df
        except Exception as e:
            print(f"Error reading Excel file: {e}")

    