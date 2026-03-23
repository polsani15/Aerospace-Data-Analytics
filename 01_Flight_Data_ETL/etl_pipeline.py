import pandas as pd
import numpy as np
import os

class FlightDataETL:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None

    def extract(self):
        """Load raw CSV data into a Pandas DataFrame."""
        if os.path.exists(self.input_file):
            self.df = pd.read_csv(self.input_file)
            print(f"[SUCCESS] Extracted {len(self.df)} rows from {self.input_file}")
        else:
            print(f"[ERROR] {self.input_file} not found.")

    def transform(self):
        """Clean data: Handle missing values and perform feature engineering."""
        print("[PROCESS] Transforming data...")
        
        # 1. Linear interpolation for missing sensor pings (Common in Flight Test)
        self.df = self.df.interpolate(method='linear')
        
        # 2. Feature Engineering: Calculate Mach Number (Sea Level Speed of Sound ~340 m/s)
        self.df['Mach'] = self.df['Velocity'] / 340.0
        
        # 3. Data Labeling: Flag High Alpha (AoA) excursions for structural review
        self.df['Load_Warning'] = self.df['Alpha'].apply(lambda x: 1 if x > 15 else 0)

    def load(self, output_name="processed_flight_data.csv"):
        """Export the validated and cleaned dataset."""
        self.df.to_csv(output_name, index=False)
        print(f"[SUCCESS] Processed data saved to {output_name}")

if __name__ == "__main__":
    # Create a small mock dataset automatically for testing
    mock_data = {
        'Timestamp': np.arange(0, 10, 1),
        'Alpha': [2.1, 2.3, np.nan, 2.8, 16.5, 3.1, np.nan, 2.9, 2.5, 2.2],
        'Velocity': [150, 152, 155, 158, 160, 162, 165, 168, 170, 172]
    }
    pd.DataFrame(mock_data).to_csv("raw_test_data.csv", index=False)

    # Execute the Pipeline
    pipeline = FlightDataETL("raw_test_data.csv")
    pipeline.extract()
    pipeline.transform()
    pipeline.load()