import pytest
import pandas as pd
import numpy as np
from etl_pipeline import FlightDataETL

def test_mach_calculation():
    """Verify Mach = Velocity / 340"""
    data = {'Timestamp': [1], 'Alpha': [2.0], 'Velocity': [680]} 
    df = pd.DataFrame(data)
    df.to_csv("test_input.csv", index=False)
    
    pipeline = FlightDataETL("test_input.csv")
    pipeline.extract()
    pipeline.transform()
    
    assert pipeline.df['Mach'].iloc[0] == 2.0
    print("Verification: Mach Calculation PASSED")

def test_interpolation():
    """Verify NaN values are filled correctly via linear interpolation"""
    data = {'Timestamp': [1, 2, 3], 'Alpha': [10.0, np.nan, 20.0], 'Velocity': [100, 100, 100]}
    pd.DataFrame(data).to_csv("test_nan.csv", index=False)
    
    pipeline = FlightDataETL("test_nan.csv")
    pipeline.extract()
    pipeline.transform()
    
    assert pipeline.df['Alpha'].iloc[1] == 15.0
    print("Verification: Interpolation PASSED")

if __name__ == "__main__":
    test_mach_calculation()
    test_interpolation()