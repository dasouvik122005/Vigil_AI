import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_samples=1000, filename="sensor_data.csv"):
    """
    Generates a realistic synthetic dataset for the hackathon demo.
    Features:
    - temperature: Normal range 20-35, anomalies > 45 or < 10
    - pressure: Normal range 1.0-1.5, anomalies > 2.0
    - vibration: Normal range 0.1-0.5, anomalies > 1.0
    """
    np.random.seed(42)
    
    # Generate normal data
    temp = np.random.normal(loc=25, scale=3, size=num_samples)
    pressure = np.random.normal(loc=1.2, scale=0.1, size=num_samples)
    vibration = np.random.normal(loc=0.3, scale=0.05, size=num_samples)
    
    df = pd.DataFrame({
        'temperature': temp,
        'pressure': pressure,
        'vibration': vibration
    })
    
    # Inject anomalies (about 5% of data)
    num_anomalies = int(num_samples * 0.05)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = random.choice(['temp_high', 'pressure_high', 'vibration_high', 'all_high'])
        
        if anomaly_type == 'temp_high' or anomaly_type == 'all_high':
            df.loc[idx, 'temperature'] = random.uniform(45, 60)
        if anomaly_type == 'pressure_high' or anomaly_type == 'all_high':
            df.loc[idx, 'pressure'] = random.uniform(2.0, 3.0)
        if anomaly_type == 'vibration_high' or anomaly_type == 'all_high':
            df.loc[idx, 'vibration'] = random.uniform(1.2, 2.5)
            
    df.to_csv(filename, index=False)
    print(f"Generated {num_samples} samples to {filename}")
    return df

class DataStreamer:
    def __init__(self, filename="sensor_data.csv"):
        if not os.path.exists(filename):
            generate_synthetic_data(filename=filename)
        self.df = pd.read_csv(filename)
        self.current_index = 0
        self.total_rows = len(self.df)
        
    def get_next_batch(self, count=2, missing_prob=0.25):
        """
        Returns the next batch of data. Simulates missing data.
        """
        batch = []
        for _ in range(count):
            if self.current_index >= self.total_rows:
                self.current_index = 0 # loop back
                
            row = self.df.iloc[self.current_index].to_dict()
            self.current_index += 1
            
            # Simulate "Hard Mode" - 25% missing data
            is_corrupted = random.random() < missing_prob
            
            batch.append({
                'temperature': row['temperature'] if not is_corrupted else None,
                'pressure': row['pressure'] if not is_corrupted else None,
                'vibration': row['vibration'] if not is_corrupted else None,
                'is_corrupted': is_corrupted
            })
            
        return batch

if __name__ == "__main__":
    # Test generation
    generate_synthetic_data()
