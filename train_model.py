import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def create_sample_data():
    """Create sample crop data for training if crops.csv doesn't exist"""
    np.random.seed(42)
    
    crops = ['Rice', 'Wheat', 'Maize', 'Barley', 'Soybean', 'Cotton']
    n_samples = 1000
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        
        # Generate realistic data based on crop type
        if crop == 'Rice':
            rainfall = np.random.normal(1000, 200)
            temp = np.random.normal(28, 3)
            humidity = np.random.normal(75, 10)
            base_yield = 4000
        elif crop == 'Wheat':
            rainfall = np.random.normal(400, 100)
            temp = np.random.normal(20, 4)
            humidity = np.random.normal(60, 10)
            base_yield = 3000
        elif crop == 'Maize':
            rainfall = np.random.normal(600, 150)
            temp = np.random.normal(25, 3)
            humidity = np.random.normal(65, 8)
            base_yield = 5000
        elif crop == 'Barley':
            rainfall = np.random.normal(350, 80)
            temp = np.random.normal(18, 3)
            humidity = np.random.normal(55, 8)
            base_yield = 2500
        elif crop == 'Soybean':
            rainfall = np.random.normal(500, 120)
            temp = np.random.normal(24, 3)
            humidity = np.random.normal(70, 8)
            base_yield = 2800
        else:  # Cotton
            rainfall = np.random.normal(800, 180)
            temp = np.random.normal(30, 4)
            humidity = np.random.normal(65, 12)
            base_yield = 1800
        
        # Ensure positive values
        rainfall = max(50, rainfall)
        temp = max(5, temp)
        humidity = max(30, min(100, humidity))
        area = np.random.uniform(0.5, 10)
        
        # Calculate yield with some realistic relationships
        yield_factor = (
            (rainfall / 500) * 0.3 +
            (temp / 25) * 0.2 +
            (humidity / 70) * 0.15 +
            np.random.normal(1, 0.2)
        )
        
        yield_per_hectare = max(100, base_yield * yield_factor + np.random.normal(0, 200))
        
        data.append({
            'Crop': crop,
            'Rainfall_mm': round(rainfall, 1),
            'Temperature_C': round(temp, 1),
            'Humidity_percent': round(humidity, 1),
            'Area_hectares': round(area, 2),
            'Yield_kg_per_hectare': round(yield_per_hectare, 2)
        })
    
    df = pd.DataFrame(data)
    return df

def train_crop_model():
    """Train the crop yield prediction model"""
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create model directory if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')
    
    # Load or create data
    if os.path.exists('data/crops.csv'):
        df = pd.read_csv('data/crops.csv')
        print("Loaded existing crops.csv")
    else:
        df = create_sample_data()
        df.to_csv('data/crops.csv', index=False)
        print("Created sample crops.csv with 1000 records")
    
    print(f"Dataset shape: {df.shape}")
    print("\nDataset info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Crop_encoded'] = le.fit_transform(df['Crop'])
    
    # Save label encoder
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Prepare features and target
    features = ['Crop_encoded', 'Rainfall_mm', 'Temperature_C', 'Humidity_percent', 'Area_hectares']
    X = df[features]
    y = df['Yield_kg_per_hectare']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.2f} kg/hectare")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    # Save the model
    with open('model/crop_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved successfully!")
    
    # Test prediction
    print(f"\nSample Prediction:")
    sample_input = [[1, 850, 26.5, 78, 2.0]]  # Rice, 850mm rainfall, 26.5°C, 78% humidity, 2 hectares
    sample_pred = model.predict(sample_input)
    print(f"Input: Rice, 850mm rainfall, 26.5°C, 78% humidity, 2 hectares")
    print(f"Predicted Yield: {sample_pred[0]:.2f} kg/hectare")
    
    return model, le

if __name__ == "__main__":
    model, label_encoder = train_crop_model()