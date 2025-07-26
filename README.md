# 🌾 AI-Powered Crop Yield Prediction System

A smart agricultural planning system that uses machine learning to predict crop yields based on environmental conditions. This project helps farmers, researchers, and agricultural planners make informed decisions to maximize productivity.

## 🎯 Project Overview

This system predicts crop yield (kg/hectare) using environmental factors:
- **Crop Type** (Rice, Wheat, Maize, Barley, Soybean, Cotton)
- **Rainfall** (mm)
- **Temperature** (°C)
- **Humidity** (%)
- **Land Area** (hectares)

## 🏗️ Project Structure

```
CropYieldPrediction/
│
├── model/
│   ├── crop_model.pkl          # Trained Random Forest model
│   └── label_encoder.pkl       # Crop name encoder
│
├── webapp/
│   ├── app.py                  # Flask web application
│   └── templates/
│       └── index.html          # Web interface
│
├── data/
│   └── crops.csv              # Training dataset
│
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd CropYieldPrediction

# Create virtual environment
python -m venv crop_env
source crop_env/bin/activate  # On Windows: crop_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Generate sample training data (if crops.csv doesn't exist)
- Train a Random Forest model
- Save the model and label encoder
- Display performance metrics

### 3. Run the Web Application

```bash
cd webapp
python app.py
```

The application will be available at `http://localhost:5000`

## 📊 Model Performance

The Random Forest model typically achieves:
- **RMSE**: ~300-500 kg/hectare
- **R² Score**: ~0.85-0.95

### Feature Importance
1. **Crop Type**: Most important factor
2. **Rainfall**: Critical for yield prediction
3. **Temperature**: Affects growth rates
4. **Humidity**: Influences disease risk
5. **Area**: Linear scaling factor

## 🌐 Web Interface Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Prediction**: Instant results
- **Input Validation**: Ensures data quality
- **Visual Feedback**: Loading states and error handling
- **Results Display**: Shows both per-hectare and total yield

## 📝 API Endpoints

### POST /predict
Predict crop yield based on input parameters.

**Request Body:**
```json
{
    "crop": "Rice",
    "rainfall": 850,
    "temperature": 26.5,
    "humidity": 78,
    "area": 2.0
}
```

**Response:**
```json
{
    "yield_per_hectare": 3055.60,
    "total_yield": 6111.20,
    "crop": "Rice",
    "rainfall": 850,
    "temperature": 26.5,
    "humidity": 78,
    "area": 2.0
}
```

### GET /api/crops
Get list of available crop types.

### GET /health
Health check endpoint.

## 🧪 Example Usage

### Command Line Prediction
```python
import pickle
import numpy as np

# Load model
with open('model/crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Predict for Rice
crop_encoded = encoder.transform(['Rice'])[0]
prediction = model.predict([[crop_encoded, 850, 26.5, 78, 2.0]])
print(f"Predicted yield: {prediction[0]:.2f} kg/hectare")
```

## 🔧 Customization

### Adding New Crops
1. Update the training data with new crop samples
2. Retrain the model using `train_model.py`
3. Update the web interface dropdown in `index.html`

### Improving Model Accuracy
1. **More Data**: Add more diverse training samples
2. **Feature Engineering**: Add soil type, fertilizer usage, etc.
3. **Model Tuning**: Optimize hyperparameters
4. **Ensemble Methods**: Combine multiple models

### Advanced Features
- **Weather API Integration**: Real-time weather data
- **Satellite Imagery**: Crop health monitoring
- **Historical Analysis**: Trend prediction
- **Mobile App**: React Native or Flutter

## 📈 Model Training Details

### Algorithm: Random Forest Regressor

**Why Random Forest?**
- **Robustness**: Handles non-linear relationships well
- **Feature Importance**: Provides insights into which factors matter most
- **Overfitting Resistance**: Ensemble method reduces overfitting
- **Missing Data**: Can handle incomplete datasets
- **No Scaling Required**: Works with features of different scales

**Model Parameters:**
```python
RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples per leaf
    random_state=42        # Reproducibility
)
```

### Data Preprocessing
1. **Label Encoding**: Convert crop names to numerical values
2. **Feature Scaling**: Not required for Random Forest
3. **Data Validation**: Remove outliers and invalid entries
4. **Train-Test Split**: 80% training, 20% testing

## 🌱 Sample Data Generation

The system generates realistic synthetic data based on agricultural knowledge:

### Crop-Specific Parameters

| Crop | Optimal Rainfall (mm) | Temperature (°C) | Humidity (%) | Base Yield (kg/ha) |
|------|----------------------|------------------|--------------|-------------------|
| Rice | 1000 ± 200 | 28 ± 3 | 75 ± 10 | 4000 |
| Wheat | 400 ± 100 | 20 ± 4 | 60 ± 10 | 3000 |
| Maize | 600 ± 150 | 25 ± 3 | 65 ± 8 | 5000 |
| Barley | 350 ± 80 | 18 ± 3 | 55 ± 8 | 2500 |
| Soybean | 500 ± 120 | 24 ± 3 | 70 ± 8 | 2800 |
| Cotton | 800 ± 180 | 30 ± 4 | 65 ± 12 | 1800 |

## 🔍 Troubleshooting

### Common Issues

**1. Model files not found**
```bash
Error: Model files not found. Please run train_model.py first.
```
**Solution:** Run the training script before starting the web app.

**2. Port already in use**
```bash
Error: [Errno 48] Address already in use
```
**Solution:** Change the port in `app.py` or kill the existing process.

**3. Import errors**
```bash
ModuleNotFoundError: No module named 'sklearn'
```
**Solution:** Install requirements: `pip install -r requirements.txt`

**4. Low prediction accuracy**
- Check data quality and remove outliers
- Increase training data size
- Tune model hyperparameters
- Add more relevant features

## 📊 Performance Monitoring

### Metrics to Track
- **RMSE**: Root Mean Square Error (lower is better)
- **R² Score**: Coefficient of determination (higher is better)
- **MAE**: Mean Absolute Error
- **Prediction Time**: Response latency

### Model Validation
```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

## 🚀 Deployment Options

### Local Development
```bash
python webapp/app.py
```

### Production Deployment

**1. Using Gunicorn (Linux/Mac)**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 webapp.app:app
```

**2. Using Docker**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python train_model.py

EXPOSE 5000
CMD ["python", "webapp/app.py"]
```

**3. Cloud Platforms**
- **Heroku**: Easy deployment with git
- **AWS EC2**: Full control over environment
- **Google Cloud Run**: Serverless containers
- **Azure Web Apps**: Managed hosting

## 🔒 Security Considerations

### Input Validation
- Validate numeric ranges for all inputs
- Sanitize string inputs to prevent injection
- Rate limiting for API endpoints
- HTTPS in production

### Example Security Measures
```python
def validate_input(rainfall, temperature, humidity, area):
    if not (0 <= rainfall <= 3000):
        raise ValueError("Rainfall must be 0-3000mm")
    if not (-10 <= temperature <= 50):
        raise ValueError("Temperature must be -10-50°C")
    if not (0 <= humidity <= 100):
        raise ValueError("Humidity must be 0-100%")
    if not (0 < area <= 1000):
        raise ValueError("Area must be 0-1000 hectares")
```

## 📈 Future Enhancements

### Technical Improvements
1. **Real-time Data Integration**
   - Weather APIs (OpenWeatherMap, AccuWeather)
   - Satellite imagery analysis
   - IoT sensor data

2. **Advanced ML Features**
   - Deep learning models (CNN, LSTM)
   - Time series forecasting
   - Multi-crop rotation optimization
   - Pest and disease prediction

3. **User Experience**
   - Mobile application
   - Offline capability
   - Multi-language support
   - Voice input/output

4. **Business Intelligence**
   - Historical trend analysis
   - Market price integration
   - Profit margin optimization
   - Risk assessment tools

### Agricultural Extensions
- **Soil Analysis**: pH, nutrients, organic matter
- **Irrigation Optimization**: Water usage efficiency
- **Fertilizer Recommendations**: NPK requirements
- **Harvest Timing**: Optimal harvest dates
- **Climate Change Adaptation**: Long-term planning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit changes: `git commit -m "Add feature"`
5. Push to branch: `git push origin feature-name`
6. Create a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Test on multiple Python versions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning framework
- **Flask**: Web framework
- **Agricultural Research**: Domain knowledge sources
- **Open Source Community**: Tools and libraries


**Happy Farming! 🌾**
