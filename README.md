# Deploying a Scalable ML Pipeline with FastAPI

[![GitHub Actions](https://github.com/aliceklink/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions/workflows/python-app.yml/badge.svg)](https://github.com/aliceklink/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions)

A complete machine learning pipeline that predicts whether an individual's annual income exceeds $50,000 based on census data. The project implements a Random Forest classifier deployed through a FastAPI REST API with comprehensive testing and continuous integration.

##  Features

- **Machine Learning Pipeline**: Complete data preprocessing, model training, and evaluation pipeline
- **RESTful API**: FastAPI-based web service for model inference
- **Comprehensive Testing**: Unit tests for ML functions with pytest
- **Model Performance Analysis**: Slice-based performance evaluation across demographic groups
- **Continuous Integration**: GitHub Actions workflow with automated testing
- **Model Card**: Detailed documentation of model performance and ethical considerations

##  Model Performance

- **Precision**: 80.58%
- **Recall**: 54.53%  
- **F1-Score**: 65.04%

The model shows varying performance across different demographic slices, with detailed analysis available in `slice_output.txt`.

##  Installation

### Environment Setup

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

**Option 1: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate fastapi
```

**Option 2: Using Pip**
```bash
pip install -r requirements.txt
```

### Repository Setup

1. Clone the repository:
```bash
git clone https://github.com/aliceklink/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI
```

2. Set up the environment using one of the options above

##  Project Structure

```
├── data/
│   └── census.csv              # Census dataset
├── ml/
│   ├── data.py                 # Data processing functions
│   └── model.py                # Model training and evaluation
├── model/
│   ├── model.pkl               # Trained Random Forest model
│   └── encoder.pkl             # Fitted OneHot encoder
├── screenshots/
│   ├── continuous_integration.png
│   ├── unit_test.png
│   └── local_api.png
├── main.py                     # FastAPI application
├── train_model.py              # Model training script
├── test_ml.py                  # Unit tests
├── local_api.py                # API client for testing
├── slice_output.txt            # Model performance by demographic slices
├── model_card_template.md      # Model documentation
└── README.md
```

##  Training the Model

The model uses 1994 U.S. Census data to predict income levels. To train the model:

```bash
python train_model.py
```

This script will:
- Load and preprocess the census data
- Split data into training and testing sets (80/20)
- Train a Random Forest classifier
- Evaluate model performance overall and on demographic slices
- Save the trained model and encoder

##  Testing

Run the comprehensive unit test suite:

```bash
pytest test_ml.py -v
```

The tests cover:
- Data processing function return types
- Model algorithm verification
- Metrics computation accuracy
- Dataset size and type validation
- Slice-based performance evaluation

##  API Usage

### Starting the API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### API Endpoints

**GET /** - Welcome message
```bash
curl http://127.0.0.1:8000/
```

**POST /data/** - Income prediction
```bash
curl -X POST "http://127.0.0.1:8000/data/" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
  }'
```

### Interactive API Documentation

Visit `http://127.0.0.1:8000/docs` for interactive API documentation powered by Swagger UI.

### Testing the API

Use the provided client script to test both endpoints:

```bash
python local_api.py
```

##  Data

The project uses the 1994 U.S. Census Bureau database from the UCI Machine Learning Repository. The dataset includes demographic and employment information for income prediction.

**Key Features:**
- **Categorical**: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Numerical**: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- **Target**: salary (<=50K, >50K)

**Data Preprocessing:**
- One-hot encoding for categorical features
- Label binarization for target variable
- Stratified train-test split to maintain class distribution

##  CI/CD Pipeline

This project uses GitHub Actions for continuous integration:

- **Automated Testing**: Runs pytest on every push and pull request
- **Code Quality**: Enforces coding standards with flake8
- **Python Version**: Maintains consistency with Python 3.10
- **Status Badge**: [![CI Status](https://github.com/aliceklink/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions/workflows/python-app.yml/badge.svg)](https://github.com/aliceklink/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions)

##  Model Documentation

Comprehensive model documentation is available in [`model_card_template.md`](model_card_template.md), including:
- Model architecture and hyperparameters
- Training data description
- Performance metrics and evaluation
- Ethical considerations and bias analysis
- Usage recommendations and limitations

##  Performance Analysis

The model's performance varies across demographic groups. Key findings:

- **Gender**: Performance differences between male and female predictions
- **Education**: Higher accuracy for individuals with advanced degrees
- **Occupation**: Variation across different professional categories
- **Race**: Performance disparities across racial groups

Detailed slice-by-slice analysis is available in `slice_output.txt`.

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  Ethical Considerations

This model is trained on 1994 census data and may not reflect current socioeconomic conditions. Users should be aware of potential biases and avoid using this model for high-stakes individual decisions without proper validation and human oversight.

##  License

This project is licensed under the terms specified in `LICENSE.txt`.

##  Acknowledgments

- UCI Machine Learning Repository for the Census Income dataset
- Udacity for the project framework and educational content
- scikit-learn and FastAPI communities for excellent documentation and tools
