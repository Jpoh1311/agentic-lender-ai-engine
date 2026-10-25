import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Load the property dataset
data = pd.read_csv("data/sample_irish_properties.csv")

print("Dataset loaded successfully")
print("Number of properties:", len(data))
print(data.head())


# Property information used to estimate market value
features = [
    "county",
    "floor_area",
    "bedrooms",
    "property_type",
    "ber_rating"
]

X = data[features]
y = data["market_value"]


# Separate categorical and numerical columns
categorical_features = [
    "county",
    "property_type",
    "ber_rating"
]

numerical_features = [
    "floor_area",
    "bedrooms"
]


# Convert text categories into values the model can process
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Create the complete model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# Use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train and test the model
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel training completed")
print(f"Mean Absolute Error: €{mae:,.2f}")
print(f"R² Score: {r2:.3f}")


# Save the trained pipeline for the API
joblib.dump(model, "property_valuation_model.joblib")

print("\nModel saved as property_valuation_model.joblib")