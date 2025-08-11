import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and clean the dataset
disease_data = pd.read_csv(r"C:\Users\BHARDWAJ\Desktop\Heart_Disease_Predictor\dataset\heart_disease_data.csv", encoding='latin1')
disease_data.columns = disease_data.columns.str.replace('ï»¿', '', regex=False)

# Split features and target
x = disease_data.drop(['target'], axis=1)
y = disease_data['target']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=2)

# Feature scaling
scaler = StandardScaler()
x_train_scaled_df = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
x_test_scaled_df = pd.DataFrame(scaler.transform(x_test), columns=x.columns)