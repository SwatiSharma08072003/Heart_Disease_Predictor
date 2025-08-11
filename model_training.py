from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_preprocessing import x_train_scaled_df, x_test_scaled_df, y_train, y_test

# Train the model
model_scaled = LogisticRegression()
model_scaled.fit(x_train_scaled_df, y_train)

# Evaluate accuracy
x_train_acc = accuracy_score(model_scaled.predict(x_train_scaled_df), y_train)
x_test_acc = accuracy_score(model_scaled.predict(x_test_scaled_df), y_test)