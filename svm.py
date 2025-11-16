from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import json

with open('train_part1.json', 'r') as f:
    data = json.load(f)
# Prepare features and labels

X = []
y = []

for record in data:
# Concatenate image and text embeddings
    features = record['image_embedding'] + record['text_embedding']
    X.append(features)
    y.append(record['label'])

X = np.array(X)  # Shape: (1530, 1024)
y = np.array(y)  # Shape: (1530,)

# Create the pipeline
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        C=0.15,
        kernel='rbf',
        degree=4,          # ignored by RBF, required by constructor
        gamma='scale',
        class_weight='balanced',
        random_state=42
    ))
])

# Fit on training data
svm_model.fit(X, y)
with open('test.json', 'r') as f:
    test_data = json.load(f)

X_test = []
test_ids = []

for record in test_data:
    features = record['image_embedding'] + record['text_embedding']
    X_test.append(features)
    test_ids.append(record['id'])

X_test = np.array(X_test)

# Predict on test set
predictions = svm_model.predict(X_test)

# 7. Create submission file

submission = pd.DataFrame({
'row_id': test_ids,
'target': predictions
})
submission.to_csv('svm_full_train_v2.csv', index=False)
print("Submission file created!")