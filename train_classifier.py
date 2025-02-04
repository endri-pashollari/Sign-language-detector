import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np


# Get the correct path to data.pickle
SIGN_LANG_DIR = os.path.dirname(os.path.abspath(__file__))
pickle_path = os.path.join(SIGN_LANG_DIR, "data.pickle")  

# Load the pickle file
with open(pickle_path, 'rb') as f:
    data_dict = pickle.load(f)  

#print("Loaded data.pickle successfully!")
#print("Keys in dictionary:", data_dict.keys())  # Check available keys
#print(data_dict)  # Print the actual data


data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify= labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly '.format(score * 100))

SIGN_LANG_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the full path for saving the model
model_file_path = os.path.join(SIGN_LANG_DIR, "model.p")

# Save the trained model
with open(model_file_path, 'wb') as f:
    pickle.dump({'model': model}, f)


y_predict = model.predict(x_test[:10])  # Predict first 10 test samples
print("Sample Predictions:", y_predict)  # Should show multiple classes



