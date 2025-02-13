import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Synthetic data generation
def generate_synthetic_data(samples=2000):
    np.random.seed(42)

    # Synthetic features
    defect_count = np.random.randint(0, 11, samples)  # 0-10 defects
    color_variance = np.round(np.random.uniform(0, 1, samples), 2)  # 0-1.0
    texture_variance = np.round(np.random.uniform(0, 1, samples), 2)  # 0-1.0

    # Assign grades based on criteria (1-4)
    grade_defects = np.where(defect_count == 0, 1,
                            np.where(defect_count <= 2, 2,
                                    np.where(defect_count <= 5, 3, 4)))

    grade_color = np.where(color_variance < 0.1, 1,
                          np.where(color_variance < 0.3, 2,
                                  np.where(color_variance < 0.6, 3, 4)))

    grade_texture = np.where(texture_variance < 0.1, 1,
                            np.where(texture_variance < 0.3, 2,
                                    np.where(texture_variance < 0.6, 3, 4)))

    data = pd.DataFrame({
        'defects': defect_count,
        'color_var': color_variance,
        'texture_var': texture_variance,
        'grade_defects': grade_defects,
        'grade_color': grade_color,
        'grade_texture': grade_texture
    })
    return data

# Build and train model
def train_and_save_model():
    # Generate synthetic data
    data = generate_synthetic_data(samples=2000)

    # Features and labels
    X = data[['defects', 'color_var', 'texture_var']]
    y = data[['grade_defects', 'grade_color', 'grade_texture']]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert grades to categorical (one-hot encoding)
    y_defects = to_categorical(y['grade_defects'] - 1, num_classes=4)
    y_color = to_categorical(y['grade_color'] - 1, num_classes=4)
    y_texture = to_categorical(y['grade_texture'] - 1, num_classes=4)

    # Split data into training and testing sets
    X_train, X_test, y_train_defects, y_test_defects, y_train_color, y_test_color, y_train_texture, y_test_texture = train_test_split(
        X_scaled, y_defects, y_color, y_texture, test_size=0.2, random_state=42
    )

    # Build the model
    input_layer = Input(shape=(3,))
    dense = Dense(64, activation='relu')(input_layer)
    dense = Dropout(0.2)(dense)

    # Separate output layers for each attribute
    output_defects = Dense(4, activation='softmax', name='defects')(dense)
    output_color = Dense(4, activation='softmax', name='color')(dense)
    output_texture = Dense(4, activation='softmax', name='texture')(dense)

    model = Model(
        inputs=input_layer,
        outputs=[output_defects, output_color, output_texture]
    )

    # Compile the model
    model.compile(
        optimizer='adam',
        loss={'defects': 'categorical_crossentropy',
              'color': 'categorical_crossentropy',
              'texture': 'categorical_crossentropy'},
        metrics={'defects': 'accuracy', 'color': 'accuracy', 'texture': 'accuracy'}
    )

    # Train the model
    history = model.fit(
        X_train,
        {'defects': y_train_defects, 'color': y_train_color, 'texture': y_train_texture},
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    # Save the model and scaler
    save_model(model, 'ml_model/fabric_grader.h5')
    joblib.dump(scaler, 'ml_model/scaler.pkl')

    print("Model and scaler saved successfully!")

if __name__ == '__main__':
    train_and_save_model()