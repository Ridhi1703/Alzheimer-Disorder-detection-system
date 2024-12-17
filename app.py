import os
from flask import Flask, render_template, request, redirect, url_for, flash

import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from PIL import Image

app = Flask(__name__)
import os
app.secret_key = os.getenv('SECRET_KEY', 'fallback_key')


# Load models
mri_model = tf.keras.models.load_model('mri_model.h5')
clinical_model = joblib.load('clinical_model.pkl')

# Preprocessing functions
# def preprocess_mri_image(image_path):
#     img = Image.open(image_path).resize((150, 150))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array
def preprocess_mri_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((150, 150))  # Ensure RGB and resizing
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    print("Preprocessed image shape:", img_array.shape)  # Debugging output
    return img_array


def map_yes_no(value):
    return 1 if value.lower() == 'yes' else 0

def map_education(education_value):
    mapping = {'None': 0, 'Bachelors': 1, 'Higher': 2, 'Masters': 3}
    return mapping.get(education_value, 0)



# Prediction functions
def predict_clinical(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = clinical_model.predict(input_df)
    return prediction[0]

# def predict_mri_with_stage(image_path):
#     processed_image = preprocess_mri_image(image_path)
#     prediction = mri_model.predict(processed_image)
#     stages = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
#     stage_index = np.argmax(prediction[0])
#     presence = 1 if stage_index > 0 else 0
#     return presence, stages[stage_index]
def predict_mri_with_stage(image_path):
    processed_image = preprocess_mri_image(image_path)
    print("Image ready for prediction:", processed_image.shape)

    prediction = mri_model.predict(processed_image)
    print("Raw prediction output:", prediction)

    stages = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    stage_index = np.argmax(prediction[0])  # Index with highest probability
    presence = 1 if stage_index > 0 else 0
    return presence, stages[stage_index]



#change the multimodel logic afterwards
def multimodal_predict_with_stage(clinical_input, mri_presence, mri_stage):
    clinical_output = predict_clinical(clinical_input)
    final_presence = (clinical_output + mri_presence) / 2
    final_result = 1 if final_presence >= 0.5 else 0
    return final_result, mri_stage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clinical', methods=['GET', 'POST'])
def clinical_prediction():
    if request.method == 'POST':
        data = {
            'Age': int(request.form['age']),
            'Gender': 0 if request.form['gender'] == 'Male' else 1,
            'EducationLevel': map_education(request.form['education']),
            'BMI': float(request.form['bmi']),
            'Smoking': map_yes_no(request.form['smoking']),
            'AlcoholConsumption': int(request.form['alcohol']),
            'PhysicalActivity': int(request.form['physical_activity']),
            'DietQuality': int(request.form['diet_quality']),
            'SleepQuality': int(request.form['sleep_quality']),
            'FamilyHistoryAlzheimers': map_yes_no(request.form['family_history']),
            'CardiovascularDisease': map_yes_no(request.form['cardiovascular_disease']),
            'Diabetes': map_yes_no(request.form['diabetes']),
            'Depression': map_yes_no(request.form['depression']),
            'HeadInjury': map_yes_no(request.form['head_injury']),
            'Hypertension': map_yes_no(request.form['hypertension']),
            'SystolicBP': int(request.form['systolic_bp']),
            'DiastolicBP': int(request.form['diastolic_bp']),
            'CholesterolTotal': int(request.form['cholesterol_total']),
            'CholesterolLDL': int(request.form['cholesterol_ldl']),
            'CholesterolHDL': int(request.form['cholesterol_hdl']),
            'CholesterolTriglycerides': int(request.form['cholesterol_triglycerides']),
            'MMSE': int(request.form['mmse']),
            'MemoryComplaints': map_yes_no(request.form['memory_complaints']),
            'BehavioralProblems': map_yes_no(request.form['behavioral_problems']),
            'Confusion': map_yes_no(request.form['confusion']),
            'Disorientation': map_yes_no(request.form['disorientation']),
            'PersonalityChanges': map_yes_no(request.form['personality_changes']),
            'DifficultyCompletingTasks': map_yes_no(request.form['difficulty_completing_tasks']),
            'Forgetfulness': map_yes_no(request.form['forgetfulness']),
        }
        prediction = predict_clinical(data)
        result = "Alzheimer's Detected" if prediction == 1 else "No Alzheimer's Detected"
        return render_template('result.html', result=result)
    return render_template('clinical.html')

@app.route('/multimodal', methods=['GET', 'POST'])
def multimodal_prediction():
    if request.method == 'POST':
        clinical_data = {
            'Age': int(request.form['age']),
             'Gender': 0 if request.form['gender'] == 'Male' else 1,
            'EducationLevel': map_education(request.form['education']),
            'BMI': float(request.form['bmi']),
            'Smoking': map_yes_no(request.form['smoking']),
            'AlcoholConsumption': int(request.form['alcohol']),
            'PhysicalActivity': int(request.form['physical_activity']),
            'DietQuality': int(request.form['diet_quality']),
            'SleepQuality': int(request.form['sleep_quality']),
            'FamilyHistoryAlzheimers': map_yes_no(request.form['family_history']),
            'CardiovascularDisease': map_yes_no(request.form['cardiovascular_disease']),
            'Diabetes': map_yes_no(request.form['diabetes']),
            'Depression': map_yes_no(request.form['depression']),
            'HeadInjury': map_yes_no(request.form['head_injury']),
            'Hypertension': map_yes_no(request.form['hypertension']),
            'SystolicBP': int(request.form['systolic_bp']),
            'DiastolicBP': int(request.form['diastolic_bp']),
            'CholesterolTotal': int(request.form['cholesterol_total']),
            'CholesterolLDL': int(request.form['cholesterol_ldl']),
            'CholesterolHDL': int(request.form['cholesterol_hdl']),
            'CholesterolTriglycerides': int(request.form['cholesterol_triglycerides']),
            'MMSE': int(request.form['mmse']),
            'MemoryComplaints': map_yes_no(request.form['memory_complaints']),
            'BehavioralProblems': map_yes_no(request.form['behavioral_problems']),
            'Confusion': map_yes_no(request.form['confusion']),
            'Disorientation': map_yes_no(request.form['disorientation']),
            'PersonalityChanges': map_yes_no(request.form['personality_changes']),
            'DifficultyCompletingTasks': map_yes_no(request.form['difficulty_completing_tasks']),
            'Forgetfulness': map_yes_no(request.form['forgetfulness']),
        }
        if 'mri_image' not in request.files or not request.files['mri_image']:
            flash('Please upload an MRI image.')
            return redirect(request.url)
        
        mri_image = request.files['mri_image']
        mri_presence, mri_stage = predict_mri_with_stage(mri_image)
        final_result, stage = multimodal_predict_with_stage(clinical_data, mri_presence, mri_stage)
        result = "Alzheimer's Detected" if final_result == 1 else "No Alzheimer's Detected"
        return render_template('result.html', result=result, stage=stage)
    return render_template('multimodal.html')

if __name__ == '__main__':
    app.run(debug=True)
