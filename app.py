import os
# from flask import Flask, render_template, request, redirect, url_for, flash

# ✨ NEW import to enable session handling
from flask import Flask, render_template, request, redirect, url_for, flash, session

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

from flask import session, make_response, render_template_string
from datetime import datetime
from io import BytesIO
from xhtml2pdf import pisa





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

def generate_treatment_plan(result, stage=None):
    result = result.lower().strip() if result else "unknown"
    stage = stage.lower().strip() if stage else "n/a"

    if "alzheimer" in result:
        if stage == "very mild":
            therapy = "Suggested therapies: Cognitive stimulation therapy, memory enhancement programs."
            lifestyle = "Lifestyle recommendations: Engage in light exercises like walking, maintain a balanced diet with Omega-3."
        
        elif stage == "mild":
            therapy = "Suggested therapies: Cognitive rehabilitation, family therapy, memory training."
            lifestyle = "Lifestyle recommendations: Moderate physical exercise, social activities, Mediterranean diet."
        
        elif stage == "moderate":
            therapy = "Suggested therapies: Physical therapy, speech therapy, assistance with daily living."
            lifestyle = "Lifestyle recommendations: Structured daily routines, caregiver support, and nutritional therapy."
        
        else:
            # For clinical-only where stage is N/A
            therapy = "Suggested therapies: Cognitive assessment, regular neurologist consultations, memory support programs."
            lifestyle = "Lifestyle recommendations: Balanced diet, brain games, safe home environment, regular walks."
    
    elif "no alzheimer" in result.lower():
        therapy = "Suggested therapies: Regular check-ups, mental exercises."
        lifestyle = "Lifestyle recommendations: Keep active with social activities, eat a healthy diet, focus on cognitive exercises."

    else:
        therapy = "No recommendations available."
        lifestyle = "Please consult a healthcare professional for personalized treatment."
    
    return therapy, lifestyle




# Prediction functions
def predict_clinical(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = clinical_model.predict(input_df)
    return prediction[0]

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

 # Store friendly patient details separately
        patient_info = {
            'Name': request.form.get('name', 'N/A'),
            'DOB': request.form.get('dob', 'N/A'),
            'Gender': request.form.get('gender', 'N/A'),
            'Contact': request.form.get('contact', 'N/A'),
            'Age':request.form.get('age','N/A')          
        }
        Patient_clinical_data={
             'BMI': request.form.get('bmi','N/A'),
             'SystolicBP': request.form.get('systolic_bp','N/A'),
            'DiastolicBP': request.form.get('diastolic_bp','N/A'),
            'CholesterolTotal': request.form.get('cholesterol_total','N/A'),
            'CholesterolLDL': request.form.get('cholesterol_ldl','N/A'),
            'CholesterolHDL': request.form.get('cholesterol_hdl','N/A'),
            'CholesterolTriglycerides': request.form.get('cholesterol_triglycerides','N/A'),
            'MMSE': request.form.get('mmse','N/A'), 
            'FamilyHistoryAlzheimers': request.form.get('family_history','N/A'),
            'CardiovascularDisease': request.form.get('cardiovascular_disease','N/A'),
            'Diabetes':request.form.get('diabetes','N/A'),
            'Depression':request.form.get('depression','N/A'),
            'HeadInjury': request.form.get('head_injury','N/A'),
            'Hypertension': request.form.get('hypertension','N/A')}


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
 # ✨ Save to session
      
        session['patient_data'] = patient_info
        session['medical_data']=Patient_clinical_data
        session['result'] = result
        session['stage'] = "N/A"  # Not available here

        therapy, lifestyle = generate_treatment_plan(result)

        return render_template('result.html', result=result,therapy=therapy, lifestyle=lifestyle)
    return render_template('clinical.html')

@app.route('/multimodal', methods=['GET', 'POST'])
def multimodal_prediction():



    if request.method == 'POST':

 # Store friendly patient details separately
        patient_info = {
            'Name': request.form.get('name', 'N/A'),
            'DOB': request.form.get('dob', 'N/A'),
            'Gender': request.form.get('gender', 'N/A'),
            'Contact': request.form.get('contact', 'N/A'),
            'Age':request.form.get('age','N/A')          
        }
        Patient_clinical_data={
             'BMI': request.form.get('bmi','N/A'),
             'SystolicBP': request.form.get('systolic_bp','N/A'),
            'DiastolicBP': request.form.get('diastolic_bp','N/A'),
            'CholesterolTotal': request.form.get('cholesterol_total','N/A'),
            'CholesterolLDL': request.form.get('cholesterol_ldl','N/A'),
            'CholesterolHDL': request.form.get('cholesterol_hdl','N/A'),
            'CholesterolTriglycerides': request.form.get('cholesterol_triglycerides','N/A'),
            'MMSE': request.form.get('mmse','N/A'), 
            'FamilyHistoryAlzheimers': request.form.get('family_history','N/A'),
            'CardiovascularDisease': request.form.get('cardiovascular_disease','N/A'),
            'Diabetes':request.form.get('diabetes','N/A'),
            'Depression':request.form.get('depression','N/A'),
            'HeadInjury': request.form.get('head_injury','N/A'),
            'Hypertension': request.form.get('hypertension','N/A')}

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
        
        # === NEW CODE START === Save the uploaded MRI image to the 'uploads' folder
        upload_folder = 'static/uploads'  # Make sure this exists or is created earlier
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        image_path = os.path.join(upload_folder, mri_image.filename)
        mri_image.save(image_path)
        # === NEW CODE END ===

        # === MODIFIED: Use saved image path instead of direct file ===
        mri_presence, mri_stage = predict_mri_with_stage(image_path)


        final_result, stage = multimodal_predict_with_stage(clinical_data, mri_presence, mri_stage)
        result = " Alzheimer's Detected" if final_result == 1 else "No Alzheimer's Detected"

        session['patient_data'] = patient_info
        session['medical_data']=Patient_clinical_data
        session['result'] = result
        session['stage'] = stage

        therapy, lifestyle = generate_treatment_plan(result, stage)

        return render_template('result.html', result=result, stage=stage, therapy=therapy, lifestyle=lifestyle)
    return render_template('multimodal.html')

@app.route('/view_report')
def view_report():
    # ✨ Access session data
    patient_data = session.get('patient_data', {})
    medical_data = session.get('medical_data',{})
    result = session.get('result', 'No Result')
    stage = session.get('stage', 'N/A')
    therapy, lifestyle = generate_treatment_plan(result, stage)
    report_date = datetime.now().strftime("%d-%m-%Y")

    return render_template('report.html', patient=patient_data,medical=medical_data, result=result, stage=stage, therapy=therapy, 
                           lifestyle=lifestyle,  report_date=report_date)


@app.route('/download_report')
def download_report():
    # Get all stored session data
   # ✨ Access session data
    patient_data = session.get('patient_data', {})
    medical_data = session.get('medical_data',{})
    result = session.get('result', 'No Result')
    stage = session.get('stage', 'N/A')
    report_date = datetime.now().strftime("%d-%m-%Y")
    therapy, lifestyle = generate_treatment_plan(result, stage)

    # Render HTML for PDF generation
    html = render_template('report.html',
                           patient=patient_data,
                           medical=medical_data,
                           result=result,
                           stage=stage,
                           therapy=therapy, 
                           lifestyle=lifestyle,
                           report_date=report_date)
    
    

    # Generate PDF
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_buffer)

    if pisa_status.err:
        return "Error generating PDF", 500

    # Prepare PDF for download
    pdf_buffer.seek(0)
    response = make_response(pdf_buffer.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=alzheimers_report.pdf'
    return response



if __name__ == '__main__':
    app.run(debug=True)
