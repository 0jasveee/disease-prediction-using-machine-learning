from tkinter import *
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Complete list of 132 symptoms
l1 = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
      'joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting',
      'burning_micturition','spotting_urination','fatigue','weight_gain','anxiety',
      'cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
      'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes',
      'breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin',
      'dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
      'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
      'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
      'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
      'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion',
      'chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
      'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness',
      'cramps','bruising','obesity','swollen_legs','swollen_blood_vessels',
      'puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties',
      'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech',
      'knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
      'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
      'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of_urine',
      'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
      'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body',
      'belly_pain','abnormal_menstruation','dischromic_patches','watering_from_eyes',
      'increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum',
      'lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
      'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
      'history_of_alcohol_consumption','blood_in_sputum','prominent_veins_on_calf',
      'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring',
      'skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails',
      'blister','red_sore_around_nose','yellow_crust_ooze']

# 41 diseases
disease = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
           'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma',
           'Hypertension','Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)',
           'Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
           'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis',
           'Tuberculosis','Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
           'Heart attack','Varicose veins','Hypothyroidism','Hyperthyroidism','Hypoglycemia',
           'Osteoarthristis','Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne',
           'Urinary tract infection','Psoriasis','Impetigo']

class DiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disease Prediction System")
        self.root.configure(background='#2C3E50')
        self.root.geometry('700x650')
        
        # Initialize models
        self.models_trained = False
        self.create_sample_data()
        self.train_models()
        
        # Setup GUI
        self.setup_gui()
    
    def create_sample_data(self):
        """Create sample training data with realistic symptom-disease patterns"""
        # Define symptom patterns for each disease
        disease_symptoms = {
            0: ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches'],  # Fungal infection
            1: ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],  # Allergy
            2: ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough'],  # GERD
            3: ['itching', 'vomiting', 'yellowish_skin', 'nausea', 'loss_of_appetite'],  # Chronic cholestasis
            4: ['itching', 'skin_rash', 'stomach_pain', 'burning_micturition', 'spotting_urination'],  # Drug Reaction
            5: ['vomiting', 'indigestion', 'loss_of_appetite', 'abdominal_pain', 'passage_of_gases'],  # Peptic ulcer
            6: ['muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts'],  # AIDS
            7: ['fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level', 'blurred_and_distorted_vision', 'obesity', 'excessive_hunger', 'increased_appetite', 'polyuria'],  # Diabetes
            8: ['vomiting', 'sunken_eyes', 'dehydration', 'diarrhoea'],  # Gastroenteritis
            9: ['fatigue', 'cough', 'high_fever', 'breathlessness', 'family_history', 'mucoid_sputum'],  # Bronchial Asthma
            10: ['headache', 'chest_pain', 'dizziness', 'loss_of_balance', 'lack_of_concentration'],  # Hypertension
            11: ['acidity', 'indigestion', 'headache', 'blurred_and_distorted_vision', 'excessive_hunger', 'stiff_neck', 'depression', 'irritability', 'visual_disturbances'],  # Migraine
            12: ['back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness', 'loss_of_balance'],  # Cervical spondylosis
            13: ['vomiting', 'headache', 'weakness_of_one_body_side', 'altered_sensorium'],  # Paralysis
            14: ['itching', 'vomiting', 'fatigue', 'weight_loss', 'high_fever', 'yellowish_skin', 'dark_urine', 'abdominal_pain'],  # Jaundice
            15: ['chills', 'vomiting', 'high_fever', 'sweating', 'headache', 'nausea', 'diarrhoea', 'muscle_pain'],  # Malaria
            16: ['itching', 'skin_rash', 'fatigue', 'lethargy', 'high_fever', 'headache', 'loss_of_appetite', 'mild_fever', 'swelled_lymph_nodes', 'malaise', 'red_spots_over_body'],  # Chicken pox
            17: ['skin_rash', 'chills', 'joint_pain', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'malaise', 'muscle_pain', 'red_spots_over_body'],  # Dengue
            18: ['chills', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'constipation', 'abdominal_pain', 'diarrhoea', 'toxic_look_(typhos)', 'belly_pain'],  # Typhoid
            19: ['joint_pain', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowish_skin', 'yellowing_of_eyes', 'dark_urine'],  # Hepatitis A
            20: ['itching', 'fatigue', 'lethargy', 'yellowish_skin', 'dark_urine', 'loss_of_appetite', 'abdominal_pain', 'yellow_urine', 'yellowing_of_eyes', 'malaise', 'receiving_blood_transfusion', 'receiving_unsterile_injections'],  # Hepatitis B
            21: ['fatigue', 'weight_loss', 'restlessness', 'lethargy', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowish_skin', 'dark_urine', 'yellowing_of_eyes'],  # Hepatitis C
            22: ['joint_pain', 'vomiting', 'fatigue', 'high_fever', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes'],  # Hepatitis D
            23: ['joint_pain', 'vomiting', 'fatigue', 'high_fever', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes'],  # Hepatitis E
            24: ['vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload'],  # Alcoholic hepatitis
            25: ['chills', 'vomiting', 'fatigue', 'weight_loss', 'cough', 'high_fever', 'breathlessness', 'sweating', 'loss_of_appetite', 'mild_fever', 'yellowing_of_eyes', 'swelled_lymph_nodes', 'malaise', 'phlegm', 'chest_pain', 'blood_in_sputum'],  # Tuberculosis
            26: ['continuous_sneezing', 'chills', 'fatigue', 'cough', 'high_fever', 'headache', 'swelled_lymph_nodes', 'malaise', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'loss_of_smell', 'muscle_pain'],  # Common Cold
            27: ['chills', 'fatigue', 'cough', 'high_fever', 'breathlessness', 'sweating', 'malaise', 'phlegm', 'chest_pain', 'fast_heart_rate', 'rusty_sputum'],  # Pneumonia
            28: ['constipation', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus'],  # Dimorphic hemorrhoids
            29: ['vomiting', 'breathlessness', 'sweating', 'chest_pain'],  # Heart attack
            30: ['fatigue', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'prominent_veins_on_calf'],  # Varicose veins
            31: ['fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings', 'lethargy', 'dizziness', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'depression', 'irritability', 'abnormal_menstruation'],  # Hypothyroidism
            32: ['fatigue', 'mood_swings', 'weight_loss', 'restlessness', 'sweating', 'diarrhoea', 'fast_heart_rate', 'excessive_hunger', 'muscle_weakness', 'irritability', 'abnormal_menstruation'],  # Hyperthyroidism
            33: ['vomiting', 'fatigue', 'anxiety', 'sweating', 'headache', 'nausea', 'blurred_and_distorted_vision', 'slurred_speech', 'palpitations'],  # Hypoglycemia
            34: ['joint_pain', 'neck_pain', 'knee_pain', 'hip_joint_pain', 'swelling_joints', 'painful_walking'],  # Osteoarthritis
            35: ['muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'painful_walking'],  # Arthritis
            36: ['vomiting', 'headache', 'nausea', 'spinning_movements', 'loss_of_balance', 'unsteadiness'],  # Vertigo
            37: ['skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'],  # Acne
            38: ['burning_micturition', 'foul_smell_of_urine', 'continuous_feel_of_urine', 'bladder_discomfort'],  # UTI
            39: ['skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails'],  # Psoriasis
            40: ['skin_rash', 'high_fever', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'],  # Impetigo
        }
        
        # Generate training data
        data = []
        samples_per_disease = 15  # Generate 15 samples per disease
        
        for disease_id, symptoms in disease_symptoms.items():
            for _ in range(samples_per_disease):
                # Create a sample with main symptoms
                sample = [0] * len(l1)
                
                # Add main symptoms (80-100% probability)
                for symptom in symptoms:
                    if symptom in l1:
                        if np.random.random() < 0.9:  # 90% chance to include
                            sample[l1.index(symptom)] = 1
                
                # Add some random noise (5% of other symptoms)
                num_noise = np.random.randint(0, 3)
                noise_indices = np.random.choice(len(l1), size=num_noise, replace=False)
                for idx in noise_indices:
                    if sample[idx] == 0:  # Don't overwrite existing symptoms
                        sample[idx] = 1
                
                sample.append(disease_id)
                data.append(sample)
        
        # Create DataFrame
        columns = l1 + ['prognosis']
        self.df = pd.DataFrame(data, columns=columns)
        
    def train_models(self):
        """Train all machine learning models"""
        try:
            # Prepare data
            self.X = self.df[l1]
            self.y = self.df['prognosis'].values
            
            # Train Decision Tree
            self.dt_model = tree.DecisionTreeClassifier(
                max_depth=15, 
                min_samples_split=3,
                random_state=42
            )
            self.dt_model.fit(self.X, self.y)
            
            # Train Random Forest
            self.rf_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=15,
                random_state=42
            )
            self.rf_model.fit(self.X, self.y)
            
            # Train Naive Bayes
            self.nb_model = GaussianNB()
            self.nb_model.fit(self.X, self.y)
            
            self.models_trained = True
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training models: {str(e)}")
    
    def setup_gui(self):
        """Setup the GUI elements"""
        # Main frame
        main_frame = Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = Label(main_frame, text="🏥 Disease Prediction System", 
                     font=("Arial", 22, "bold"), fg="#ECF0F1", bg="#2C3E50")
        title.pack(pady=(0, 5))
        
        subtitle = Label(main_frame, text="AI-Powered Medical Diagnosis Assistant", 
                        font=("Arial", 11), fg="#BDC3C7", bg="#2C3E50")
        subtitle.pack(pady=(0, 20))
        
        # Input frame
        input_frame = Frame(main_frame, bg='#34495E', relief=GROOVE, bd=2)
        input_frame.pack(fill=BOTH, expand=True, pady=(0, 15))
        
        # Patient name
        name_frame = Frame(input_frame, bg='#34495E')
        name_frame.pack(fill=X, padx=15, pady=(15, 10))
        
        Label(name_frame, text="Patient Name:", font=("Arial", 11, "bold"), 
              fg="#ECF0F1", bg="#34495E").pack(side=LEFT)
        
        self.name_var = StringVar()
        name_entry = Entry(name_frame, textvariable=self.name_var, 
                          font=("Arial", 11), width=30, bg='#ECF0F1')
        name_entry.pack(side=RIGHT, padx=10)
        
        # Symptoms section
        symptoms_label = Label(input_frame, text="Select Symptoms (up to 5):", 
                              font=("Arial", 11, "bold"), 
                              fg="#ECF0F1", bg="#34495E")
        symptoms_label.pack(anchor=W, padx=15, pady=(10, 5))
        
        # Symptom dropdowns
        self.symptom_vars = []
        OPTIONS = ['None'] + sorted(l1)
        
        for i in range(5):
            symptom_frame = Frame(input_frame, bg='#34495E')
            symptom_frame.pack(fill=X, padx=15, pady=5)
            
            Label(symptom_frame, text=f"Symptom {i+1}:", 
                  font=("Arial", 10), fg="#BDC3C7", bg="#34495E", width=12, 
                  anchor=W).pack(side=LEFT)
            
            var = StringVar(value="None")
            self.symptom_vars.append(var)
            
            dropdown = ttk.Combobox(symptom_frame, textvariable=var, 
                                   values=OPTIONS, font=("Arial", 10), 
                                   width=35, state='readonly')
            dropdown.pack(side=LEFT, padx=5)
        
        # Predict button
        predict_btn = Button(input_frame, text="🔍 Predict Disease", 
                           command=self.predict_disease, 
                           font=("Arial", 12, "bold"),
                           bg="#27AE60", fg="white", 
                           activebackground="#229954",
                           cursor="hand2", relief=RAISED, bd=3,
                           padx=20, pady=8)
        predict_btn.pack(pady=15)
        
        # Results frame
        results_frame = Frame(main_frame, bg='#34495E', relief=GROOVE, bd=2)
        results_frame.pack(fill=BOTH, expand=True)
        
        Label(results_frame, text="📊 Prediction Results", 
              font=("Arial", 13, "bold"), 
              fg="#ECF0F1", bg="#34495E").pack(pady=(15, 10))
        
        # Result labels for each algorithm
        algorithms = [
            ("Decision Tree", "#3498DB"),
            ("Random Forest", "#9B59B6"),
            ("Naive Bayes", "#E67E22")
        ]
        
        self.result_labels = []
        
        for algo, color in algorithms:
            result_container = Frame(results_frame, bg='#34495E')
            result_container.pack(fill=X, padx=20, pady=5)
            
            Label(result_container, text=f"{algo}:", 
                  font=("Arial", 10, "bold"), 
                  fg=color, bg="#34495E", width=15, anchor=W).pack(side=LEFT)
            
            result_label = Label(result_container, text="No prediction yet", 
                               font=("Arial", 10), 
                               fg="#BDC3C7", bg="#2C3E50", 
                               width=35, anchor=W, relief=SUNKEN, padx=10, pady=5)
            result_label.pack(side=LEFT, padx=10)
            self.result_labels.append(result_label)
        
        # Consensus label
        self.consensus_label = Label(results_frame, text="", 
                                    font=("Arial", 11, "bold"), 
                                    fg="#F1C40F", bg="#34495E")
        self.consensus_label.pack(pady=10)
        
        # Clear button
        clear_btn = Button(results_frame, text="🔄 Clear All", 
                          command=self.clear_fields,
                          font=("Arial", 10), bg="#E74C3C", fg="white",
                          activebackground="#C0392B",
                          cursor="hand2", relief=RAISED, bd=2,
                          padx=15, pady=5)
        clear_btn.pack(pady=(5, 15))
    
    def predict_disease(self):
        """Predict disease based on selected symptoms"""
        if not self.models_trained:
            messagebox.showerror("Error", "Models not trained yet!")
            return
        
        # Validate patient name
        if not self.name_var.get().strip():
            messagebox.showwarning("⚠ Warning", "Please enter patient name!")
            return
        
        # Get selected symptoms
        selected_symptoms = [var.get() for var in self.symptom_vars 
                           if var.get() != "None"]
        
        if len(selected_symptoms) == 0:
            messagebox.showwarning("⚠ Warning", "Please select at least one symptom!")
            return
        
        # Create feature vector
        feature_vector = [1 if symptom in selected_symptoms else 0 for symptom in l1]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Make predictions
        try:
            dt_pred = self.dt_model.predict(feature_vector)[0]
            rf_pred = self.rf_model.predict(feature_vector)[0]
            nb_pred = self.nb_model.predict(feature_vector)[0]
            
            predictions = [dt_pred, rf_pred, nb_pred]
            
            # Update result labels
            colors = ["#3498DB", "#9B59B6", "#E67E22"]
            for i, pred in enumerate(predictions):
                if 0 <= pred < len(disease):
                    self.result_labels[i].config(
                        text=disease[pred], 
                        fg=colors[i],
                        font=("Arial", 10, "bold")
                    )
                else:
                    self.result_labels[i].config(text="Unknown", fg="#E74C3C")
            
            # Check consensus
            unique_predictions = set(predictions)
            if len(unique_predictions) == 1:
                self.consensus_label.config(
                    text=f"✅ All models agree: {disease[predictions[0]]}",
                    fg="#27AE60"
                )
            else:
                # Find most common prediction
                pred_counts = {p: predictions.count(p) for p in unique_predictions}
                most_common = max(pred_counts, key=pred_counts.get)
                
                if pred_counts[most_common] >= 2:
                    self.consensus_label.config(
                        text=f"⚡ Majority prediction: {disease[most_common]}",
                        fg="#F39C12"
                    )
                else:
                    self.consensus_label.config(
                        text="⚠ No consensus - consult a doctor",
                        fg="#E74C3C"
                    )
                    
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")
    
    def clear_fields(self):
        """Clear all input fields and results"""
        self.name_var.set("")
        for var in self.symptom_vars:
            var.set("None")
        for label in self.result_labels:
            label.config(text="No prediction yet", fg="#BDC3C7", 
                        font=("Arial", 10))
        self.consensus_label.config(text="")

# Run the application
if __name__ == "__main__":
    root = Tk()
    app = DiseasePredictorApp(root)
    root.mainloop()