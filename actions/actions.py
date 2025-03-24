import json
import os
import re
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

from thefuzz import process, fuzz


# Helper function to load JSON data
def load_json(filename: str):
    file_path = os.path.join("json_data", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

class ActionLogin(Action):
    def name(self) -> Text:
        return "action_login"

    async def run(
            self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[Dict[Text, Any]]:
        # Try to collect worker_id and password from the user (or from slot).
        worker_id = tracker.get_slot("worker_id")
        password = tracker.get_slot("password")

        if not worker_id:
            # We haven't gotten the worker ID yet
            dispatcher.utter_message(text="Please provide your worker ID.")
            return []
        if not password:
            # We haven't gotten the password yet
            dispatcher.utter_message(text="Please provide your password.")
            return []

        # Load worker informations
        workers = load_json("worker_informations.json")

        # Validate
        station_number = None
        for w in workers:
            if w["worker_id"] == worker_id and w["password"] == password:
                station_number = w["station_number"]
                break

        if station_number is None:
            dispatcher.utter_message(response="utter_login_failure")
            return []

        dispatcher.utter_message(response="utter_login_success")
        return [SlotSet("station_number", str(station_number))]

class ActionShowDailyReport(Action):
    def name(self) -> Text:
        return "action_show_daily_report"

    async def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        patient_name_slot = tracker.get_slot("patient_name")
        if not patient_name_slot:
            dispatcher.utter_message(text="Which patient are you referring to?")
            return []

        user_input_lower = patient_name_slot.lower().strip()

        patients_data = load_json("patient_datas.json")

        all_names = [p["name"] for p in patients_data]
        all_names_lower = [n.lower().strip() for n in all_names]

        # 5) Fuzzy match
        match, score = process.extractOne(
            user_input_lower,
            all_names_lower,
            scorer=fuzz.token_set_ratio
        )

        THRESHOLD = 70
        if match and score >= THRESHOLD:
            index = all_names_lower.index(match)
            matched_patient = patients_data[index]

            # 6) Retrieve the daily_report from matched_patient
            daily_report = matched_patient.get("daily_report", [])
            if not daily_report:
                dispatcher.utter_message(text=f"No daily report found for {matched_patient['name']}.")
                return []

            # 7) Construct a message from each day's report
            response_lines = []
            response_lines.append(f"Daily report for {matched_patient['name']}:")
            for i, report in enumerate(daily_report, start=1):
                date = report.get("date", "Unknown date")
                notes = report.get("notes", "No notes")
                dinner = report.get("dinner", "unknown")
                lunch = report.get("lunch", "unknown")
                breakfast = report.get("breakfast", "unknown")
                drinking_amount = report.get("drinking_amount", "N/A")

                response_lines.append(
                    f"\nEntry {i}: [Date: {date}]\n"
                    f"  Notes: {notes}\n"
                    f"  Breakfast: {breakfast}, Lunch: {lunch}, Dinner: {dinner}\n"
                    f"  Drinking amount: {drinking_amount} ml"
                )

            # 8) Send it to user
            dispatcher.utter_message(text="\n".join(response_lines))
        else:
            # If no fuzzy match above threshold
            dispatcher.utter_message(text="No matching patient found.")

        return []
####################################################personal_details#####################################################################

class ActionShowPatientPersonalDetails(Action):
    def name(self) -> Text:
        return "action_show_patient_personal_details"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        # 1) Get the patient's name from the slot
        patient_name_slot = tracker.get_slot("patient_name")
        if not patient_name_slot:
            dispatcher.utter_message(text="Which patient are you asking about?")
            return []

        # 2) Convert to lowercase for matching
        user_input_lower = patient_name_slot.lower().strip()

        # 3) Load patient data
        patients = load_json("patient_datas.json")

        # 4) Gather all patient names
        all_names = [p["name"] for p in patients]
        all_names_lower = [n.lower().strip() for n in all_names]

        # 5) Fuzzy match the user input
        match, score = process.extractOne(
            user_input_lower,
            all_names_lower,
            scorer=fuzz.token_set_ratio
        )

        # 6) Set a threshold
        THRESHOLD = 70
        if match and score >= THRESHOLD:
            index = all_names_lower.index(match)
            matched_patient = patients[index]

            # 7) Extract personal details.
            #    If a field is missing, use .get() with a default
            name = matched_patient.get("name", "Unknown")
            age = matched_patient.get("age", "Unknown")
            gender = matched_patient.get("gender", "Unknown")
            room_number = matched_patient.get("room_number", "Unknown")
            dementia = matched_patient.get("dementia", "Unknown")
            emergency_contact = matched_patient.get("emergency_contact", "Unknown")
            address_list = matched_patient.get("adress", [])
            address_str = "Unknown"
            if address_list and isinstance(address_list[0], dict):
                addr = address_list[0]
                # e.g. "Mainstreet,10, 10001, New York"
                address_str = f"{addr.get('road','')}, {addr.get('postcode','')}, {addr.get('and','')}"

            # 8) Construct the response
            response_msg = (
                f"Personal details for {name}:\n\n"
                f"Age: {age}\n"
                f"Gender: {gender}\n"
                f"Address: {address_str}\n"
                f"Emergency Contact: {emergency_contact}\n"
                f"Room Number: {room_number}\n"
                f"Dementia: {dementia}"
            )

            dispatcher.utter_message(text=response_msg)
        else:
            # If no match or below threshold
            dispatcher.utter_message(text="No matching patient found with that name.")

        return []


#####################################################medical_conditation#########################
class ActionShowPatientMedicalDetails(Action):
    def name(self) -> Text:
        return "action_show_patient_medical_details"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        patient_name_slot = tracker.get_slot("patient_name")
        if not patient_name_slot:
            dispatcher.utter_message(text="Which patient are you referring to?")
            return []

        user_input_lower = patient_name_slot.lower().strip()
        patients_data = load_json("patient_datas.json")
        all_names = [p["name"] for p in patients_data]
        all_names_lower = [n.lower().strip() for n in all_names]

        # 5) fuzzy match
        match, score = process.extractOne(
            user_input_lower,
            all_names_lower,
            scorer=fuzz.token_set_ratio
        )

        # 6) threshold
        THRESHOLD = 70
        if match and score >= THRESHOLD:
            index = all_names_lower.index(match)
            matched_patient = patients_data[index]

            # 7) Extract medical fields
            blood_type = matched_patient.get("blood_type", "Unknown")
            medical_condition = matched_patient.get("medical_condition", "Unknown")
            doctor = matched_patient.get("doctor", "Unknown")
            insurance = matched_patient.get("insurance_Provider", "Unknown")
            admission_type = matched_patient.get("admission_type", "Unknown")

            # medication might be a list of strings
            meds_list = matched_patient.get("medication", [])
            meds_str = ", ".join(meds_list) if meds_list else "None listed"

            # 8) Construct the response
            response = (
                f"Medical details for {matched_patient['name']}:\n\n"
                f"Blood Type: {blood_type}\n"
                f"Medical Condition: {medical_condition}\n"
                f"Doctor: {doctor}\n"
                f"Insurance Provider: {insurance}\n"
                f"Admission Type: {admission_type}\n"
                f"Current Medication: {meds_str}"
            )

            dispatcher.utter_message(text=response)
        else:
            # no good match
            dispatcher.utter_message(text="No matching patient found with that name.")

        return []
########################################################################################
class ActionShowPatientList(Action):
    def name(self) -> Text:
        return "action_show_all_patients_station"

    async def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        station_number = tracker.get_slot("station_number")

        # If the user typed "show me my patient list in station 5"
        # Rasa should have station_number="5"
        if not station_number:
            dispatcher.utter_message(text="No station number found in your message.")
            return []

        patients_data = load_json("patient_datas.json")

        # Filter patient_datas by station
        matched = [
            p for p in patients_data
            if p.get("station_number", "").strip() == station_number.strip()
        ]
        if not matched:
            dispatcher.utter_message(
                text=f"No patients found for station {station_number}."
            )
            return []

        names = [p["name"] for p in matched]
        msg = f"Patients in station {station_number}:\n" + "\n".join(names)
        dispatcher.utter_message(text=msg)

        return []
#
#########################################################################################################################
class ActionShowAllPatients(Action):
    def name(self) -> Text:
        return "action_show_all_patients"

    async def run(self,dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        patients_data = load_json("patient_datas.json")
        names = [p["name"] for p in patients_data]
        if not names:
            dispatcher.utter_message(text="No patients found.")
        else:
            msg = "Here are all the patients:\n" + "\n".join(names)
            dispatcher.utter_message(text=msg)

        return []


###############################################patientList##############################################################
class ActionShowPatientList(Action):
    def name(self) -> Text:
        return "action_show_patient_list"
    async def run( self,dispatcher: CollectingDispatcher, tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        station_number = tracker.get_slot("station_number")

        if not station_number:
            # We have no station number yet, ask the user
            dispatcher.utter_message(response="utter_ask_station_number")
            return []

        # We do have a station number, so let's filter
        patients_data = load_json("patient_datas.json")

        # Filter
        matched = [
            p for p in patients_data
            if p.get("station_number", "").strip() == station_number.strip()
        ]

        if not matched:
            dispatcher.utter_message(
                text=f"No patients found for station {station_number}."
            )
            return []

        # Gather the patient names
        names = [m["name"] for m in matched]
        message = f"Patients in station {station_number}:\n" + "\n".join(names)
        dispatcher.utter_message(text=message)

        # Optionally reset the slot or keep it
        # return [SlotSet("station_number", None)]
        return []

#######################################################################################################################
class ActionShowMedicationPlan(Action):
    def name(self) -> Text:
        return "action_show_medication_plan"

    async def run(
        self,dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        # 1) Retrieve the user's patient name slot
        patient_name_slot = tracker.get_slot("patient_name")
        print(f"DEBUG: patient_name_slot = {patient_name_slot}")  # Debug print

        # If the user didn't provide any name at all
        if not patient_name_slot:
            dispatcher.utter_message(text="Which patient are you referring to?")
            return []

        # 2) Convert user input to lowercase (for case-insensitive matching)
        patient_name_lower = patient_name_slot.lower().strip()
        print(f"DEBUG: patient_name_lower = {patient_name_lower}")  # Debug print

        # 3) Load your medication data
        meds_data = load_json("medication.json")

        # 4) Build a list of known patient names for fuzzy matching
        all_names = [m["Name"] for m in meds_data]
        all_names_lower = [n.lower().strip() for n in all_names]
        print(f"DEBUG: all_names_lower = {all_names_lower}")  # Debug print

        # 5) Perform fuzzy matching
        match, score = process.extractOne(
            patient_name_lower,
            all_names_lower,
            scorer=fuzz.token_set_ratio
        )
        print(f"DEBUG: best match = {match}, score = {score}")  # Debug print

        # 6) Decide if match is good enough
        THRESHOLD = 70  # Adjust to your preference
        if match and score >= THRESHOLD:
            # Find the original medication entry by index
            index = all_names_lower.index(match)
            matched_entry = meds_data[index]

            # 7) Build the medication plan message
            response_lines = []
            response_lines.append(
                f"Medication Plan for {matched_entry['Name']} (schedule: {matched_entry.get('schedule','N/A')})"
            )
            # Include every key except _id, Name, schedule
            for key, value in matched_entry.items():
                if key not in ["_id", "Name", "schedule"]:
                    response_lines.append(f"{key}: {value}")

            dispatcher.utter_message(text="\n".join(response_lines))

        else:
            # If no match or below threshold
            dispatcher.utter_message(text="No medication plan found for that patient.")

        return []
########################################----disease_precaution-------###########################################################

class ActionShowDiseasePrecaution(Action):
    def name(self) -> Text:
        return "action_show_disease_precaution"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        # 1. Retrieve the user's slot value for disease_name
        disease_slot_value = tracker.get_slot("disease_name")
        if not disease_slot_value:
            dispatcher.utter_message(text="Which disease are you interested in for precautions?")
            return []

        # 2. Clean up user input (lowercase, strip, etc.)
        disease_slot_lower = disease_slot_value.lower().strip()

        # 3. Load the precaution data
        precaution_data = load_json("disease_precaution.json")
        # 4. Gather the disease names in a list for fuzzy matching
        all_diseases = [entry["Disease"] for entry in precaution_data]
        all_diseases_lower = [d.lower().strip() for d in all_diseases]

        # 5. Perform fuzzy matching between the user text and known diseases
        match, score = process.extractOne(
            disease_slot_lower,
            all_diseases_lower,
            scorer=fuzz.token_set_ratio
        )

        # 6. Set a threshold for a "good enough" match
        THRESHOLD = 60
        if match and score >= THRESHOLD:
            # Find the matched disease in the original data
            index = all_diseases_lower.index(match)
            matched_entry = precaution_data[index]

            # 7. Build a response string
            precaution_msg = (
                f"Disease: {matched_entry['Disease']}\n"
                f"Precaution 1: {matched_entry.get('Precaution_1', 'No data')}\n"
                f"Precaution 2: {matched_entry.get('Precaution_2', 'No data')}\n"
                f"Precaution 3: {matched_entry.get('Precaution_3', 'No data')}\n"
                f"Precaution 4: {matched_entry.get('Precaution_4', 'No data')}"
            )
            dispatcher.utter_message(text=precaution_msg)
        else:
            # No good match found or below the threshold
            dispatcher.utter_message(text="No precaution data found for that disease.")

        return []

##########################################--disease_info--#########################################################



class ActionShowDiseaseInfo(Action):
    def name(self) -> Text:
        return "action_show_disease_info"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        disease_slot_value = tracker.get_slot("disease_name")
        if not disease_slot_value:
            dispatcher.utter_message(text="Which disease are you interested in?")
            return []

        user_text_lower = disease_slot_value.lower().strip()

        common_phrases = [
            "i want to know about",
            "i want to learn about",
            "can you tell me about",
            "tell me about",
            "know about",
            "learn about"
        ]
        for phrase in common_phrases:
            user_text_lower = user_text_lower.replace(phrase, "")

        # Clean up any leftover extra spaces
        user_text_lower = user_text_lower.strip()

        # 4) Load  JSON data for diseases
        disease_data = load_json("disease_description.json")

        all_diseases = [d["Disease"] for d in disease_data]
        all_diseases_lower = [d.lower().strip() for d in all_diseases]

        # 6) Fuzzy match the user input against the list of known disease names
        match, score = process.extractOne(
            user_text_lower,
            all_diseases_lower,
            scorer=fuzz.token_set_ratio
        )

        # 7) Decide if the match is "good enough" by checking the score
        # Adjust 60 → 70+ if you want a stricter match
        threshold = 60
        if match and score >= threshold:
            # Find the original disease data by index
            index = all_diseases_lower.index(match)
            matched_disease = disease_data[index]

            disease_name = matched_disease["Disease"]
            description = matched_disease["Description"]

            # 8) Respond with the matched disease info
            dispatcher.utter_message(
                text=f"Disease: {disease_name}\n\nDescription:\n{description}"
            )
        else:
            # If no good match, inform the user
            dispatcher.utter_message(text="No description found for that disease.")

        return []

############################################################################################

class ActionPredictDisease(Action):
    def name(self) -> Text:
        return "action_predict_disease"

    async def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        print("DEBUG: Inside action_predict_disease")
        user_symptoms = tracker.get_slot("symptoms")
        print(user_symptoms)

        # If the slot is a string, convert it to a list by splitting on commas.
        if isinstance(user_symptoms, str):
            user_symptoms = [s.strip() for s in user_symptoms.split(",")]

        # 2. Load the disease dataset (disease.json)
        # The file should contain a list of disease records.
        # Each record should include a "Disease" field and multiple "Symptom_x" fields.
        disease_data = load_json("disease.json")

        # 3. Build a set of all unique symptoms from the dataset.
        all_symptoms = set()
        for item in disease_data:
            for key, value in item.items():
                if "Symptom" in key and value:
                    all_symptoms.add(value.strip().lower())
        all_symptoms = sorted(list(all_symptoms))

        # 4. Prepare training data: build a binary feature vector for each disease record.
        X = []
        y = []
        for item in disease_data:
            disease_name = item["Disease"]
            # Create a binary vector indicating the presence of each unique symptom.
            symptom_vector = [0] * len(all_symptoms)
            for key, value in item.items():
                if "Symptom" in key and value:
                    symptom_value = value.strip().lower()
                    if symptom_value in all_symptoms:
                        index = all_symptoms.index(symptom_value)
                        symptom_vector[index] = 1
            X.append(symptom_vector)
            y.append(disease_name)

        # Convert lists to numpy arrays.
        X = np.array(X)
        y = np.array(y)

        # 5. Train multiple ML models.
        dt_clf = DecisionTreeClassifier().fit(X, y)
        rf_clf = RandomForestClassifier().fit(X, y)
        nb_clf = GaussianNB().fit(X, y)
        knn_clf = KNeighborsClassifier().fit(X, y)
        svc_clf = SVC().fit(X, y)


        # 6. Build the feature vector for the user's symptoms.
        user_vector = [0] * len(all_symptoms)
        for sym in user_symptoms:
            sym_lower = sym.lower()
            if sym_lower in all_symptoms:
                index = all_symptoms.index(sym_lower)
                user_vector[index] = 1
        user_vector = np.array([user_vector])

        # 7. Make predictions using each classifier.
        pred_dt = dt_clf.predict(user_vector)[0]
        pred_rf = rf_clf.predict(user_vector)[0]
        pred_nb = nb_clf.predict(user_vector)[0]
        pred_knn = knn_clf.predict(user_vector)[0]
        pred_svc = svc_clf.predict(user_vector)[0]
        print("result")
        print(pred_dt)
        dispatcher.utter_message(
            text=#f"Prediction-1 disease: {pred_dt},"
                 f" Prediction-2 disease: {pred_rf},"
                # f"  Prediction-2 disease: {pred_nb},"
                # f" Prediction-2 disease: {pred_knn}"
                # f" Prediction-2 disease: {pred_svc}"
        )
        return []