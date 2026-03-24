from ultralytics import YOLO
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from fpdf import FPDF
import datetime
import os
import pyttsx3

# Initialize voice engine
engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Load Classification Model
model = YOLO("runs/classify/train/weights/best.pt")

# Disease → Ayurvedic Treatment + Doctor Mapping
doctor_details = {
    "Vascular lesion": {
        "doctor": "Dr. Partap Chauhan, BAMS",
        "clinic": "Jiva Ayurveda, Faridabad",
        "treatment": "Herbal blood purifiers, Neem-based oils",
        "advantage": "Improves blood circulation naturally",
        "child_safe": "Yes, mild herbal treatment only"
    },
    "Atopic Dermatitis": {
        "doctor": "Dr. Vaidyanathan, MD (Ayurveda)",
        "clinic": "Chennai Ayurveda Clinic",
        "treatment": "Aloe vera gel, Turmeric milk",
        "advantage": "Reduces itching and dryness naturally",
        "child_safe": "Yes under supervision"
    },
    "Benign keratosis": {
        "doctor": "Dr. Gowthaman, BAMS",
        "clinic": "Sanjeevani Ayurveda, Chennai",
        "treatment": "Herbal skin detox oils",
        "advantage": "Prevents skin thickening naturally",
        "child_safe": "Consult before use"
    },
    "Melanocytic nevus": {
        "doctor": "Dr. Hariprasad, BAMS",
        "clinic": "Arya Vaidya Pharmacy, Coimbatore",
        "treatment": "Skin rejuvenation herbs",
        "advantage": "Maintains skin balance",
        "child_safe": "Yes with guidance"
    },
    "Melanoma": {
        "doctor": "Immediate Dermatologist Consultation Required",
        "clinic": "Refer Cancer Specialist",
        "treatment": "Emergency medical evaluation required",
        "advantage": "Early detection saves life",
        "child_safe": "Immediate hospital visit required"
    }
}

# Tkinter window
root = tk.Tk()
root.title("AI Multi-Disease Detection PDF + Voice")
root.geometry("950x600")

video_panel = tk.Label(root)
video_panel.pack(side="left")

info_panel = tk.Frame(root, width=350)
info_panel.pack(side="right", fill="both", expand=True)
info_text = tk.Text(info_panel, wrap="word", font=("Arial", 12))
info_text.pack(fill="both", expand=True)

cap = cv2.VideoCapture(0)

detected_diseases = {}  # Track multiple diseases
start_time = datetime.datetime.now()
voice_done_list = []  # Keep track of which diseases already voiced

# Function to save PDF
def save_pdf():
    if not detected_diseases:
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AI Multi-Disease Detection Report", ln=True, align="C")
    pdf.ln(10)

    for disease, details in detected_diseases.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Disease: {disease}", ln=True)
        pdf.set_font("Arial", '', 12)
        for key, value in details.items():
            pdf.multi_cell(0, 8, f"{key}: {value}")
        pdf.ln(4)

    if not os.path.exists("reports"):
        os.makedirs("reports")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/multi_disease_report_{timestamp}.pdf"
    pdf.output(filename)
    print(f"PDF saved: {filename}")

# Function to update frame
def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Predict
    results = model(frame)
    probs = results[0].probs
    class_id = probs.top1
    confidence = float(probs.top1conf)
    disease_name = model.names[class_id]

    # Update detected diseases dict
    if disease_name not in detected_diseases:
        detected_diseases[disease_name] = {"Confidence": f"{round(confidence*100,2)}%"}
        if disease_name in doctor_details:
            detected_diseases[disease_name].update(doctor_details[disease_name])

    # Voice per new disease only
    if disease_name not in voice_done_list and disease_name in doctor_details:
        speak_text = f"{disease_name} detected. Ayurvedic treatment: {doctor_details[disease_name]['treatment']}. Suggested doctor: {doctor_details[disease_name]['doctor']}."
        engine.say(speak_text)
        engine.runAndWait()
        voice_done_list.append(disease_name)

    # Update info panel
    info_text.delete(1.0, tk.END)
    for disease, details in detected_diseases.items():
        info_text.insert(tk.END, f"Disease: {disease}\n")
        for k, v in details.items():
            info_text.insert(tk.END, f"{k}: {v}\n")
        info_text.insert(tk.END, "\n")

    # Convert frame to Tkinter image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_panel.imgtk = imgtk
    video_panel.configure(image=imgtk)

    # Auto-close after 10 sec
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    if elapsed >= 10:
        save_pdf()
        close_app()
        return

    root.after(10, update_frame)

# Close function
def close_app():
    cap.release()
    root.destroy()

# Buttons
close_btn = tk.Button(info_panel, text="Close", command=close_app, bg="red", fg="white", font=("Arial", 12))
close_btn.pack(pady=5)

pdf_btn = tk.Button(info_panel, text="Save PDF Report", command=save_pdf, bg="green", fg="white", font=("Arial", 12))
pdf_btn.pack(pady=5)

# Start
update_frame()
root.mainloop()