import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from fpdf import FPDF
import pyttsx3
from ultralytics import YOLO
import cv2
import threading
import datetime
import numpy as np

# Load YOLO model
model = YOLO("best.pt")

# Ayurvedic treatment mapping
treatment_mapping = {
    "Atopic Dermatitis": {
        "Doctor": "Dr. Vaidyanathan, MD (Ayurveda)",
        "Clinic": "Chennai Ayurveda Clinic",
        "Treatment": "Aloe vera gel, Turmeric milk",
        "Child": "Yes under supervision",
    },
    "Vascular lesion": {
        "Doctor": "Dr. Anitha, BAMS",
        "Clinic": "Trichy Ayurvedic Clinic",
        "Treatment": "Neem paste, Coconut oil",
        "Child": "Yes under supervision",
    },
    "Melanocytic nevus": {
        "Doctor": "Dr. Ramesh, MD (Ayurveda)",
        "Clinic": "Tanjore Ayurveda Clinic",
        "Treatment": "Aloe vera gel, Herbal ointment",
        "Child": "Yes under supervision",
    },
    # Add more diseases here
}

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Function to generate PDF
def generate_pdf(disease, confidence, treatment_info):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/skin_report_{now}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Skin Disease Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Disease Detected: {disease}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 10, f"Doctor: {treatment_info['Doctor']}", ln=True)
    pdf.cell(0, 10, f"Clinic: {treatment_info['Clinic']}", ln=True)
    pdf.cell(0, 10, f"Ayurvedic Treatment: {treatment_info['Treatment']}", ln=True)
    pdf.cell(0, 10, f"Child Usage: {treatment_info['Child']}", ln=True)
    pdf.output(filename)
    return filename

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Upload Image Mode
def upload_image_mode():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    img = Image.open(file_path)
    img_resized = img.resize((224, 224))
    photo = ImageTk.PhotoImage(img_resized)
    image_label.config(image=photo)
    image_label.image = photo

    # Model prediction
    results = model.predict(img_resized)
    top1 = results[0].probs.top1
    disease = results[0].names[top1]
    confidence = results[0].probs[top1] * 100

    # Get treatment info
    treatment_info = treatment_mapping.get(disease, {
        "Doctor": "N/A",
        "Clinic": "N/A",
        "Treatment": "N/A",
        "Child": "N/A"
    })

    # Update GUI
    result_text.set(f"Disease: {disease}\nConfidence: {confidence:.2f}%\nDoctor: {treatment_info['Doctor']}\nTreatment: {treatment_info['Treatment']}")

    # Voice
    voice_text = f"The detected disease is {disease} with confidence {confidence:.2f} percent. Ayurvedic treatment suggested is {treatment_info['Treatment']}."
    speak(voice_text)

    # Generate PDF
    pdf_file = generate_pdf(disease, confidence, treatment_info)
    messagebox.showinfo("PDF Saved", f"Report saved as {pdf_file}")

# Camera Mode
def camera_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        return

    def process_frame():
        start_time = datetime.datetime.now()
        while (datetime.datetime.now() - start_time).seconds < 10:  # 10 sec auto stop
            ret, frame = cap.read()
            if not ret:
                continue
            # Convert to PIL image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_resized = img_pil.resize((224, 224))
            photo = ImageTk.PhotoImage(img_resized)
            image_label.config(image=photo)
            image_label.image = photo

            # Model prediction
            results = model.predict(img_resized)
            top1 = results[0].probs.top1
            disease = results[0].names[top1]
            confidence = results[0].probs[top1] * 100

            # Treatment info
            treatment_info = treatment_mapping.get(disease, {
                "Doctor": "N/A",
                "Clinic": "N/A",
                "Treatment": "N/A",
                "Child": "N/A"
            })

            # Update GUI
            result_text.set(f"Disease: {disease}\nConfidence: {confidence:.2f}%\nDoctor: {treatment_info['Doctor']}\nTreatment: {treatment_info['Treatment']}")

            # Voice
            voice_text = f"The detected disease is {disease} with confidence {confidence:.2f} percent. Ayurvedic treatment suggested is {treatment_info['Treatment']}."
            speak(voice_text)

        cap.release()
        messagebox.showinfo("Info", "Camera session finished")

    threading.Thread(target=process_frame).start()

# Tkinter GUI
root = tk.Tk()
root.title("Skin Disease Detection")
root.geometry("650x550")

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Use Camera", command=camera_mode, width=20, bg="lightblue").pack(side="left", padx=10)
tk.Button(btn_frame, text="Upload Image", command=upload_image_mode, width=20, bg="lightgreen").pack(side="left", padx=10)

# Image display
image_label = tk.Label(root)
image_label.pack(pady=10)

# Result display
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, justify="left")
result_label.pack(pady=10)

root.mainloop()