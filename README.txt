AI Skin Disease Detection Project – Final Submission
===================================================

Project Purpose:
- Real-time skin disease detection using webcam.
- Multi-disease classification using YOLO/CNN model.
- Ayurvedic treatment suggestion + doctor info.
- Voice output per detected disease.
- PDF report auto-save per session.

How to Run:
1. Ensure Python 3.11+ installed.
2. Install dependencies:
   pip install -r requirements.txt
3. Open Command Prompt / Terminal in project folder.
4. Run project:
   python camera_gui_multi_disease_pdf_voice.py
5. Camera will open, detect disease, voice readout will play, and PDF will auto-save.
6. Camera closes automatically after 10 seconds. Last frame remains visible.

Folder Structure:
- camera_gui_multi_disease_pdf_voice.py  --> Main code
- best.pt                               --> Trained YOLO model
- reports/                               --> Auto-saved PDF reports
- requirements.txt                        --> Python dependencies
- README.txt                              --> This file

Notes:
- Ensure 'best.pt' and code file are in the same folder.
- Reports folder will be created automatically if not present.
- Only Ayurvedic treatment suggestions included.