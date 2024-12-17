# **Image-Recognition**  
A system to recognize known people stored in a database using images or video streams.

---

## **Project Overview**  

This project implements an **image recognition system** that can identify known individuals by comparing faces in an image or video stream against a database of pre-stored people. It leverages advanced machine learning techniques and computer vision libraries to ensure accurate recognition and real-time processing.

---

## **Features**  

- **Face Recognition**: Detect and recognize faces in images or videos.  
- **Database Integration**: Store and manage information of known individuals in a database.  
- **Real-Time Detection**: Process video streams and images for live recognition.  
- **Scalability**: Easily add or remove individuals from the database.  

---

## **Tech Stack**  

- **Programming Language**: Python  
- **Libraries/Frameworks**:  
   - OpenCV (Image and Video Processing)  
   - Face Recognition (Dlib-based library for face comparison)  
   - NumPy (Data Processing)  
   - Flask (Web Interface - Optional)  
- **Database**: Excel and folder(for storing known individuals' information)  

---

## **Installation**

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/image-recognition.git
   cd image-recognition
   ```
   
4. **Set Up the Database**  
   - Create a database for storing known people.
   - Use a preloaded script or migration command to initialize tables.

5. **Run the Application**  
   - For command-line usage:
     ```bash
     python app.py

---

## **Usage**  

1. **Add Known Individuals**:  
   - Upload images of known people and store them in the database using the provided interface or script.

2. **Recognize Faces**:  
   - Run the program to scan an input image or video.  
   - The system will detect and identify people if their data is available in the database.  

3. **Supported Input Formats**:  
   - Image: `.jpg`, `.png`, `.jpeg`  
   - Video: `.mp4`, `.avi`, etc.  

---


## **License**  

This project is licensed under the MIT License. See the `LICENSE` file for more details.
