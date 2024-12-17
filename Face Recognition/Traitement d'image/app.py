import os
import numpy as np
from flask import Flask, request, render_template, url_for, flash, redirect
from werkzeug.utils import secure_filename
import cv2
import face_recognition
import pickle
import csv
from datetime import datetime

# Configuration de l'application
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"}
app.secret_key = 'your_secret_key_here'

# Fonction pour vérifier les extensions de fichiers
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# Sauvegarder les visages et empreintes dans un fichier pickle
def sauvegarder_visages(noms_visages, empreintes_visages):
    with open('chemin_vers_votre_fichier_encodages.pkl', 'wb') as f:
        pickle.dump((noms_visages, empreintes_visages), f)

# Charger les visages enregistrés depuis le fichier pickle
def charger_visages_pickle():
    try:
        with open('chemin_vers_votre_fichier_encodages.pkl', 'rb') as f:
            noms_visages, empreintes_visages = pickle.load(f)
            return noms_visages, empreintes_visages
    except FileNotFoundError:
        return [], []

# Charger les visages enregistrés et leurs noms
def charger_visages_enregistres(dossier_visages):
    noms_visages, empreintes_visages = charger_visages_pickle()

    for fichier in os.listdir(dossier_visages):
        chemin = os.path.join(dossier_visages, fichier)
        if fichier.endswith(".jpg") or fichier.endswith(".png"):
            image = face_recognition.load_image_file(chemin)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                empreintes_visages.append(encodings[0])
                noms_visages.append(os.path.splitext(fichier)[0])

    return noms_visages, empreintes_visages

# Fonction pour enregistrer l'historique dans un fichier CSV
def enregistrer_historique(nom, source):
    with open('historique_reconnaissances.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([nom, date, source])

# Route principale - Page d'accueil avec deux boutons
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

# Route pour la reconnaissance d'image
@app.route("/reconnaissance_image", methods=["GET", "POST"])
def reconnaissance_image():
    uploaded_file_url = None
    resultats = []
    bouton_affiche = False

    if request.method == "POST":
        if "file" not in request.files:
            return "Aucun fichier sélectionné !", 400
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            path = "C:/Users/brice/Downloads/Traitement d'image/Traitement d'image/visage_enregistrés/"
            noms_visages, empreintes_visages = charger_visages_enregistres(path)

            image = face_recognition.load_image_file(filepath)
            locations_visages = face_recognition.face_locations(image)
            encodings_visages = face_recognition.face_encodings(image, locations_visages)

            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            for (top, right, bottom, left), enc in zip(locations_visages, encodings_visages):
                correspondances = face_recognition.compare_faces(empreintes_visages, enc)
                distances = face_recognition.face_distance(empreintes_visages, enc)

                if len(distances) > 0:
                    meilleur_match_index = np.argmin(distances)
                    if correspondances[meilleur_match_index]:
                        nom = noms_visages[meilleur_match_index]
                    else:
                        nom = "Inconnu"
                        bouton_affiche = True
                else:
                    nom = "Inconnu"
                    bouton_affiche = True

                resultats.append(nom)

                # Enregistrer dans l'historique
                enregistrer_historique(nom, "Image")

                cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(image_bgr, nom, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            processed_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_" + filename)
            cv2.imwrite(processed_image_path, image_bgr)

            uploaded_file_url = url_for("static", filename="uploads/" + "processed_" + filename)

    return render_template("reconnaissance_image.html", file_url=uploaded_file_url, resultats=resultats, bouton_affiche=bouton_affiche)

# Route pour la reconnaissance vidéo

@app.route("/reconnaissance_video", methods=["GET", "POST"])
def reconnaissance_video():
    video_file_url = None
    video_results = []  # Liste pour stocker les frames traitées

    if request.method == "POST":
        # Vérifier si un fichier vidéo est téléchargé
        if "file" not in request.files:
            return "Aucun fichier sélectionné !", 400
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            print(f"Fichier sauvegardé : {filepath}")

            # Charger les visages enregistrés
            path = "C:/Users/brice/Downloads/Traitement d'image/Traitement d'image/visage_enregistrés/"
            noms_visages, empreintes_visages = charger_visages_enregistres(path)

            # Lire la vidéo
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return "Erreur lors de l'ouverture de la vidéo !", 400

            frame_count = 0
            video_frames = []
            max_frames = 100  # Limite le nombre de frames pour tester

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                print(f"Traitement de la frame {frame_count}")

                if frame_count % 10 == 0:  # Extraire une frame tous les 10 frames pour la reconnaissance
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locations_visages = face_recognition.face_locations(rgb_frame)
                    encodings_visages = face_recognition.face_encodings(rgb_frame, locations_visages)

                    print(f"Nombre de visages détectés : {len(locations_visages)}")

                    for (top, right, bottom, left), enc in zip(locations_visages, encodings_visages):
                        correspondances = face_recognition.compare_faces(empreintes_visages, enc)
                        distances = face_recognition.face_distance(empreintes_visages, enc)

                        if len(distances) > 0:
                            meilleur_match_index = np.argmin(distances)
                            if correspondances[meilleur_match_index]:
                                nom = noms_visages[meilleur_match_index]
                            else:
                                nom = "Inconnu"
                        else:
                            nom = "Inconnu"

                        # Dessiner un rectangle rouge autour du visage
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        # Ajouter le nom sous le rectangle
                        cv2.putText(frame, nom, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Sauvegarder la frame traitée
                    frame_filename = f"frame_{frame_count}.jpg"
                    frame_path = os.path.join(app.config["UPLOAD_FOLDER"], frame_filename)
                    cv2.imwrite(frame_path, frame)

                    # Ajouter l'URL de la frame à la liste
                    video_frames.append(url_for("static", filename="uploads/" + frame_filename))

            cap.release()
            video_file_url = video_frames

    return render_template("reconnaissance_video.html", video_frames=video_file_url)




# Fonction pour ajouter un visage
@app.route("/ajouter_visage", methods=["GET", "POST"])
def ajouter_visage():
    if request.method == "POST":
        nom = request.form["nom"]
        prenom = request.form["prenom"]
        is_celebrity = request.form["is_celebrity"]
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            path = "C:/Users/brice/Downloads/Traitement d'image/Traitement d'image/visage_enregistrés/"
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                nouveau_chemin = os.path.join(path, f"{nom}_{prenom}.jpg")
                cv2.imwrite(nouveau_chemin, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                noms_visages, empreintes_visages = charger_visages_enregistres(path)
                empreintes_visages.append(encodings[0])
                noms_visages.append(f"{nom} {prenom}")

                sauvegarder_visages(noms_visages, empreintes_visages)
                flash("Visage ajouté avec succès !")
                return redirect(url_for('reconnaissance_image'))

        flash("Erreur lors de l'ajout du visage")
        return redirect(url_for('ajouter_visage'))

    return render_template("ajouter_visage.html")






# Route pour afficher les visages enregistrés
@app.route("/voir_visages", methods=["GET"])
def voir_visages():
    # Charger les visages enregistrés depuis le fichier pickle
    noms_visages, empreintes_visages = charger_visages_pickle()

    # Charger les informations des visages (nom, prénom, célébrité, etc.)
    visages_info = []
    for nom in noms_visages:
        # Pour chaque visage, récupérer les informations associées (nom, prénom, célébrité)
        # Ici, nous utilisons des informations fictives pour l'exemple
        nom_prenom = nom.split(" ")
        is_celebrity = "Oui" if nom_prenom[0] == "Célébrité" else "Non"  # Exemple de condition fictive
        visages_info.append({
            "nom": nom_prenom[0],
            "prenom": nom_prenom[1] if len(nom_prenom) > 1 else "",
            "is_celebrity": is_celebrity,
            "photo": f"uploads/{nom}.jpg"  # Lien vers la photo enregistrée
        })

    return render_template("voir_visages.html", visages=visages_info)


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
