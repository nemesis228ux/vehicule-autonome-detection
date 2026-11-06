# Détecteur d'objets routiers critiques pour véhicule semi-autonome

> **Projet Big Data – IAI Togo**  
> **Sujet :** L'intelligence artificielle et le véhicule autonome

---

## 1. Description du projet

Ce projet développe un système de perception capable de détecter en temps réel des objets routiers critiques dans un contexte africain (notamment togolais) et de déclencher des alertes selon le niveau de danger.

### Objets détectés
- Piétons
- Motos / deux-roues (*zemidjans*)
- Véhicules
- Nids-de-poule

### Fonctionnalités
1. Détection d'objets avec **YOLOv8**
2. Tracking des objets dans le temps
3. Évaluation du niveau de danger (*Critique / Attention / Information*)
4. Alertes visuelles et sonores différenciées

Le système illustre la première brique d'un véhicule autonome (**la Perception**) et met en avant les défis spécifiques aux routes africaines (nids-de-poule, forte présence de deux-roues, signalisation limitée, etc.).

---

## 2. Architecture du projet

```text
vehicule-autonome-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── videos/          # Vidéos de test (Lomé, routes dégradées, etc.)
│   └── images/          # Images de test (optionnel)
│
├── models/              # Modèles YOLO (.pt)
│
├── src/
│   ├── video_reader.py  # Lecture vidéo / webcam et fourniture des images
│   ├── detector.py      # Détection avec YOLO
│   ├── tracker.py       # Suivi des objets
│   ├── risk_evaluator.py# Évaluation du danger
│   ├── alerts.py        # Alertes visuelles + sonores
│   └── main.py          # Point d'entrée principal
│
└── outputs/             # Résultats (vidéos annotées, captures)
```

---

## 3. Outils et technologies

| Outil / Bibliothèque | Rôle |
| :--- | :--- |
| **Python 3.9+** | Langage principal |
| **Ultralytics YOLOv8** | Détection d'objets |
| **OpenCV** | Lecture vidéo, affichage, dessin |
| **NumPy** | Calculs numériques |
| **Pillow** | Gestion d'images |
| **playsound** ou **pygame** | Alertes sonores |

---

## 4. Installation

### Prérequis
- Python 3.9 ou supérieur
- Git
- *(Recommandé)* Un environnement virtuel

### Étapes

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/VOTRE_ORGANISATION/vehicule-autonome-detection.git
   cd vehicule-autonome-detection
   ```

2. **Créer un environnement virtuel :**
   ```bash
   python -m venv venv
   ```

3. **Activer l'environnement virtuel :**
   - **Sur Windows :**
     ```bash
     venv\Scripts\activate
     ```
   - **Sur Linux / macOS :**
     ```bash
     source venv/bin/activate
     ```

4. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Lancement du programme

### Utiliser directement le lecteur vidéo

Le lecteur fournit les images au format BGR utilisé par OpenCV. Il accepte un chemin
de fichier, une URL de flux vidéo ou un identifiant de caméra :

```python
from src.video_reader import VideoReader

with VideoReader("data/videos/lome.mp4") as reader:
    print(f"Résolution : {reader.frame_width}x{reader.frame_height}")
    for frame in reader:
        # Transmettre frame au détecteur ou à l'affichage OpenCV
        pass
```

La lecture d'un fichier s'arrête à la fin de la vidéo. Une caméra ou un flux en direct
qui ne peut plus fournir d'image déclenche une erreur ; le lecteur libère la ressource
à la fin de l'itération ou à la sortie du bloc `with`.

### Détection sur une vidéo
```bash
python src/main.py --source data/videos/votre_video.mp4
```

### Détection en direct (webcam ou téléphone via DroidCam / Iriun)
```bash
python src/main.py --source 0
```
> **Note :** `0` correspond généralement à la webcam par défaut. Si vous utilisez DroidCam ou Iriun, le numéro peut être différent (`1`, `2`, etc.).

### Options utiles (exemples)
```bash
# Sauvegarder la vidéo annotée
python src/main.py --source data/videos/lome.mp4 --save

# Afficher les FPS
python src/main.py --source data/videos/lome.mp4 --show-fps
```

---

## 6. Niveaux d'alerte

| Niveau | Couleur | Signification | Exemple |
| :--- | :---: | :--- | :--- |
| **Critique** | 🔴 Rouge | Danger immédiat | Piéton ou moto très proche |
| **Attention** | 🟠 Orange | Situation à surveiller | Nid-de-poule devant, objet moyen |
| **Information** | 🟢 Vert / 🔵 Bleu | Présence détectée mais non urgente | Objets loin ou sur les côtés |

---

## 7. Organisation du travail en groupe

| Partie | Fichier(s) concerné(s) | Responsable possible |
| :--- | :--- | :--- |
| **Lecture vidéo** | `video_reader.py` | Membre 1 |
| **Détection YOLO** | `detector.py` | Membre 1 ou 2 |
| **Tracking** | `tracker.py` | Membre 2 |
| **Évaluation du danger** | `risk_evaluator.py` | Membre 2 ou 3 |
| **Alertes** | `alerts.py` | Membre 3 |
| **Intégration + démo** | `main.py` + tests | Tout le monde |

> Chaque membre peut travailler sur une branche Git séparée (`feature/detection`, `feature/alerts`, etc.).

---

## 8. Données de test recommandées

- Vidéos de conduite à Lomé / Togo (YouTube ou filmées par le groupe)
- Vidéos contenant des nids-de-poule
- Scènes avec beaucoup de zemidjans et de piétons

Placez vos vidéos dans le dossier `data/videos/`.

---

## 9. Auteurs

Projet réalisé dans le cadre du cours de **Big Data – IAI Togo**.  
*Année académique 2025-2026*

---

## 10. Licence

Ce projet est réalisé à des fins pédagogiques.