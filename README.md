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
2. Lecture de vidéos, de caméras et de flux réseau
3. Suivi des objets avec **ByteTrack** : identifiants, vitesse et trajectoire
4. Évaluation monoculaire du risque (*Critique / Attention / Information*)
5. Affichage des détections, du suivi et du niveau de risque dans une fenêtre OpenCV

Le système illustre la première brique d'un véhicule autonome (**la Perception**) et met en avant les défis spécifiques aux routes africaines (nids-de-poule, forte présence de deux-roues, signalisation limitée, etc.).

> L'évaluation du risque est une estimation visuelle à partir des images, et non une mesure réelle de distance ou une garantie de sécurité. Les alertes actuellement implémentées sont visuelles ; aucune alerte sonore n'est intégrée.

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
│   ├── videos/          # Vidéos de test
│   └── images/          # Images de test
│
├── models/              # Modèles YOLO (.pt), dont yolov8n.pt
│
├── src/
│   ├── demo_detection.py # Point d'entrée : détection, suivi et risque
│   ├── demo_video.py     # Démonstration du lecteur vidéo seul
│   ├── video_reader.py   # Lecture vidéo, caméra et flux
│   ├── detector.py       # Détection avec YOLOv8
│   ├── tracker.py        # Suivi ByteTrack, vitesse et trajectoire
│   ├── risk_evaluator.py # Évaluation du danger
│   ├── hud.py            # Affichage du niveau de risque
│   └── window_utils.py   # Adaptation de la fenêtre à l'écran
│
├── tests/                # Tests unitaires
└── outputs/              # Vidéos annotées générées avec --save
```

---

## 3. Outils et technologies

| Outil / Bibliothèque | Rôle |
| :--- | :--- |
| **Python 3.9+** | Langage principal |
| **Ultralytics YOLOv8** | Détection d'objets |
| **ByteTrack** | Suivi des objets entre les images |
| **OpenCV** | Lecture vidéo, affichage, dessin |
| **NumPy** | Calculs numériques |
| **lap** | Association utilisée par le suivi |

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

Le point d'entrée de la démonstration complète est `src/demo_detection.py`.
Les commandes ci-dessous sont à lancer depuis la racine du projet.

### Détection sur une vidéo
```bash
python -m src.demo_detection --source data/videos/votre_video.mp4
```

### Détection en direct (webcam ou téléphone via DroidCam / Iriun)
```bash
python -m src.demo_detection --source 0
```
> **Note :** `0` correspond généralement à la webcam par défaut. Si vous utilisez DroidCam ou Iriun, le numéro peut être différent (`1`, `2`, etc.).

Un fichier vidéo, un identifiant de caméra ou une URL de flux peuvent être fournis avec `--source`.

### Suivi et évaluation du risque
```bash
# Activer le suivi ByteTrack et afficher les identifiants
python -m src.demo_detection --source data/videos/votre_video.mp4 --track

# Évaluer le risque et afficher le HUD (active aussi le suivi)
python -m src.demo_detection --source data/videos/votre_video.mp4 --risk

# Sauvegarder la vidéo annotée et afficher les FPS
python -m src.demo_detection --source data/videos/votre_video.mp4 --risk --save --show-fps

# Utiliser un autre modèle ou accélérer l'inférence sur CPU
python -m src.demo_detection --source 0 --model yolov8n.pt --imgsz 320
```

Les vidéos créées avec `--save` sont enregistrées dans `outputs/`. Quittez la fenêtre avec `q` ou Échap.
Pour tester uniquement la lecture d'une source vidéo, utilisez `python -m src.demo_video --source 0`.

---

## 6. Niveaux de risque

| Niveau | Couleur | Signification | Exemple |
| :--- | :---: | :--- | :--- |
| **Critique** | 🔴 Rouge | Risque élevé dans la trajectoire | Objet proche ou temps avant collision estimé court |
| **Attention** | 🟠 Orange | Situation à surveiller | Objet qui se rapproche ou usager vulnérable en traversée |
| **Information** | 🟢 Vert | Objet détecté sans risque immédiat estimé | Objet éloigné ou hors trajectoire |

Le calcul s'appuie sur la position et la taille des objets dans l'image, leur suivi entre les frames, une estimation de proximité et le temps avant collision (TTC). Il s'agit d'heuristiques monoculaires dépendantes de la caméra et de la scène, pas d'une distance physique calibrée. La classe « nid-de-poule » nécessite un modèle qui a été entraîné à la détecter ; le modèle YOLO général fourni peut ne pas la reconnaître.

---

## 7. Organisation du travail en groupe

| Partie | Fichier(s) concerné(s) | Responsable possible |
| :--- | :--- | :--- |
| **Lecture vidéo et fenêtre** | `video_reader.py`, `window_utils.py` | Membre 1 |
| **Détection YOLO** | `detector.py` | Membre 1 ou 2 |
| **Suivi des objets** | `tracker.py` | Membre 2 |
| **Évaluation du risque et HUD** | `risk_evaluator.py`, `hud.py` | Membre 2 ou 3 |
| **Intégration + démonstrations** | `demo_detection.py`, `demo_video.py`, `tests/` | Tout le monde |

> Chaque membre peut travailler sur une branche Git séparée (`feature/detection`, `feature/alerts`, etc.).

---

## 8. Données de test recommandées

- Vidéos de conduite à Lomé / Togo (YouTube ou filmées par le groupe)
- Vidéos contenant des nids-de-poule
- Scènes avec beaucoup de zemidjans et de piétons

Placez vos vidéos dans le dossier `data/videos/`.

---

## 9. Tests

Depuis la racine du projet, exécutez les tests avec :

```bash
python -m unittest discover -s tests -v
```

---

## 10. Auteurs

Projet réalisé dans le cadre du cours de **Big Data – IAI Togo**.  
*Année académique 2025-2026*

---

## 11. Licence

Ce projet est réalisé à des fins pédagogiques.