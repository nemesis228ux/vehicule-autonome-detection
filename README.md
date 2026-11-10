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
6. Alertes visuelles et sonores avec klaxon synthétique, anti-spam et mute

Le système illustre la première brique d'un véhicule autonome (**la Perception**) et met en avant les défis spécifiques aux routes africaines (nids-de-poule, forte présence de deux-roues, signalisation limitée, etc.).

> L'évaluation du risque est une estimation visuelle à partir des images, et non une mesure réelle de distance ou une garantie de sécurité. Les alertes sonores utilisent un klaxon synthétique et ne remplacent pas un dispositif de sécurité certifié.

---

## 2. Architecture du projet

```text
vehicule-autonome-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/
│   └── sounds/           # Klaxons MP3 prioritaires + WAV de secours
│
├── data/
│   ├── videos/          # Vidéos de test
│   └── images/          # Images de test
│
├── models/              # Modèles YOLO (.pt), dont yolov8n.pt
│
├── scripts/
│   └── generate_horn.py # Générateur des klaxons synthétiques
│
├── src/
│   ├── demo_detection.py # Point d'entrée : détection, suivi, risque et alertes
│   ├── demo_video.py     # Démonstration du lecteur vidéo seul
│   ├── video_reader.py   # Lecture vidéo, caméra et flux
│   ├── detector.py       # Compatibilité avec les anciens imports
│   ├── tracker.py        # Suivi ByteTrack, vitesse et trajectoire
│   ├── risk_evaluator.py # Compatibilité avec les anciens imports
│   ├── hud.py            # Compatibilité avec les anciens imports
│   ├── alerts.py         # Compatibilité avec les anciens imports
│   └── window_utils.py   # Adaptation de la fenêtre à l'écran
│   ├── detection/        # Détection YOLOv8, modèles et rendu
│   ├── alerts/           # Configuration, gestion, sons et overlay d'alerte
│   ├── risk/             # Modèles et calcul du niveau de risque
│   └── presentation/     # HUD et composants d'affichage
│
├── tests/                # Tests unitaires
└── outputs/              # Vidéos annotées générées avec --save
```

Les modules historiques (`detector.py`, `risk_evaluator.py`, `hud.py` et
`alerts.py`) conservent leurs imports publics et redirigent vers les packages
organisés par fonctionnalité.

---

## 3. Outils et technologies

| Outil / Bibliothèque | Rôle |
| :--- | :--- |
| **Python 3.9+** | Langage principal |
| **Ultralytics YOLOv8** | Détection d'objets |
| **ByteTrack** | Suivi des objets entre les images |
| **OpenCV** | Lecture vidéo, affichage, dessin |
| **NumPy** | Calculs numériques et génération du klaxon |
| **lap** | Association utilisée par le suivi |
| **winsound / MCI / wave** | Lecture asynchrone des sons et génération des WAV de secours |

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

## 5. Lancer la démonstration réelle

Le point d'entrée est `src/demo_detection.py`. Il enchaîne :

```text
VideoReader -> Detector YOLOv8 -> Tracker ByteTrack
            -> RiskEvaluator -> AlertManager
            -> HUD/overlay -> fenêtre OpenCV
```

La fenêtre affiche la vidéo annotée : cadres, IDs, trajectoires, HUD de risque, bordures d'alerte et FPS.

### 5.1 Activer l'environnement

Depuis la racine du projet, sous Windows PowerShell :

```powershell
cd "D:\mak\IAI 3 EME ANNEE\COURS\BIG DATA\DEV BIG DATA\vehicule-autonome-detection"
.venv\Scripts\Activate.ps1
```

Vérification rapide des dépendances :

```powershell
python -c "import cv2, ultralytics, numpy; print('OK')"
python -m unittest discover -s tests -v
```

### 5.2 Vérifier le modèle YOLO

Le modèle local attendu est :

```text
models\yolov8n.pt
```

Les poids ne sont pas versionnés par Git (`.gitignore`). S'ils sont absents, Ultralytics peut télécharger `yolov8n.pt` au premier lancement. Pour imposer un autre modèle :

```powershell
python -m src.demo_detection --source data/videos/s.mp4 --model models\best.pt
```

### 5.3 Fournir la vidéo

Place ta vidéo dans `data\videos\`. Ce dossier est ignoré par Git : chaque machine doit fournir sa propre vidéo.

```text
data\videos\s.mp4
```

> `s.mp4` est un triptyque de trois scènes empilées : la démo fonctionne, mais les objets reçoivent trois IDs et le risque sature souvent en Critique. Pour un test réaliste, utilise une vidéo dashcam normale.

### 5.4 Commande minimale

```powershell
python -m src.demo_detection --source data/videos/s.mp4 --imgsz 320
```

Tu dois voir :

- une fenêtre OpenCV avec la vidéo ;
- des cadres colorés autour des objets détectés ;
- les catégories et scores de confiance ;
- la console avec `Source`, `Modèle`, `Résolution` et le nombre de frames traitées.

### 5.5 Démonstration complète

```powershell
python -m src.demo_detection --source data/videos/s.mp4 --alerts --show-fps --imgsz 320
```

Tu dois voir en plus :

- les IDs ByteTrack et les trajectoires ;
- le HUD `INFORMATION / ATTENTION / CRITIQUE` ;
- le panneau des niveaux et la légende ;
- la bordure orange ou rouge selon le danger ;
- le klaxon d'attention ou le triple klaxon critique ;
- les FPS en haut de la fenêtre.

Commandes pendant l'exécution :

- `q` ou Échap : quitter ;
- `m` : couper/réactiver le son.

### 5.6 Options utiles de `demo_detection`

| Option | Effet |
|---|---|
| `--source` | Fichier vidéo, caméra (`0`, `1`, ...) ou URL de flux |
| `--model` | Modèle Ultralytics ou chemin `.pt` |
| `--conf` | Seuil de confiance (baisser à `0.10` si peu de détections) |
| `--imgsz` | Taille d'inférence (`320` plus rapide sur CPU) |
| `--track` | Active ByteTrack et les IDs |
| `--risk` | Active le risque et le HUD (implique `--track`) |
| `--alerts` | Active les alertes visuelles et sonores (implique `--risk`) |
| `--no-sound` | Garde les alertes visuelles sans klaxon |
| `--save` | Enregistre la vidéo annotée dans `outputs/` |
| `--show-fps` | Affiche les FPS |
| `--labels` | `full`, `id` ou `none` |
| `--trail-length` | Nombre de points de trajectoire |
| `--max-age` | Frames avant purge d'un ID perdu |

Exemples :

```powershell
# Suivi seul
python -m src.demo_detection --source data/videos/s.mp4 --track --imgsz 320

# Risque seul
python -m src.demo_detection --source data/videos/s.mp4 --risk --imgsz 320

# Alertes visuelles sans son
python -m src.demo_detection --source data/videos/s.mp4 --alerts --no-sound --imgsz 320

# Sauvegarde de la vidéo annotée
python -m src.demo_detection --source data/videos/s.mp4 --alerts --save --show-fps --imgsz 320
```

La sauvegarde produit `outputs\s_alertes.mp4` (ou `s_risque.mp4`, `s_tracke.mp4` selon les options).

### 5.7 Webcam

```powershell
python -m src.demo_detection --source 0 --alerts --imgsz 320
```

`0` correspond généralement à la webcam par défaut. Avec DroidCam/Iriun, essayer `1`, `2`, etc.

### 5.8 Dépannage

- **Aucune fenêtre** : lancer la commande dans une session Windows avec affichage, pas dans un terminal sans GUI.
- **Aucun cadre** : baisser `--conf 0.10`, essayer `--all-classes`, vérifier `--model` et `--source`.
- **Trop lent** : garder `yolov8n.pt` et `--imgsz 320`.
- **Pas de son** : vérifier la présence des MP3 dans `assets\sounds\`, le volume Windows, et que `--no-sound` n'est pas actif.
- **Trop d'alertes** : normal sur `s.mp4` (triptyque) ; tester sur une vraie vidéo dashcam.

### 5.9 Fichiers audio

Placez les klaxons dans `assets/sounds/` :

- `klaxon_attention.mp3` : klaxon d'attention (prioritaire).
- `klaxon_critique.mp3` : klaxon critique (prioritaire).

Les MP3 sont lus via **MCI Windows**, de façon asynchrone. Si un MP3 est absent, le WAV correspondant (`klaxon_attention.wav`, `klaxon_critique.wav`) sert de secours. Ces WAV peuvent être régénérés avec :

```powershell
python scripts/generate_horn.py
```

### 5.10 Comportement des alertes

- **Information** : aucun son, aucune bordure d'alerte.
- **Attention** : bordure orange stable + un coup de klaxon, avec rappel espacé.
- **Critique** : bordure rouge clignotante + triple klaxon, avec rappel plus fréquent.
- Une escalade Attention → Critique déclenche immédiatement le klaxon critique.
- Le son est asynchrone et ne bloque pas la boucle vidéo ; MCI est utilisé pour les MP3, `winsound` pour les WAV.
- Hors Windows, ou si aucun fichier audio n'est disponible, les alertes restent visuelles.

---

## 6. Niveaux de risque

| Niveau | Couleur | Signification | Exemple |
| :--- | :---: | :--- | :--- |
| **Critique** | 🔴 Rouge | Risque élevé dans la trajectoire | Objet proche ou temps avant collision estimé court |
| **Attention** | 🟠 Orange | Situation à surveiller | Objet qui se rapproche ou usager vulnérable en traversée |
| **Information** | 🟢 Vert | Objet détecté sans risque immédiat estimé | Objet éloigné ou hors trajectoire |

Le calcul s'appuie sur la position et la taille des objets dans l'image, leur suivi entre les frames, une estimation de proximité et le temps avant collision (TTC). Il s'agit d'heuristiques monoculaires dépendantes de la caméra et de la scène, pas d'une distance physique calibrée. La classe « nid-de-poule » nécessite un modèle qui a été entraîné à la détecter ; le modèle YOLO général fourni peut ne pas la reconnaître.

Le module `alerts.py` transforme ces niveaux en alertes stables : debounce pour l'Attention, immédiateté pour le Critique, cooldowns pour éviter le spam, mute utilisateur et overlay visuel. Le son utilise les MP3 fournis dans `assets/sounds/`, avec un klaxon synthétique WAV de secours généré localement.

---

## 7. Organisation du travail en groupe

| Partie | Fichier(s) concerné(s) | Responsable possible |
| :--- | :--- | :--- |
| **Lecture vidéo et fenêtre** | `video_reader.py`, `window_utils.py` | Membre 1 |
| **Détection YOLO** | `detector.py` | Membre 1 ou 2 |
| **Suivi des objets** | `tracker.py` | Membre 2 |
| **Évaluation du risque et HUD** | `risk_evaluator.py`, `hud.py` | Membre 2 ou 3 |
| **Alertes visuelles et sonores** | `alerts.py`, `assets/sounds/` | Membre 3 |
| **Intégration + démonstrations** | `demo_detection.py`, `demo_video.py`, `tests/` | Tout le monde |

> Chaque membre peut travailler sur une branche Git séparée. Les branches réellement utilisées sont détaillées à la section 10.

---

## 8. Données de test recommandées

- Vidéos de conduite à Lomé / Togo (YouTube ou filmées par le groupe)
- Vidéos contenant des nids-de-poule
- Scènes avec beaucoup de zemidjans et de piétons

Placez vos vidéos dans le dossier `data/videos/`.

> `data/videos/s.mp4` est un **triptyque de trois scènes empilées** : chaque objet peut recevoir trois IDs et le risque sature souvent en Critique. Il sert uniquement aux tests de plomberie, pas à la calibration. Une vidéo dashcam réelle est nécessaire pour régler les seuils.

---

## 9. Tests

Les tests sont unitaires et ne nécessitent ni vidéo ni modèle : ils utilisent des objets simulés et un faux backend audio. Exécutez-les depuis la racine du projet, avec l'environnement virtuel activé.

### Sous Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

### Sous Linux / macOS

```bash
cd vehicule-autonome-detection
source .venv/bin/activate
python -m unittest discover -s tests -v
```

### Lancer un fichier ou un test précis

```powershell
python -m unittest tests.test_alerts -v
python -m unittest tests.test_alerts.AlertManagerTests.test_critical_is_immediate -v
```

### Résultat attendu

```text
Ran 79 tests in ...
OK
```

La suite couvre :

- `test_video_reader.py` : lecture vidéo, caméra, erreurs et libération des ressources ;
- `test_detector.py` : géométrie, filtrage des classes, seuils et appels Ultralytics ;
- `test_tracker.py` : IDs, vitesse, trajectoire, expiration et rendu ;
- `test_risk_evaluator.py` : TTC, corridor, VRU, hystérésis, purge et HUD ;
- `test_alerts.py` : debounce, cooldowns, escalade, mute, backend audio et overlay ;
- `test_window_utils.py` : redimensionnement et conservation du ratio.

Résultat actuel : **79 tests OK**.

---

### 10.7 Commandes de vérification rapide

```bash
python -m unittest discover -s tests -v
python scripts/generate_horn.py
python -m src.demo_detection --source data/videos/s.mp4 --alerts --no-sound --imgsz 320
```

---

## 11. Auteurs

Projet réalisé dans le cadre du cours de **Big Data – IAI Togo**.  
Par:
   **- BARRY ALI**
   **- ABOTCHI-SODJEHOUN Clara**
   **- AGBEGNIGAN Helmut**
   **- KPOGO Yayra**
*Année académique 2025-2026*

---

## 12. Licence

Ce projet est réalisé à des fins pédagogiques.