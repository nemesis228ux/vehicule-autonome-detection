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
│   └── sounds/            # Klaxons MP3 prioritaires + WAV de secours
│
├── data/
│   ├── videos/            # Vidéos de test (à fournir)
│   └── images/            # Images de test (à fournir)
│
├── models/                # Modèles YOLO locaux (non versionnés)
│
├── scripts/
│   └── generate_horn.py   # Générateur des klaxons synthétiques WAV
│
├── src/
│   ├── main.py            # Point d'entrée principal
│   ├── demo_detection.py  # Lanceur de compatibilité
│   ├── demo_video.py      # Démonstration du lecteur vidéo seul
│   ├── video_reader.py    # Lecture vidéo, caméra et flux
│   ├── detector.py        # Compatibilité avec les anciens imports
│   ├── tracker.py         # Suivi ByteTrack, vitesse et trajectoire
│   ├── risk_evaluator.py  # Compatibilité avec les anciens imports
│   ├── hud.py             # Compatibilité avec les anciens imports
│   ├── alerts.py          # Compatibilité avec les anciens imports
│   ├── window_utils.py    # Adaptation de la fenêtre à l'écran
│   ├── detection_app/     # CLI, orchestration et rendu de la démo
│   ├── detection/         # Détection YOLOv8, modèles et rendu
│   ├── alerts/            # Configuration, gestion, sons et overlay d'alerte
│   ├── risk/              # Modèles et calcul du niveau de risque
│   └── presentation/      # HUD et composants d'affichage
│
├── tests/                 # Tests unitaires
└── outputs/               # Vidéos annotées générées avec --save
```

Les modules historiques (`detector.py`, `risk_evaluator.py`, `hud.py` et
`alerts.py`) conservent leurs imports publics et redirigent vers les packages
organisés par fonctionnalité.

---

## 3. Outils et technologies

| Outil / Bibliothèque | Rôle |
| :--- | :--- |
| **Python 3.10+** | Langage principal |
| **Ultralytics YOLOv8** | Détection d'objets |
| **ByteTrack** | Suivi des objets entre les images |
| **OpenCV** | Lecture vidéo, affichage, dessin |
| **NumPy** | Calculs numériques et génération du klaxon |
| **lap** | Association utilisée par le suivi |
| **winsound / MCI / wave** | Lecture audio sous Windows et génération des WAV de secours |

---

## 4. Installation

### Prérequis
- Python 3.10 ou supérieur
- Git
- *(Recommandé)* Un environnement virtuel

### Étapes

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/nemesis228ux/vehicule-autonome-detection.git
   cd vehicule-autonome-detection
   ```

2. **Créer un environnement virtuel :**
   ```bash
   python -m venv .venv
   ```

3. **Activer l'environnement virtuel :**
   - **Sur Windows :**
     ```bash
     .venv\Scripts\Activate.ps1
     ```
   - **Sur Linux / macOS :**
     ```bash
     source .venv/bin/activate
     ```

4. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Lancer la démonstration réelle

Le point d'entrée recommandé est `src/main.py`. Le module
`src/demo_detection.py` reste disponible comme ancien lanceur compatible.
Depuis la racine du dépôt, lancez l'application ainsi :

```powershell
python -m src.main --help
```

Le pipeline enchaîne :

```text
VideoReader -> Detector YOLOv8 -> Tracker ByteTrack
            -> RiskEvaluator -> AlertManager
            -> HUD/overlay -> fenêtre OpenCV
```

La fenêtre affiche la vidéo annotée : cadres, IDs et trajectoires lorsque le
suivi est activé, HUD et alertes selon les options, et FPS avec `--show-fps`.

### 5.1 Vérifier l'installation

Activez l'environnement virtuel créé à l'étape d'installation, puis vérifiez
les dépendances et les tests :

```powershell
.venv\Scripts\Activate.ps1
python -c "import cv2, ultralytics, numpy; print('OK')"
python -m unittest discover -s tests -v
```

### 5.2 Vérifier le modèle YOLO

Le fichier `.pt` contient les poids appris de YOLO, utilisés pour reconnaître
les objets dans les images. `yolov8n.pt` est le modèle nano pré-entraîné et
léger utilisé par défaut. Placez-le dans `models\` :

```text
models\yolov8n.pt
```

Vous n'avez pas besoin de passer `--model` pour le modèle par défaut. Pour
utiliser un modèle différent (par exemple vos poids entraînés `best.pt`),
placez-le dans `models\` et indiquez son chemin :

```powershell
python -m src.main --source data/videos/route.mp4 --model models\best.pt
```

Les poids ne sont pas versionnés par Git (`.gitignore`). Si le fichier local
est absent, Ultralytics peut tenter de télécharger `yolov8n.pt` au premier
lancement. Le modèle général pré-entraîné ne reconnaît pas nécessairement les
nids-de-poule : cette classe demande un modèle entraîné pour cela.

### 5.3 Fournir la vidéo

Placez votre propre vidéo dans `data\videos\`. Les médias de test ne sont pas
versionnés par Git ; le dépôt ne fournit pas de vidéo d'exemple.

```text
data\videos\route.mp4
```

Pour évaluer le suivi et le risque, privilégiez une vidéo dashcam normale.

### 5.4 Commande minimale

```powershell
python -m src.main --source data/videos/route.mp4 --imgsz 320
```

Vous devez voir :

- une fenêtre OpenCV avec la vidéo ;
- des cadres colorés autour des objets détectés ;
- les catégories et scores de confiance ;
- la console avec `Source`, `Modèle`, `Résolution` et le nombre de frames traitées.

Sans `--source`, l'application essaie d'ouvrir la caméra locale d'index `0`.

### 5.5 Démonstration complète

```powershell
python -m src.main --source data/videos/route.mp4 --alerts --show-fps --imgsz 320
```

Vous devez voir en plus :

- les IDs ByteTrack et les trajectoires ;
- le HUD `INFORMATION / ATTENTION / CRITIQUE` ;
- le panneau des niveaux et la légende ;
- la bordure orange ou rouge selon le danger ;
- le klaxon d'attention ou le triple klaxon critique ;
- les FPS en haut de la fenêtre.

Commandes pendant l'exécution :

- `q` ou Échap : quitter ;
- `m` : couper/réactiver le son.

### 5.6 Options de `main`

| Option | Effet |
|---|---|
| `--source` | Fichier vidéo, caméra (`0`, `1`, ...) ou URL de flux |
| `--model` | Modèle Ultralytics ou chemin `.pt` (défaut : `yolov8n.pt`) |
| `--conf` | Seuil de confiance (défaut : `0.35`) |
| `--iou` | Seuil IoU du NMS (défaut : `0.45`) |
| `--imgsz` | Taille d'inférence (défaut : `640`, `320` peut être plus rapide sur CPU) |
| `--all-classes` | Conserve aussi les classes hors catégories routières |
| `--track` | Active ByteTrack et les IDs |
| `--risk` | Active le risque et le HUD (implique `--track`) |
| `--alerts` | Active les alertes visuelles et sonores (implique `--risk`) |
| `--no-sound` | Garde les alertes visuelles sans klaxon |
| `--save` | Enregistre la vidéo annotée dans `outputs/` |
| `--show-fps` | Affiche les FPS |
| `--tracker` | Tracker Ultralytics (défaut : `bytetrack.yaml`, ex. `botsort.yaml`) |
| `--labels` | `full`, `id` ou `none` |
| `--trail-length` | Nombre de points de trajectoire |
| `--max-age` | Frames avant purge d'un ID perdu |
| `--window` | Nom de la fenêtre OpenCV |

Exemples :

```powershell
# Suivi seul
python -m src.main --source data/videos/route.mp4 --track --imgsz 320

# Risque seul
python -m src.main --source data/videos/route.mp4 --risk --imgsz 320

# Alertes visuelles sans son
python -m src.main --source data/videos/route.mp4 --alerts --no-sound --imgsz 320

# Sauvegarde de la vidéo annotée
python -m src.main --source data/videos/route.mp4 --alerts --save --show-fps --imgsz 320
```

La sauvegarde crée un fichier dans `outputs\`, nommé d'après la source et les
options (par exemple `route_alertes.mp4`, `route_risque.mp4` ou
`route_tracke.mp4`). Pour un flux direct ou une caméra, le nom commence par
`direct`.

### 5.7 Webcam

```powershell
python -m src.main --source 0 --alerts --imgsz 320
```

`0` correspond généralement à la webcam par défaut. Avec DroidCam/Iriun, essayer `1`, `2`, etc.

### 5.8 Caméra d'un téléphone

Une application de caméra réseau doit être installée sur le téléphone. Elle
fournit soit une caméra virtuelle (sélectionnée comme un index `0`, `1`, ...),
soit une URL vidéo compatible avec OpenCV. Pour une URL, le téléphone et
l'ordinateur doivent généralement être sur le même réseau Wi-Fi ; démarrez le
flux dans l'application et utilisez l'adresse exacte qu'elle affiche. Le chemin
(`/video`, `/mjpeg`, etc.) et le protocole dépendent de l'application.

Le lecteur vidéo accepte déjà les URL via OpenCV : aucun module Python
supplémentaire ni code spécifique à l'application n'est requis. Depuis la racine
du projet, lancez par exemple :

```powershell
python -m src.main --source "http://ADRESSE_DU_TELEPHONE:PORT/video" --alerts --imgsz 320
```

Remplacez l'adresse et le chemin par ceux fournis par l'application du
téléphone. Pour une caméra virtuelle, utilisez plutôt son index :

```powershell
python -m src.main --source 1 --alerts --imgsz 320
```

Autorisez l'accès au réseau local dans le pare-feu si nécessaire. Pour valider
le flux avant la détection, vous pouvez également lancer
`python -m src.demo_video --source "URL_DU_FLUX"`.

### 5.9 Dépannage

- **Aucune fenêtre** : lancer la commande dans une session Windows avec affichage, pas dans un terminal sans GUI.
- **Aucun cadre** : baisser `--conf 0.10`, essayer `--all-classes`, vérifier `--model` et `--source`.
- **Trop lent** : garder `yolov8n.pt` et `--imgsz 320`.
- **Pas de son** : vérifier la présence des MP3 dans `assets\sounds\`, le volume Windows, et que `--no-sound` n'est pas actif.
- **Trop d'alertes** : le risque est une heuristique monoculaire ; vérifiez avec
  une vidéo représentative et ajustez les paramètres de l'évaluateur si besoin.

### 5.10 Fichiers audio

Placez les klaxons dans `assets/sounds/` :

- `klaxon_attention.mp3` : klaxon d'attention (prioritaire).
- `klaxon_critique.mp3` : klaxon critique (prioritaire).

Les MP3 fournis sont lus via **MCI Windows**, de façon asynchrone. Si un MP3
n'est pas disponible, le WAV correspondant sert de secours. Les fichiers WAV
peuvent être régénérés (et remplacés) avec :

```powershell
python scripts/generate_horn.py
```

### 5.11 Comportement des alertes

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

Le package `src/alerts/` transforme ces niveaux en alertes stables : debounce
pour l'Attention, immédiateté pour le Critique, cooldowns pour éviter le spam,
mute utilisateur et overlay visuel. Le son utilise les MP3 fournis dans
`assets/sounds/`, avec les WAV de secours générés localement.

---

## 7. Organisation du travail en groupe

| Partie | Fichier(s) concerné(s) | Responsable possible |
| :--- | :--- | :--- |
| **Lecture vidéo et fenêtre** | `src/video_reader.py`, `src/window_utils.py` | Membre 1 |
| **Détection YOLO** | `src/detection/`, `src/detector.py` | Membre 1 ou 2 |
| **Suivi des objets** | `src/tracker.py` | Membre 2 |
| **Évaluation du risque et HUD** | `src/risk/`, `src/presentation/` | Membre 2 ou 3 |
| **Alertes visuelles et sonores** | `src/alerts/`, `assets/sounds/` | Membre 3 |
| **Intégration + démonstrations** | `src/main.py`, `src/detection_app/`, `src/demo_video.py`, `tests/` | Tout le monde |

---

## 8. Données de test recommandées

- Vidéos de conduite à Lomé / Togo (YouTube ou filmées par le groupe)
- Vidéos contenant des nids-de-poule
- Scènes avec beaucoup de zemidjans et de piétons

Les vidéos de test ne sont pas incluses dans le dépôt. Une vidéo dashcam réelle
est préférable pour évaluer le suivi et le risque ; ces résultats dépendent
fortement de la scène et de la caméra.

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
Ran 80 tests in ...
OK
```

La suite couvre :

- `test_video_reader.py` : lecture vidéo, caméra, erreurs et libération des ressources ;
- `test_detector.py` : géométrie, filtrage des classes, seuils et appels Ultralytics ;
- `test_tracker.py` : IDs, vitesse, trajectoire, expiration et rendu ;
- `test_risk_evaluator.py` : TTC, corridor, VRU, hystérésis, purge et HUD ;
- `test_alerts.py` : debounce, cooldowns, escalade, mute, backend audio et overlay ;
- `test_window_utils.py` : redimensionnement et conservation du ratio ;
- `test_main.py` : transmission des arguments au point d'entrée principal.

Les tests simulent les sources vidéo et les modèles ; ils ne valident pas
l'ouverture réelle d'une caméra ni la qualité du modèle YOLO.

---

## 10. Commandes de vérification rapide

Depuis la racine du dépôt, avec l'environnement virtuel activé :

```powershell
# Tests unitaires
python -m unittest discover -s tests -v

# Détection en direct depuis la caméra par défaut (q ou Échap pour quitter)
python -m src.main --source 0 --alerts --imgsz 320

# Générer à nouveau les WAV de secours si nécessaire
python scripts/generate_horn.py
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