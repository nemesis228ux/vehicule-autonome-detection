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
│   ├── detector.py       # Détection avec YOLOv8
│   ├── tracker.py        # Suivi ByteTrack, vitesse et trajectoire
│   ├── risk_evaluator.py # Évaluation du danger
│   ├── hud.py            # Affichage du niveau de risque
│   ├── alerts.py         # Alertes visuelles et klaxon synthétique
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

### Suivi, risque et alertes
```bash
# Activer le suivi ByteTrack et afficher les identifiants
python -m src.demo_detection --source data/videos/votre_video.mp4 --track

# Évaluer le risque et afficher le HUD (active aussi le suivi)
python -m src.demo_detection --source data/videos/votre_video.mp4 --risk

# Activer les alertes visuelles et le klaxon (active le suivi et le risque)
python -m src.demo_detection --source data/videos/votre_video.mp4 --alerts

# Alertes visuelles seules, sans klaxon
python -m src.demo_detection --source data/videos/votre_video.mp4 --alerts --no-sound

# Sauvegarder la vidéo annotée et afficher les FPS
python -m src.demo_detection --source data/videos/votre_video.mp4 --alerts --save --show-fps

# Utiliser un autre modèle ou accélérer l'inférence sur CPU
python -m src.demo_detection --source 0 --model yolov8n.pt --imgsz 320
```

Pendant l'exécution :
- `q` ou Échap : quitter ;
- `m` : couper/réactiver le son.

Les vidéos créées avec `--save` sont enregistrées dans `outputs/` (`*_alertes.mp4` avec `--alerts`).
Pour tester uniquement la lecture d'une source vidéo, utilisez `python -m src.demo_video --source 0`.

### Fichiers audio
Placez les klaxons dans `assets/sounds/` :

- `klaxon_attention.mp3` : klaxon d'attention (prioritaire).
- `klaxon_critique.mp3` : klaxon critique (prioritaire).

Les MP3 sont lus via **MCI Windows**, de façon asynchrone. Si un MP3 est absent, le WAV correspondant (`klaxon_attention.wav`, `klaxon_critique.wav`) sert de secours. Ces WAV peuvent être régénérés avec :

```bash
python scripts/generate_horn.py
```

### Comportement des alertes
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

Depuis la racine du projet, exécutez les tests avec :

```bash
python -m unittest discover -s tests -v
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

## 10. Branches développées et contexte de reprise

### 10.1 Branches et spécificités

| Branche | Commit | Spécificité principale | État |
|---|---|---|---|
| `main` | `2b616c5` | Squelette initial, README de référence | Poussée |
| `feature/video-reader` | `3781fbc` → `4cc8b3d` → `b02d505` | `VideoReader`, `window_utils`, détection YOLOv8, filtrage des classes routières et première démo | Locale ; `origin` est à `4cc8b3d`, `b02d505` non poussé |
| `feature/tracking` | `4f7e676` | ByteTrack, dépendance `lap`, `track_id`, vitesse, trajectoire, `draw_tracks`, option `--track` | Poussée |
| `feature/risk-evaluator` | `a3fc8c5` | Risque monoculaire, HUD ADAS, option `--risk`, tests et README | Poussée |
| `feature/alerts` | en cours | `alerts.py`, klaxon synthétique, cooldowns, mute, overlay et tests | Locale, non commitée |
| `feature/detection` | — | **N'existe pas** ; la détection a été développée dans `feature/video-reader` | À ne pas recréer |

### 10.2 Ce que contient réellement chaque branche

- `feature/video-reader` ne contient pas seulement la lecture vidéo : elle porte aussi la détection YOLOv8, `window_utils` et l'intégration initiale dans `demo_detection.py`.
- `feature/tracking` dépend de `feature/video-reader` ; elle ajoute les IDs stables, la vitesse, les trajectoires et les modes d'étiquettes.
- `feature/risk-evaluator` dépend de `feature/tracking` ; elle ajoute `RiskEvaluator`, le HUD, l'option `--risk` et les tests de risque.
- `feature/alerts` dépend de `feature/risk-evaluator` ; elle ajoute `AlertManager`, le klaxon synthétique, l'anti-spam, le mute et l'overlay.
- Ordre logique : `main` → `feature/video-reader` → `feature/tracking` → `feature/risk-evaluator` → `feature/alerts`.

### 10.3 État d'avancement

- Terminés : lecture vidéo, détection YOLOv8, suivi ByteTrack, risque monoculaire, HUD et alertes.
- `src/main.py` est encore vide : le point d'entrée actuel est `src/demo_detection.py`.
- Les MP3 `klaxon_attention.mp3` et `klaxon_critique.mp3` sont prioritaires dans `assets/sounds/`.
- Les WAV de secours sont générés par `scripts/generate_horn.py`.
- Le son est asynchrone sous Windows : MCI pour les MP3, `winsound` pour les WAV. Hors Windows, les alertes restent visuelles.
- La prochaine étape logique est l'intégration dans `main.py`, puis la calibration sur une vraie vidéo dashcam.

### 10.4 Interfaces clés pour reprendre

- `RiskEvaluator.evaluate(objects, frame_shape, fps) -> RiskSummary`
- `draw_risk_hud(frame, objects, summary, fps=None, frame_index=0)`
- `AlertManager.update(summary, now_s=None, frame_index=None) -> AlertEvent`
- `draw_alert_overlay(frame, event, thickness=None)`
- `WinsoundBackend` / `NullSoundBackend` : lecture audio injectable, utile pour les tests.

### 10.5 Contraintes et limites connues

- Windows : Python 3.13, environnement `.venv`, OpenCV 5, Ultralytics 8.4.207, `lap`.
- Pas de `pygame` ni de `playsound` ; MCI Windows lit les MP3 et `winsound` lit les WAV.
- Les textes HUD et les noms de fenêtres sont en **ASCII** pour éviter les problèmes de rendu OpenCV.
- `models/yolov8n.pt` est gitignoré et doit être présent localement.
- `data/videos/s.mp4` est un triptyque : IDs triplés et risque saturé, à ne pas utiliser pour la calibration.
- Les nids-de-poule nécessitent un modèle entraîné spécifiquement ; YOLOv8n ne les détecte pas de façon fiable.
- L'estimation de distance/TTC est monoculaire et non calibrée physiquement.
- Aucune vraie vidéo dashcam n'est encore intégrée au dépôt.

### 10.6 Consignes pour une autre IA

1. Lire cette section avant toute modification.
2. Partir de `feature/alerts` (ou de `feature/risk-evaluator` si les alertes ne sont pas encore fusionnées).
3. Ne pas recréer `feature/detection` : elle n'existe pas.
4. Lancer `python -m unittest discover -s tests -v` avant et après chaque changement.
5. Conserver les textes HUD en ASCII et ne pas ajouter `pygame`/`playsound` sans décision explicite.
6. Ne jamais committer de clé API, de secret ou de fichier `.env`.
7. Vérifier les chemins Windows avec l'environnement `.venv` avant de conclure qu'un module est cassé.

### 10.7 Commandes de vérification rapide

```bash
python -m unittest discover -s tests -v
python scripts/generate_horn.py
python -m src.demo_detection --source data/videos/s.mp4 --alerts --no-sound --imgsz 320
```

---

## 11. Auteurs

Projet réalisé dans le cadre du cours de **Big Data – IAI Togo**.  
*Année académique 2025-2026*

---

## 12. Licence

Ce projet est réalisé à des fins pédagogiques.