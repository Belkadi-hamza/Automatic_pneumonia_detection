# 🫁 PneumoScan – Détection de pneumonie par intelligence artificielle

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**PneumoScan** est une application web de diagnostic assisté par IA qui analyse des radiographies pulmonaires pour détecter automatiquement les signes de pneumonie.  
Le système utilise un modèle **EfficientNetB3** fine‑tuné sur le jeu de données *Chest X‑Ray Pneumonia*, et propose une visualisation explicative par **heatmap d’occlusion** (Grad‑CAM alternatif) pour mettre en évidence les zones de l’image qui ont influencé la prédiction.

---

## ✨ Fonctionnalités

- 🔍 **Prédiction binaire** : `PNEUMONIE` / `NORMAL` avec score de confiance
- 🔥 **Heatmap d’occlusion** – visualisation des régions critiques pour le diagnostic
- 📁 **Traitement par lot** – analyse d’une ou plusieurs images simultanément
- 🎚️ **Ajustement dynamique** – seuil de décision et sensibilité de la heatmap
- 💾 **Export des résultats** – téléchargement individuel ou archivage ZIP
- 🌐 **Interface web moderne** – compatible mobile, drag & drop, retour visuel temps réel

---

## 🧠 Architecture du modèle

| Composant                | Détail                                                                 |
|--------------------------|------------------------------------------------------------------------|
| **Modèle de base**       | EfficientNetB3 pré‑entraîné sur ImageNet                               |
| **Fine‑tuning**          | Dégel des dernières couches – apprentissage sur radiographies          |
| **Couches supplémentaires** | GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.3) → Dense(1, sigmoid) |
| **Seuil optimal**        | 0.9220 (maximisation du F1‑score sur la validation)                    |
| **Métriques**            | Accuracy ~92% ; AUC ~0.97 ; Sensibilité ~94% ; Spécificité ~90%        |

> Le modèle est sauvegardé au format Keras (`.keras`) et chargé au démarrage de l’API.

---

## 📦 Structure du projet
