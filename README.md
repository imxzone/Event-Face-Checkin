Real-Time Face Recognition for Event Check-in
Introduction

This project presents a real-time face recognition–based event check-in system designed for student clubs and small to medium-sized events.
By leveraging computer vision and a pretrained face recognition model, the system automatically identifies participants via webcam and records their attendance efficiently.

Motivation

Conventional check-in methods such as manual lists or QR codes are often time-consuming, inconvenient, and prone to misuse.
The goal of this project is to develop a fast, contactless, and easy-to-deploy check-in solution that focuses on practical implementation rather than complex model training.

Method Overview

The system follows a straightforward pipeline:

Capture face frames from a webcam in real time

Extract face embeddings using a pretrained face recognition model

Match embeddings against registered participants

Log successful check-ins and prevent duplicate entries

This project emphasizes real-time inference and system integration, not training models from scratch.

How to run
python preprocess.py
python build_embeddings.py
python main.py


Check-in records are stored in checkins.csv.

Requirements

Python 3.9+

OpenCV

InsightFace

NumPy

Pandas
