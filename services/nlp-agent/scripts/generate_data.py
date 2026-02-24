import pandas as pd
import numpy as np
import os

data = [
    # Dolor
    ("Me duele mucho la espalda hoy", 0.9, 0.1, 0.2, 0.4, 0.2),
    ("Tengo un dolor punzante en la rodilla", 0.85, 0.1, 0.2, 0.3, 0.2),
    ("No siento ningún dolor", 0.0, 0.0, 0.1, 0.8, 0.1),
    
    # Fatiga
    ("Estoy extremadamente cansado, no puedo ni levantarme", 0.2, 0.95, 0.4, 0.2, 0.1),
    ("Siento mucha fatiga hoy", 0.1, 0.8, 0.3, 0.4, 0.2),
    ("Tengo mucha energía y vitalidad", 0.0, 0.0, 0.1, 0.9, 0.1),
    
    # Ansiedad
    ("Me siento muy ansioso y con palpitaciones", 0.3, 0.4, 0.9, 0.3, 0.3),
    ("Estoy muy preocupado por mi salud", 0.2, 0.3, 0.8, 0.4, 0.3),
    ("Estoy tranquilo y relajado", 0.0, 0.1, 0.05, 0.9, 0.1),
    
    # Sueño
    ("No he podido dormir en toda la noche", 0.3, 0.7, 0.5, 0.3, 0.9),
    ("Me desperté varias veces durante la noche", 0.2, 0.5, 0.4, 0.4, 0.8),
    ("He dormido como un tronco, muy reparador", 0.0, 0.0, 0.1, 0.9, 0.0),
    
    # Ánimo (Mood)
    ("Me siento muy triste y sin ganas de nada", 0.2, 0.6, 0.6, 0.1, 0.4),
    ("Hoy es un día maravilloso, me siento feliz", 0.0, 0.0, 0.1, 0.95, 0.1),
    ("No tengo interés en mis hobbies", 0.1, 0.4, 0.5, 0.2, 0.3),
]

# Expandir un poco con variaciones para tener volumen mínimo
expanded_data = []
for text, pain, fatigue, anxiety, mood, sleep in data:
    expanded_data.append({"text": text, "pain": pain, "fatigue": fatigue, "anxiety": anxiety, "mood": mood, "sleep": sleep})
    # Variaciones simples
    expanded_data.append({"text": text.replace("Me ", "Hoy me "), "pain": pain, "fatigue": fatigue, "anxiety": anxiety, "mood": mood, "sleep": sleep})
    expanded_data.append({"text": text + ".", "pain": pain, "fatigue": fatigue, "anxiety": anxiety, "mood": mood, "sleep": sleep})

df = pd.DataFrame(expanded_data)
output_dir = "scripts/data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "train_v1.csv")
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} samples at {output_path}")
