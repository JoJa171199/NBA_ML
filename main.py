import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI(title="Courtside Alpha Multi-Stat Engine")

# --- 1. CONFIGURACIÓN DE MODELOS ---
MODEL_FILES = {
    "pts": "nba_model_pts_model.pkl",
    "ast": "nba_model_ast_model.pkl",
    "reb": "nba_model_reb_model.pkl",
    "blk": "nba_model_blk_model.pkl",
    "stl": "nba_model_stl_model.pkl"
}

loaded_objects = {}

print("--- INICIANDO CARGA DE MODELOS ---")
for stat_name, filename in MODEL_FILES.items():
    try:
        if os.path.exists(filename):
            loaded_objects[stat_name] = joblib.load(filename)
            type_obj = type(loaded_objects[stat_name])
            print(f"✅ Archivo cargado: {stat_name.upper()} ({filename}) - Tipo: {type_obj}")
        else:
            print(f"⚠️ Archivo no encontrado: {filename}")
    except Exception as e:
        print(f"❌ Error cargando {filename}: {e}")

# --- 2. DATA SCHEMAS ---
class PredictionRequest(BaseModel):
    player_name: str
    opponent: str
    is_home: bool = True

# --- 3. MOCK DATA BASE ---
PLAYER_BASE_STATS = {
    "LeBron James":  {"avg_pts": 25.4, "avg_ast": 8.1, "avg_reb": 7.3, "avg_blk": 0.6, "avg_stl": 1.1, "momentum": 1.05},
    "Stephen Curry": {"avg_pts": 28.1, "avg_ast": 5.0, "avg_reb": 4.4, "avg_blk": 0.3, "avg_stl": 0.8, "momentum": 0.98},
    "Luka Doncic":   {"avg_pts": 33.9, "avg_ast": 9.8, "avg_reb": 9.2, "avg_blk": 0.5, "avg_stl": 1.4, "momentum": 1.10},
    "Nikola Jokic":  {"avg_pts": 26.4, "avg_ast": 9.0, "avg_reb": 12.4, "avg_blk": 0.9, "avg_stl": 1.2, "momentum": 1.02},
    "Jayson Tatum":  {"avg_pts": 27.1, "avg_ast": 4.9, "avg_reb": 8.1, "avg_blk": 0.6, "avg_stl": 1.0, "momentum": 1.00},
    "DEFAULT":       {"avg_pts": 15.0, "avg_ast": 3.0, "avg_reb": 4.0, "avg_blk": 0.4, "avg_stl": 0.7, "momentum": 1.0}
}

OPPONENT_STATS = {
    "BOS": {"def_rating": 110.5, "pace": 98.2},
    "DEN": {"def_rating": 114.2, "pace": 97.5},
    "LAL": {"def_rating": 115.0, "pace": 101.0},
    "DEFAULT": {"def_rating": 112.0, "pace": 99.0}
}

# --- 4. ENDPOINT DE PREDICCIÓN MASIVA ---
@app.post("/predict_all")
def predict_performance(request: PredictionRequest):
    player_stats = PLAYER_BASE_STATS.get(request.player_name, PLAYER_BASE_STATS["DEFAULT"])
    opp_stats = OPPONENT_STATS.get(request.opponent, OPPONENT_STATS["DEFAULT"])
    
    predictions = {}
    
    for stat, obj in loaded_objects.items():
        try:
            # --- LOGICA DE EXTRACCIÓN DEL MODELO ---
            model_to_use = obj
            
            # Si el objeto cargado es un diccionario, buscamos el modelo dentro
            if isinstance(obj, dict):
                print(f"🔍 Inspeccionando llaves del diccionario para {stat}: {obj.keys()}")
                # Intentamos adivinar la llave común
                possible_keys = ['model', 'regressor', 'estimator', 'learner', 'pipeline']
                found = False
                for key in possible_keys:
                    if key in obj:
                        model_to_use = obj[key]
                        found = True
                        break
                
                if not found:
                    # Si no encontramos una llave conocida, asumimos que NO podemos predecir
                    print(f"❌ No se encontró modelo en el dict de {stat}. Llaves disponibles: {list(obj.keys())}")
                    predictions[stat] = -999 # Código de error visible
                    continue
            
            # --- PREPARACIÓN DEL DATAFRAME ---
            input_data = pd.DataFrame([{
                'avg_pts': player_stats['avg_pts'], 
                'avg_ast': player_stats['avg_ast'],
                'avg_reb': player_stats['avg_reb'],
                'def_rating': opp_stats['def_rating'],
                'pace': opp_stats['pace'],
                'is_home': 1 if request.is_home else 0,
                'momentum': player_stats['momentum']
            }])
            
            # --- PREDICCIÓN ---
            pred_val = model_to_use.predict(input_data)
            
            val = float(pred_val[0]) if isinstance(pred_val[0], (np.float64, float)) else float(pred_val[0][0])
            predictions[stat] = round(val, 1)
            
        except Exception as e:
            print(f"🔥 Error crítico prediciendo {stat}: {e}")
            predictions[stat] = -1 

    return {
        "player": request.player_name,
        "opponent": request.opponent,
        "predictions": predictions,
        "context": {
            "momentum_label": "High" if player_stats['momentum'] > 1.05 else "Normal",
            "matchup_difficulty": "Hard" if opp_stats['def_rating'] < 111 else "Neutral"
        }
    }
