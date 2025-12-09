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
            print(f"✅ Modelo cargado: {stat_name.upper()}")
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

# --- 4. ENGINE DE VARIABLES ---
def generate_features_dict(stat_type, player_base, opp_stats, is_home):
    """
    Genera un diccionario con TODAS las posibles variables que el modelo podría pedir.
    Nombres en MAYÚSCULAS para coincidir con el entrenamiento.
    """
    base_val = player_base.get(f"avg_{stat_type}", 0)
    momentum = player_base.get("momentum", 1.0)
    l10_val = base_val * momentum
    suffix = stat_type.upper() # PTS, AST...
    
    # Diccionario maestro de variables disponibles
    features = {
        "IS_HOME": 1 if is_home else 0,
        "IS_B2B": 0,
        "REST_DAYS": 1,
        "OPP_DEF_RATING": opp_stats["def_rating"],
        "PACE": opp_stats["pace"],
        
        # Variables dinámicas por tipo
        f"L10_{suffix}": l10_val,
        f"L10_STD_{suffix}": base_val * 0.25,
        f"H2H_AVG_{suffix}": base_val,
        f"AVG_{suffix}": base_val,
        
        # A veces el modelo pide las de otros tipos (cross-features)
        "AVG_PTS": player_base.get("avg_pts", 0),
        "AVG_AST": player_base.get("avg_ast", 0),
        "AVG_REB": player_base.get("avg_reb", 0),
    }
    return features

# --- 5. ENDPOINT ---
@app.post("/predict_all")
def predict_performance(request: PredictionRequest):
    player_stats = PLAYER_BASE_STATS.get(request.player_name, PLAYER_BASE_STATS["DEFAULT"])
    opp_stats = OPPONENT_STATS.get(request.opponent, OPPONENT_STATS["DEFAULT"])
    
    predictions = {}
    
    for stat, obj in loaded_objects.items():
        try:
            # 1. Recuperar Modelo y Lista de Features
            if isinstance(obj, dict):
                model = obj.get('model')
                # LA CLAVE DEL ÉXITO: Usar la lista 'features' guardada en el pickle
                required_columns = obj.get('features') 
            else:
                model = obj
                required_columns = getattr(model, "feature_names_in_", None)

            if not model:
                predictions[stat] = -1
                continue

            # 2. Generar nuestros datos disponibles
            available_data = generate_features_dict(stat, player_stats, opp_stats, request.is_home)
            
            # 3. ALINEACIÓN FORZADA (Force Align)
            if required_columns is not None:
                # Creamos el DataFrame usando LA LISTA DEL MODELO como columnas
                # Esto garantiza el orden perfecto.
                input_df = pd.DataFrame(columns=required_columns)
                input_df.loc[0] = 0.0 # Inicializamos todo a 0 (relleno de seguridad)
                
                # Rellenamos solo lo que tenemos
                for col in required_columns:
                    if col in available_data:
                        input_df.at[0, col] = available_data[col]
                    else:
                        # Si falta una columna, se queda en 0.0
                        pass
            else:
                # Fallback por si no encontramos la lista (esperemos que no pase)
                input_df = pd.DataFrame([available_data])

            # 4. Predecir
            pred_val = model.predict(input_df)
            val = float(pred_val[0]) if isinstance(pred_val[0], (np.float64, float)) else float(pred_val[0][0])
            predictions[stat] = round(val, 1)
            
        except Exception as e:
            # Imprimimos el error pero no rompemos el loop
            print(f"🔥 Error en {stat}: {e}")
            predictions[stat] = -999

    return {
        "player": request.player_name,
        "opponent": request.opponent,
        "predictions": predictions,
        "context": {
            "momentum_label": "High" if player_stats['momentum'] > 1.05 else "Normal",
            "matchup_difficulty": "Hard" if opp_stats['def_rating'] < 111 else "Neutral"
        }
    }
