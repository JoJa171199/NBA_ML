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

# --- 3. MOCK DATA BASE (Tus promedios base) ---
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

# --- 4. FUNCIÓN GENERADORA DE VARIABLES (Feature Engineering Simulado) ---
def generate_features_for_model(stat_type, player_base, opp_stats, is_home):
    """
    Esta función traduce los datos básicos a las columnas complejas que pide el modelo.
    Basado en los logs: H2H_AVG_X, L10_X, IS_B2B, IS_HOME, L10_STD_X
    """
    # Determinamos la estadística base (pts, ast, etc)
    base_val = player_base.get(f"avg_{stat_type}", 0)
    
    # Factor de impulso para simular "Last 10 Games" (L10)
    momentum = player_base.get("momentum", 1.0)
    l10_val = base_val * momentum
    
    # Diccionario con las columnas EXACTAS que piden tus logs
    # NOMBRES DE COLUMNAS DEBEN SER MAYÚSCULAS según tus logs (H2H_AVG_PTS)
    suffix = stat_type.upper() # PTS, AST, REB...
    
    features = {
        # Variables de tiempo/lugar
        "IS_HOME": 1 if is_home else 0,
        "IS_B2B": 0,  # Asumimos que no es Back-to-Back para el MVP
        "REST_DAYS": 1, # Descanso estándar
        
        # Variables del Oponente
        "OPP_DEF_RATING": opp_stats["def_rating"],
        "PACE": opp_stats["pace"],
        "OPP_TEAM": 0, # Placeholder numérico si el modelo usó LabelEncoder, riesgo menor
        
        # Variables Específicas de la Estadística (L10, H2H, STD)
        f"L10_{suffix}": l10_val,
        f"L10_STD_{suffix}": base_val * 0.25, # Simulamos desviación estándar del 25%
        f"H2H_AVG_{suffix}": base_val,        # Asumimos promedio histórico similar al actual
        f"AVG_{suffix}": base_val,            # A veces piden el promedio simple
    }
    
    # Crear DataFrame de una fila
    return pd.DataFrame([features])

# --- 5. ENDPOINT ---
@app.post("/predict_all")
def predict_performance(request: PredictionRequest):
    player_stats = PLAYER_BASE_STATS.get(request.player_name, PLAYER_BASE_STATS["DEFAULT"])
    opp_stats = OPPONENT_STATS.get(request.opponent, OPPONENT_STATS["DEFAULT"])
    
    predictions = {}
    
    for stat, obj in loaded_objects.items():
        try:
            # 1. Extraer el modelo real del diccionario
            if isinstance(obj, dict):
                model = obj.get('model')
                # Si tienes la lista de features guardada, úsala para filtrar
                expected_features = obj.get('features', []) 
            else:
                model = obj
                expected_features = []

            if not model:
                predictions[stat] = -1
                continue

            # 2. Generar DataFrame con nombres correctos
            input_df = generate_features_for_model(stat, player_stats, opp_stats, request.is_home)
            
            # 3. CRÍTICO: Asegurar que el DataFrame tenga SOLO las columnas que el modelo quiere
            # Si guardaste la lista de features en el pickle, filtramos:
            if hasattr(model, "feature_names_in_"):
                # Scikit-learn moderno guarda esto
                cols_needed = model.feature_names_in_
            elif expected_features:
                cols_needed = expected_features
            else:
                # Si no sabemos, mandamos todo lo que generamos (arriesgado pero mejor que nada)
                cols_needed = input_df.columns
            
            # Reordenar y rellenar columnas faltantes con 0
            final_input = pd.DataFrame()
            for col in cols_needed:
                if col in input_df.columns:
                    final_input[col] = input_df[col]
                else:
                    final_input[col] = 0.0 # Relleno de emergencia para columnas no calculadas
            
            # 4. Predecir
            pred_val = model.predict(final_input)
            val = float(pred_val[0]) if isinstance(pred_val[0], (np.float64, float)) else float(pred_val[0][0])
            predictions[stat] = round(val, 1)
            
        except Exception as e:
            print(f"🔥 Error en {stat}: {e}")
            predictions[stat] = -999 # Código de error para UI

    return {
        "player": request.player_name,
        "opponent": request.opponent,
        "predictions": predictions,
        "context": {
            "momentum_label": "High" if player_stats['momentum'] > 1.05 else "Normal",
            "matchup_difficulty": "Hard" if opp_stats['def_rating'] < 111 else "Neutral"
        }
    }
