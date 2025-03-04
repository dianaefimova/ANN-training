import ast
import pickle
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

app = FastAPI()

# Allow CORS for your React frontend running on localhost:3000 for testing
origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


model = load_model('code_model.keras')

with open('model-ct.pickle', 'rb') as f:
    ct = pickle.load(f)

with open('model-scaler_x.pickle', 'rb') as f:
    scaler_x = pickle.load(f)


class CodeRequest(BaseModel):
    code: str  

# Function to extract features from Python code
def extract_features_from_code(code):
    """Extract key complexity features from raw Python code."""
    tree = ast.parse(code)
    
    num_lines = len(code.splitlines())  
    num_functions = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))  
    num_loops = sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))  
    num_conditionals = sum(isinstance(node, ast.If) for node in ast.walk(tree))  
    
    has_recursion = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):  
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                    if sub_node.func.id == node.name:  
                        has_recursion = 1
                        break
    
    return {
        "Programming_Language": "Python",  # Hardcoded for now, will be changed later
        "Num_Lines": num_lines,
        "Num_Functions": num_functions,
        "Num_Loops": num_loops,
        "Num_Conditionals": num_conditionals,
        "Has_Recursion": has_recursion
    }

# API endpoint to predict code difficulty
@app.post("/predict")
async def predict_difficulty(request: CodeRequest):
    try:
        features = extract_features_from_code(request.code)
        df = pd.DataFrame([features])  # Convert to DataFrame

        X_new = ct.transform(df)
        X_new = scaler_x.transform(X_new)

        y_pred_proba = model.predict(X_new)
        y_pred = y_pred_proba.argmax(axis=1)

        difficulty_labels = ['Easy', 'Medium', 'Hard']
        predicted_difficulty = difficulty_labels[y_pred[0]]

        return {"difficulty": predicted_difficulty}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API (only in local development) for testing locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
