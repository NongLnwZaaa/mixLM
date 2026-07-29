import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess(df):
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
    
    # กำหนดกลุ่มประเภทของ Feature
    numeric_cols = ["RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
    categorical_cols = ["ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    
    # สร้าง Preprocessor ขึ้นมาโดยตรงในโค้ด (ไม่ต้องโหลดไฟล์ preprocessor.pkl)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit และ Transform ข้อมูล
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor