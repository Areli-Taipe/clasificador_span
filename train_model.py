import pandas as pd  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.pipeline import make_pipeline  
import joblib  

# Cargar los datos  
data = pd.read_csv('correos.csv')  

# Separar las características y las etiquetas  
X = data['Mensaje']  
y = data['Etiqueta']  

# Crear un modelo pipeline  
model = make_pipeline(CountVectorizer(), MultinomialNB())  

# Entrenar el modelo  
model.fit(X, y)  

# Guardar el modelo entrenado  
joblib.dump(model, 'spam_model.pkl')  

print("Modelo entrenado y guardado exitosamente.")  