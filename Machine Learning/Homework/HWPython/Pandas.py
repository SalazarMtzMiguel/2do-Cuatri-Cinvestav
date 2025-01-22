import pandas as pd
from sqlalchemy import create_engine

# Reemplaza con tus credenciales y detalles de la base de datos
username = 'root'
password = ''
host = 'localhost'
database = 'employees'

# Crea la cadena de conexión
connection_string = f'mysql+pymysql://{username}:{password}@{host}/{database}'

# Crea el motor de conexión
engine = create_engine(connection_string)

# Ejecuta una consulta y carga los datos en un DataFrame de pandas
query = 'SELECT * FROM tu_tabla'
df = pd.read_sql(query, engine)

# Muestra los primeros registros del DataFrame
print(df.head())