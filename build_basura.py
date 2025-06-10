# build_basura.py

import os
import sys
import tensorflow_datasets as tfds

# 1) Añadimos la carpeta my_datasets al PYTHONPATH para que Python encuentre tu módulo.
sys.path.append(os.path.abspath("my_datasets"))

# 2) Importa el módulo 'basura' para registrar el Builder en TFDS.
import basura  # Debe corresponder a my_datasets/basura/basura.py

# 3) Construye el dataset (leer imágenes, crear TFRecords).
builder = tfds.builder(
    "basura",
    data_dir="C:/Users/Dante/Downloads/DLvidrio/tensorflow_datasets"
)
builder.download_and_prepare()

print("✅ Build de 'basura' completado.")
