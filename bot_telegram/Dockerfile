# Imagen base
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
COPY bot ./bot

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Ejecutar el bot
CMD ["python", "bot/bot.py"]