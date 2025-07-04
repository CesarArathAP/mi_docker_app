# Usar imagen base de Python
FROM python:3.10

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivo al contenedor
COPY app.py .

# Comando que se ejecuta al iniciar el contenedor
CMD ["python", "app.py"]
