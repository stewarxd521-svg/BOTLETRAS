# Render usa este Procfile si no tiene render.yaml, o puedes usarlo directamente.
# Gunicorn NO lanza el hilo de fondo del bot; usa bot.py directamente en su lugar.
# El startCommand correcto ya está definido en render.yaml: python bot.py
#
# Si prefieres gunicorn con un worker + el loop de fondo integrado, usa:
# web: gunicorn --workers 1 --threads 2 --bind 0.0.0.0:$PORT "bot:app"
# PERO necesitas mover el arranque del loop a un before_first_request o similar.
#
# Recomendación: usa el comando simple que ya tiene render.yaml:
web: python bot.py
