# Configuration Gunicorn pour Render
bind = "0.0.0.0:10000"
workers = 1
worker_class = "sync"
timeout = 120  # 2 minutes au lieu de 30 secondes
keepalive = 5
loglevel = "info"
accesslog = "-"
errorlog = "-"