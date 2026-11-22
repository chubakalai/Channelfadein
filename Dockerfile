FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/data
WORKDIR /app
# stdlib only: no requirements.txt needed
COPY d.py f.py a.py ht.py ehl.py ./
# runs as root: a Fly volume is mounted root-owned, a non-root user could not write to it
CMD ["python", "ehl.py"]
