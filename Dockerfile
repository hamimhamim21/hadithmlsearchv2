# Stage 1
FROM python:3.10-slim-buster AS builder

WORKDIR /flask-app

RUN python3 -m venv venv
ENV VIRTUAL_ENV=/flask-app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install specific versions of torch, torchvision, and torchaudio in the virtual environment
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu
# Stage 2
FROM python:3.10-slim-buster AS runner

WORKDIR /flask-app

COPY --from=builder /flask-app/venv venv
COPY app.py .
COPY translator.py .
COPY utils.py .
COPY hadith_embeddings.db .
COPY hadith_search_full.db .


ENV VIRTUAL_ENV=/flask-app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV FLASK_APP=app/app.py

EXPOSE 5000

CMD ["python", "-m" , "flask", "run", "--host=0.0.0.0"]
