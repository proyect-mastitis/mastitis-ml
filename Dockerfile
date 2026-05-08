FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:main", "--host", "0.0.0.0", "--port", "7860"]