FROM python:3.8.16

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/train.py .
COPY src/eval.py .

RUN python train.py
RUN python eval.py

EXPOSE 9021
CMD ["streamlit", "run", "train_eval_script.py", "--server.port", "9021", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
