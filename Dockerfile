FROM python:3.8.13

WORKDIR /app
COPY . .

RUN apt update && apt install -y libgl1-mesa-dev
RUN python3 -m venv /opt/venv
RUN . /opt/venv/bin/activate && pip install -r requirements.txt

EXPOSE 8501
CMD . /opt/venv/bin/activate && exec streamlit run index.py --logger.level=debug