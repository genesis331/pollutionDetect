FROM python:3.8.13
ARG TARGETARCH

WORKDIR /app
COPY . .

RUN python3 -m venv /opt/venv
RUN . /opt/venv/bin/activate && pip3 install -r requirements-$TARGETARCH.txt && pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
RUN apt update && apt install -y libgl1-mesa-dev

EXPOSE 8501
CMD . /opt/venv/bin/activate && exec streamlit run index.py --logger.level=debug