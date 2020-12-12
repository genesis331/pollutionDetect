virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
apt-get update
apt-get install -y libgl1-mesa-dev
streamlit run index.py