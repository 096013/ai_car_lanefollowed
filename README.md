# ai_car_lanefollowed
rasp pi 5+ picamera + tt_motor
ipconfig
arp –a
ssh pi@172.20.10.XX
cd  ./ ai_car_project

開發環境安裝 (Step-by-Step)
建立隔離環境：避免套件衝突

python -m venv --system-site-packages .venv 
source .venv/bin/activate

安裝必要套件：
pip install Flask ultralytics opencv-python adafruit-circuitpython-pca9685 gpiozero

準備 AI 模型：下載官方最小權重
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -O yolo_model.pt 

python app.py

