from pygame import mixer
import pyttsx3
import time
from ultralytics import YOLO
import cv2
import vosk
import sounddevice as sd
import json
import threading
import easyocr

modelo_urbano = YOLO('runs/detect/train/weights/best.pt')
modelo_comando = vosk.Model('vosk-model-small-pt-0.3')
rec = vosk.KaldiRecognizer(modelo_comando, 16000)

reader = easyocr.Reader(['pt'])

ultimo_texto = ''

mixer.init()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
cap.set(cv2.CAP_PROP_CONTRAST, 300)
cap.set(cv2.CAP_PROP_SATURATION, 120)
cap.set(cv2.CAP_PROP_SHARPNESS, 180)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

classes={1: 'Buraco', 2: 'Cone', 3: 'Escadas', 4: 'Faixa de pedestre', 5: 'Fogo', 6: 'Fumaça', 7: 'Piso Molhado', 8: 'bollard', 9: 'Saída', 10: 'Semáforo de Pedestre Verde', 11: 'Semáforo de Pedestre Vermelho'}

def informa (som, texto):   
    engine = pyttsx3.init()
    mixer.music.load(som)
    mixer.music.set_volume(0.7)
    mixer.music.play()
    mixer.fadeout(1)
    engine.say(texto)
    engine.runAndWait()
    time.sleep(1)

def callback(indata, frames, time_audio, status):
    global ultimo_texto
    if rec.AcceptWaveform(bytes(indata)):
        result = json.loads(rec.Result())
        text = result.get('text', '').strip().lower()
        if not text:
            return
        print("Comando:", text)
        ultimo_texto = text

def iniciar_audio():
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype='int16',
        channels=1,
        callback=callback
    ):
        while True:
            sd.sleep(1000)

threading.Thread(
    target=iniciar_audio,
    daemon=True
).start()

while True:
    acesso, frame = cap.read()
    if not acesso:
        break
    if ultimo_texto == "leitura":
        resultado = reader.readtext(
            frame,
            detail=0,
            paragraph=True
        )
        texto_final = " ".join(resultado)
        if texto_final.strip() != "":
            informa(
                "comum.mp3",
                texto_final
            )
    ultimo_texto = ''
    largura = frame.shape[1]
    centro_tela = largura / 2

    res_perigo = modelo_urbano(
        frame,
        conf=0.7,
        classes=[5,6,7],
        verbose=False
    )[0]

    res_urbano = modelo_urbano(
        frame,
        conf=0.7,
        classes=[1,2,3,4,8,9,10,11],
        verbose=False
    )[0]
    if len(res_perigo.boxes) > 0:

        cls = int(res_perigo.boxes.cls[0])
        nome = classes[cls]

        caixas = res_perigo.boxes.xyxy.cpu().numpy()

        for caixa in caixas:

            x1, y1, x2, y2 = map(int, caixa)

            centro = (x1 + x2) / 2

            if centro < centro_tela*0.8:
                direcao = "esquerda"

            elif centro > centro_tela*1.2:
                direcao = "direita"

            else:
                direcao = "frente"

        informa(
            "perigo.mp3",
            f"{nome} à sua {direcao}"
        )

    elif len(res_urbano.boxes) > 0:

        cls = int(res_urbano.boxes.cls[0])
        nome = classes[cls]

        caixas = res_urbano.boxes.xyxy.cpu().numpy()

        for caixa in caixas:

            x1, y1, x2, y2 = map(int, caixa)

            centro = (x1 + x2) / 2

            if centro < centro_tela*0.8:
                direcao = "esquerda"

            elif centro > centro_tela*1.2:
                direcao = "direita"

            else:
                direcao = "frente"

        informa(
            "comum.mp3",
            f"{nome} à sua {direcao}"
        )
    
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break          
cap.release()
cv2.destroyAllWindows()