import easyocr
from ultralytics import YOLO
import cv2
import pyttsx3
import sounddevice as sd
import json
import vosk

reader = easyocr.Reader(['pt'])
modelo_urbano = YOLO('runs/detect/train3/weights/best.pt')
modelo_geral = YOLO('yolov8m.pt')

modelo_comando = vosk.Model('vosk-model-small-pt-0.3')
rec = vosk.KaldiRecognizer(modelo_comando, 16000)

def callback(indata, frames, time, status):
    if rec.AcceptWaveform(bytes(indata)):
        result = json.loads(rec.Result())
        text = result.get('text', '')
        if text:
            print('Reconhecido', text)
            
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
    print('Fale agora...')
    sd.sleep(60000)

engine = pyttsx3.init()
engine.setProperty('volume', 1.0)
cap = cv2.VideoCapture(0)
print(modelo_urbano.names)
print(modelo_geral.names)

while True:
    acesso, frame = cap.read()
    if not acesso:
        break
    res_texto = reader.readtext(frame, detail = 0, paragraph=True)
    res_urbano = modelo_urbano(frame, conf=0.45, verbose=False)[0]
    
    if len(res_urbano.boxes) > 0:
        caixas = res_urbano.boxes.xyxy.cpu().numpy()
        confs = res_urbano.boxes.conf.cpu().numpy()
        clss = res_urbano.boxes.cls.cpu().numpy().astype(int)
        nomes = modelo_urbano.names
        cor = (0, 255, 0)
    else:
        res_geral = modelo_geral(frame, conf=0.45, verbose=False)[0]
        caixas = res_geral.boxes.xyxy.cpu().numpy()
        confs = res_geral.boxes.conf.cpu().numpy()
        clss = res_geral.boxes.cls.cpu().numpy().astype(int)
        nomes = modelo_geral.names
        cor = (255, 0, 0)

    for caixa, conf, cls in zip(caixas, confs, clss):
        if cls!=12 and cls!=24:
            x1, y1, x2, y2 = map(int, caixa)
            nome = nomes[cls]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(
                frame,
                f'{nome} {conf:.2f}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                cor,
                2
            )
            engine.say(str(nome))
            engine.runAndWait()

    
    for fala in res_texto:
        print(fala)
        engine.say(str(fala))
        engine.runAndWait()

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
