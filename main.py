from ultralytics import YOLO
import cv2

modelo_urbano = YOLO('runs/detect/train3/weights/best.pt')
modelo_geral = YOLO('yolov8m.pt')

cap = cv2.VideoCapture(0)
print(modelo_urbano.names)
print(modelo_geral.names)
while True:
    acesso, frame = cap.read()
    if not acesso:
        break

    res_urbano = modelo_urbano(frame, conf=0.5, verbose=False)[0]

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

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()