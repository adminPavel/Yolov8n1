from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # загрузите предварительно обученную модель YOLOv8n

#model.train(data="coco128.yaml")  # обучите модель
#model.val()  # оцените производительность модели на наборе проверки
model.predict(source="C:/Users/Pavel/Downloads/cars(720p).mp4", save=True)  # предсказать по изображению или видео
model.export(format="onnx")  # экспортируйте модель в формат ONNX