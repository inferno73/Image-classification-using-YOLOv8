from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(data=r'C:\Users\Korisnik\Documents\Coding\CourseCVEmaterial\weather-dataset',
            epochs=20, imgsz=64)

