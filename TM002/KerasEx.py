import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier")
        self.setGeometry(100, 100, 800, 600)  # 너비: 800, 높이: 800

        self.model = load_model("D:/AI/TM002/model/keras_Model.h5", compile=False)
        self.class_names = open("D:/AI/TM002/model/labels.txt", "r").readlines()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)

        self.load_button = QPushButton("이미지 업로드", self)
        self.load_button.clicked.connect(self.load_image)

        self.classify_button = QPushButton("이미지 분류", self)
        self.classify_button.clicked.connect(self.classify_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.classify_button)

        self.central_widget.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image files (*.jpg *.jpeg *.png)")
        if file_path:
            image = Image.open(file_path).convert("RGB")
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)
            self.image_array = np.asarray(image)

    def classify_image(self):
        if hasattr(self, 'image_array'):
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            normalized_image_array = (self.image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array
            prediction = self.model.predict(data)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]
            if confidence_score < 0.99:
                class_name = "   알 수 없음"
            else:
                class_name = self.class_names[index]
            result_str = f"Class: {class_name[2:]}\nConfidence Score: {confidence_score}"
            self.show_result_window(result_str)
        else:
            print("Please load an image first.")

    def show_result_window(self, result_str):
        result_box = QMessageBox()
        result_box.setWindowTitle("Classification Result")
        result_box.setText(result_str)
        result_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())
