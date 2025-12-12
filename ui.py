import sys
import cv2
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap


class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load file main.ui vào widget này
        uic.loadUi("main.ui", self)

        # In stylesheet ra console để debug nếu cần
        print("Current stylesheet:")
        print(self.styleSheet())

        # Mở camera mặc định
        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(frame.data, w, h, bytes_per_line,
                      QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.Camera.setPixmap(pix)
        self.Camera.setScaledContents(True)

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec())
