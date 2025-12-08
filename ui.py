import sys
import cv2
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap


class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        # load file .ui mới (QWidget)
        uic.loadUi("main.ui", self)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(frame.data, w, h, bytes_per_line,
                      QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.Camera.setPixmap(pix)          # tên QLabel trong .ui
        self.Camera.setScaledContents(True)

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec())