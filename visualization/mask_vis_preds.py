from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QPushButton, QComboBox, QLabel, QTableWidget,
                             QTableWidgetItem)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import sys
import os
from natsort import natsorted
import numpy as np
from PIL import Image


class HoverLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image_data = None
        self.value_callback = None
        self.orig_width = None
        self.orig_height = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseMoveEvent(self, event):
        if self.image_data is not None and self.value_callback is not None:
            # Get the display size and original image size
            display_size = self.size()
            if self.orig_width is None:
                self.orig_width = self.image_data.shape[1]
                self.orig_height = self.image_data.shape[0]

            # Calculate the actual display dimensions maintaining aspect ratio
            aspect_ratio = self.orig_width / self.orig_height
            display_aspect = display_size.width() / display_size.height()

            if aspect_ratio > display_aspect:
                display_w = display_size.width()
                display_h = int(display_w / aspect_ratio)
                offset_x = 0
                offset_y = (display_size.height() - display_h) // 2
            else:
                display_h = display_size.height()
                display_w = int(display_h * aspect_ratio)
                offset_x = (display_size.width() - display_w) // 2
                offset_y = 0

            # Get mouse position relative to the actual image display area
            mouse_x = event.pos().x() - offset_x
            mouse_y = event.pos().y() - offset_y

            # Convert to image coordinates
            if 0 <= mouse_x < display_w and 0 <= mouse_y < display_h:
                img_x = int(mouse_x * self.orig_width / display_w)
                img_y = int(mouse_y * self.orig_height / display_h)

                if 0 <= img_x < self.orig_width and 0 <= img_y < self.orig_height:
                    self.value_callback(self.image_data[img_y, img_x])


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Segmentation Viewer")
        self.setGeometry(100, 100, 1800, 1000)  # Increased height for predictions

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)

        # Left side with images
        left_layout = QVBoxLayout()

        # Controls layout
        controls_layout = QHBoxLayout()
        self.folder_combo = QComboBox()
        self.folder_combo.addItems(['train', 'validation'])
        self.folder_combo.currentTextChanged.connect(self.load_folder)
        self.image_name_label = QLabel()
        self.value_label = QLabel()
        self.value_label.setStyleSheet("QLabel { background-color : white; padding: 5px; }")
        controls_layout.addWidget(self.folder_combo)
        controls_layout.addWidget(self.image_name_label)
        controls_layout.addWidget(self.value_label)
        left_layout.addLayout(controls_layout)

        # Ground truth images layout
        image_layout = QHBoxLayout()
        self.rgb_label = QLabel()
        self.colored_seg_label = HoverLabel()
        self.raw_seg_label = HoverLabel()

        for label in [self.rgb_label, self.colored_seg_label, self.raw_seg_label]:
            label.setFixedSize(512, 512)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 1px solid black")
            image_layout.addWidget(label)

        left_layout.addLayout(image_layout)

        # Prediction images layout
        pred_layout = QHBoxLayout()
        self.pred_value_label = QLabel()
        self.pred_value_label.setStyleSheet("QLabel { background-color : white; padding: 5px; }")
        self.pred_label = HoverLabel()
        self.colored_pred_label = HoverLabel()

        for label in [self.pred_label, self.colored_pred_label]:
            label.setFixedSize(512, 512)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 1px solid black")
            pred_layout.addWidget(label)

        pred_layout.addWidget(self.pred_value_label)

        # Create a widget to hold the prediction layout
        self.pred_widget = QWidget()
        self.pred_widget.setLayout(pred_layout)
        self.pred_widget.hide()  # Initially hidden
        left_layout.addWidget(self.pred_widget)

        # Navigation buttons
        button_layout = QHBoxLayout()
        prev_button = QPushButton("Previous")
        next_button = QPushButton("Next")
        prev_button.clicked.connect(self.prev_image)
        next_button.clicked.connect(self.next_image)
        button_layout.addWidget(prev_button)
        button_layout.addWidget(next_button)
        left_layout.addLayout(button_layout)

        # Right side - Tables
        right_layout = QVBoxLayout()

        # Ground truth class table
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(2)
        self.class_table.setHorizontalHeaderLabels(['Class Name', 'Train ID'])
        self.class_table.setFixedWidth(200)

        # Prediction class table
        self.pred_class_table = QTableWidget()
        self.pred_class_table.setColumnCount(2)
        self.pred_class_table.setHorizontalHeaderLabels(['Pred Class', 'Train ID'])
        self.pred_class_table.setFixedWidth(200)
        self.pred_class_table.hide()  # Initially hidden

        right_layout.addWidget(self.class_table)
        right_layout.addWidget(self.pred_class_table)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.base_path = "/Users/ibk5106/Desktop/research/few_shot/uwss_v2"
        self.current_index = 0

        self.label_dictionary = {
            0: {'name': 'unlabeled', 'train_id': 255, 'color': (0, 0, 0)},
            1: {'name': 'crab', 'train_id': 0, 'color': (255, 178, 204)},
            2: {'name': 'crocodile', 'train_id': 1, 'color': (0, 0, 128)},
            3: {'name': 'dolphin', 'train_id': 2, 'color': (0, 0, 178)},
            4: {'name': 'frog', 'train_id': 3, 'color': (51, 51, 51)},
            5: {'name': 'nettles', 'train_id': 4, 'color': (0, 0, 0)},
            6: {'name': 'octopus', 'train_id': 5, 'color': (51, 306, 51)},
            7: {'name': 'otter', 'train_id': 6, 'color': (102, 102, 102)},
            8: {'name': 'penguin', 'train_id': 7, 'color': (10, 0, 255)},
            9: {'name': 'polar_bear', 'train_id': 8, 'color': (255, 178, 102)},
            10: {'name': 'sea_anemone', 'train_id': 9, 'color': (153, 255, 255)},
            11: {'name': 'sea_urchin', 'train_id': 10, 'color': (0, 255, 255)},
            12: {'name': 'seahorse', 'train_id': 11, 'color': (255, 153, 153)},
            13: {'name': 'seal', 'train_id': 12, 'color': (255, 0, 0)},
            14: {'name': 'shark', 'train_id': 13, 'color': (178, 178, 0)},
            15: {'name': 'shrimp', 'train_id': 14, 'color': (255, 102, 178)},
            16: {'name': 'star_fish', 'train_id': 15, 'color': (153, 204, 255)},
            17: {'name': 'stingray', 'train_id': 16, 'color': (255, 153, 178)},
            18: {'name': 'squid', 'train_id': 17, 'color': (229, 0, 0)},
            19: {'name': 'turtle', 'train_id': 18, 'color': (0, 153, 0)},
            20: {'name': 'whale', 'train_id': 19, 'color': (0, 229, 77)},
            21: {'name': 'nudibranch', 'train_id': 20, 'color': (242, 243, 245)},
            22: {'name': 'coral', 'train_id': 21, 'color': (0, 0, 77)},
            23: {'name': 'rock', 'train_id': 22, 'color': (0, 178, 0)},
            24: {'name': 'water', 'train_id': 23, 'color': (255, 77, 77)},
            25: {'name': 'sand', 'train_id': 24, 'color': (178, 0, 0)},
            26: {'name': 'plant', 'train_id': 25, 'color': (255, 178, 255)},
            27: {'name': 'human', 'train_id': 26, 'color': (128, 128, 0)},
            28: {'name': 'reef', 'train_id': 27, 'color': (0, 0, 255)},
            29: {'name': 'others', 'train_id': 28, 'color': (178, 178, 178)},
            30: {'name': 'dynamic', 'train_id': 29, 'color': (0, 77, 0)},  ## begining of UWSv2 new semantic categories
            31: {'name': 'beaver', 'train_id': 30, 'color': (151, 191, 201)},
            32: {'name': 'duck', 'train_id': 31, 'color': (153, 102, 51)},
            33: {'name': 'dugong', 'train_id': 32, 'color': (229, 0, 229)},
            34: {'name': 'hippo', 'train_id': 33, 'color': (255, 255, 178)},
            35: {'name': 'lobster', 'train_id': 34, 'color': (222, 128, 4)},
            36: {'name': 'platypus', 'train_id': 35, 'color': (102, 87, 110)},
            37: {'name': 'nautilus', 'train_id': 36, 'color': (229, 229, 0)},
            38: {'name': 'sea_cucumber', 'train_id': 37, 'color': (229, 255, 255)},
            39: {'name': 'sea_lion', 'train_id': 38, 'color': (173, 173, 0)},
            40: {'name': 'sea_snake', 'train_id': 39, 'color': (0, 0, 102)},
            41: {'name': 'barracouta', 'train_id': 40, 'color': (77, 0, 0)},
            42: {'name': 'billfish', 'train_id': 41, 'color': (170, 184, 90)},
            43: {'name': 'coho', 'train_id': 42, 'color': (174, 230, 187)},
            44: {'name': 'eel', 'train_id': 43, 'color': (0, 178, 178)},
            45: {'name': 'goldfish', 'train_id': 44, 'color': (173, 121, 0)},
            46: {'name': 'jellyfish', 'train_id': 45, 'color': (97, 194, 157)},
            47: {'name': 'lionfish', 'train_id': 46, 'color': (0, 128, 255)},
            48: {'name': 'puffer', 'train_id': 47, 'color': (87, 106, 110)},
            49: {'name': 'rock_beauty', 'train_id': 48, 'color': (142, 173, 0)},
            50: {'name': 'sturgeon', 'train_id': 49, 'color': (27, 71, 74)},
            51: {'name': 'tench', 'train_id': 50, 'color': (209, 88, 88)}
        }

        self.color_map = np.zeros((256, 3), dtype=np.uint8)
        for item in self.label_dictionary.values():
            self.color_map[item['train_id']] = item['color']

        def update_value(value):
            class_info = next((v for v in self.label_dictionary.values() if v['train_id'] == value), None)
            class_name = class_info['name'] if class_info else 'unknown'
            self.value_label.setText(f"Train ID: {value}, Class: {class_name}")

        def update_pred_value(value):
            class_info = next((v for v in self.label_dictionary.values() if v['train_id'] == value), None)
            class_name = class_info['name'] if class_info else 'unknown'
            self.pred_value_label.setText(f"Prediction - Train ID: {value}, Class: {class_name}")

        self.colored_seg_label.value_callback = update_value
        self.raw_seg_label.value_callback = update_value
        self.pred_label.value_callback = update_pred_value
        self.colored_pred_label.value_callback = update_pred_value

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.load_folder('train')

    def update_class_table(self):
        unique_classes = np.unique(self.current_seg)
        self.class_table.setRowCount(len(unique_classes))

        for i, class_id in enumerate(unique_classes):
            class_info = next((v for v in self.label_dictionary.values() if v['train_id'] == class_id), None)
            if class_info:
                self.class_table.setItem(i, 0, QTableWidgetItem(class_info['name']))
                self.class_table.setItem(i, 1, QTableWidgetItem(str(class_id)))

        self.class_table.resizeColumnsToContents()

    def update_pred_class_table(self):
        if hasattr(self, 'current_pred'):
            unique_classes = np.unique(self.current_pred)
            self.pred_class_table.setRowCount(len(unique_classes))

            for i, class_id in enumerate(unique_classes):
                class_info = next((v for v in self.label_dictionary.values() if v['train_id'] == class_id), None)
                if class_info:
                    self.pred_class_table.setItem(i, 0, QTableWidgetItem(class_info['name']))
                    self.pred_class_table.setItem(i, 1, QTableWidgetItem(str(class_id)))

            self.pred_class_table.resizeColumnsToContents()

    def load_folder(self, folder):
        folder_path = os.path.join(self.base_path, folder)
        self.image_paths = natsorted([
            os.path.join(folder_path, "images", f)
            for f in os.listdir(os.path.join(folder_path, "images"))
            if f.endswith('.png')
        ])
        self.label_paths = natsorted([
            os.path.join(folder_path, "labels", f)
            for f in os.listdir(os.path.join(folder_path, "labels"))
            if f.endswith('.png')
        ])

        if folder == 'validation':
            self.pred_paths = natsorted([
                os.path.join(folder_path, "raw_predictions", f)
                for f in os.listdir(os.path.join(folder_path, "raw_predictions"))
                if f.endswith('.png')
            ])
            self.pred_widget.show()
            self.pred_class_table.show()
        else:
            self.pred_widget.hide()
            self.pred_class_table.hide()

        self.current_index = 0
        self.load_current_images()

    def load_current_images(self):
        if not self.image_paths:
            return

        def center_crop(img):
            if len(img.shape) == 3:
                h, w, _ = img.shape
            else:
                h, w = img.shape

            size = min(h, w)
            start_h = (h - size) // 2
            start_w = (w - size) // 2

            if len(img.shape) == 3:
                return img[start_h:start_h + size, start_w:start_w + size, :]
            return img[start_h:start_h + size, start_w:start_w + size]

        # Create cropped directories if they don't exist
        if self.folder_combo.currentText() == 'validation':
            validation_dir = os.path.dirname(os.path.dirname(self.image_paths[self.current_index]))
            cropped_images_dir = os.path.join(validation_dir, 'cropped_images')
            cropped_labels_dir = os.path.join(validation_dir, 'cropped_labels')
            cropped_colored_labels_dir = os.path.join(validation_dir, 'cropped_colored_labels')

            os.makedirs(cropped_images_dir, exist_ok=True)
            os.makedirs(cropped_labels_dir, exist_ok=True)
            os.makedirs(cropped_colored_labels_dir, exist_ok=True)

        current_image = os.path.basename(self.image_paths[self.current_index])
        self.image_name_label.setText(f"Image: {current_image} ({self.current_index + 1}/{len(self.image_paths)})")

        # Load and crop RGB image
        rgb_img = np.array(Image.open(self.image_paths[self.current_index]))
        rgb_img_cropped = center_crop(rgb_img)

        # Save cropped image if in validation mode and it doesn't exist
        if self.folder_combo.currentText() == 'validation':
            cropped_image_path = os.path.join(cropped_images_dir, current_image)
            if not os.path.exists(cropped_image_path):
                Image.fromarray(rgb_img_cropped).save(cropped_image_path)

        height, width = rgb_img_cropped.shape[:2]
        bytes_per_line = 3 * width
        rgb_qimg = QImage(rgb_img_cropped.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.rgb_label.setPixmap(QPixmap.fromImage(rgb_qimg).scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))

        # Load and crop segmentation data
        current_label = os.path.basename(self.label_paths[self.current_index])
        self.current_seg = np.array(Image.open(self.label_paths[self.current_index]))
        self.current_seg = center_crop(self.current_seg)

        # Save cropped label and colored label if in validation mode and they don't exist
        if self.folder_combo.currentText() == 'validation':
            # Save raw label
            cropped_label_path = os.path.join(cropped_labels_dir, current_label)
            if not os.path.exists(cropped_label_path):
                Image.fromarray(self.current_seg).save(cropped_label_path)

            # Save colored label
            colored_label_filename = os.path.splitext(current_label)[0] + '_colored.png'
            cropped_colored_label_path = os.path.join(cropped_colored_labels_dir, colored_label_filename)
            if not os.path.exists(cropped_colored_label_path):
                colored_seg = self.color_map[self.current_seg]
                Image.fromarray(colored_seg.astype(np.uint8)).save(cropped_colored_label_path)

        height, width = self.current_seg.shape

        # Reset image data for hover labels
        self.colored_seg_label.image_data = self.current_seg.copy()
        self.raw_seg_label.image_data = self.current_seg.copy()
        self.colored_seg_label.orig_width = None
        self.raw_seg_label.orig_width = None
        self.colored_seg_label.orig_height = None
        self.raw_seg_label.orig_height = None

        # Create and display colored segmentation
        colored_seg = self.color_map[self.current_seg]
        bytes_per_line = 3 * width
        colored_qimg = QImage(colored_seg.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.colored_seg_label.setPixmap(
            QPixmap.fromImage(colored_qimg).scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))

        # Display raw segmentation
        raw_qimg = QImage(self.current_seg.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
        self.raw_seg_label.setPixmap(QPixmap.fromImage(raw_qimg).scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))

        # Load predictions if in validation mode
        if self.folder_combo.currentText() == 'validation':
            self.current_pred = np.array(Image.open(self.pred_paths[self.current_index]))
            self.current_pred = center_crop(self.current_pred)

            # Reset prediction image data
            self.pred_label.image_data = self.current_pred.copy()
            self.colored_pred_label.image_data = self.current_pred.copy()
            self.pred_label.orig_width = None
            self.colored_pred_label.orig_width = None
            self.pred_label.orig_height = None
            self.colored_pred_label.orig_height = None

            height, width = self.current_pred.shape

            # Create and display colored prediction
            colored_pred = self.color_map[self.current_pred]
            pred_qimg = QImage(self.current_pred.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
            bytes_per_line = 3 * width
            colored_pred_qimg = QImage(colored_pred.tobytes(), width, height, bytes_per_line,
                                       QImage.Format.Format_RGB888)

            self.pred_label.setPixmap(QPixmap.fromImage(pred_qimg).scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))
            self.colored_pred_label.setPixmap(
                QPixmap.fromImage(colored_pred_qimg).scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))

            self.update_pred_class_table()

        self.update_class_table()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key.Key_Right:
            self.next_image()

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_current_images()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_images()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())