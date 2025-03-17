import sys
import os
import threading
import time
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, 
                           QLineEdit, QSpinBox, QTextEdit, QProgressBar,
                           QListWidget, QCheckBox, QGroupBox, QMessageBox,
                           QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QFont, QTextCursor
import string_detect  # Import your existing module

class OutputRedirector(QThread):
    output_received = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        
    def run(self):
        import io
        import sys
        
        # Create a custom output stream
        class Stream(io.StringIO):
            def __init__(self, redirect_func):
                super().__init__()
                self.redirect_func = redirect_func
                
            def write(self, text):
                self.redirect_func(text)
                return len(text)
        
        # Save the original stdout
        self.original_stdout = sys.stdout
        
        # Redirect stdout to our custom stream
        sys.stdout = Stream(self.handle_output)
        
    def handle_output(self, text):
        self.output_received.emit(text)
        
    def stop(self):
        self.running = False
        # Restore original stdout
        import sys
        sys.stdout = self.original_stdout

class ProcessingThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int)
    
    def __init__(self, mode, input_path, output_folder, target_path, cores, dpi):
        super().__init__()
        self.mode = mode  # 'single' or 'batch'
        self.input_path = input_path
        self.output_folder = output_folder
        self.target_path = target_path
        self.cores = cores
        self.dpi = dpi
    
    def run(self):
        try:
            if self.mode == 'single':
                # Process single PDF
                base_name = os.path.splitext(os.path.basename(self.input_path))[0]
                output_pdf = os.path.join(self.output_folder, f"{base_name}_highlighted.pdf")
                os.makedirs(self.output_folder, exist_ok=True)
                
                # Load target strings
                target_strings = string_detect.load_target_strings(self.target_path)
                
                # Process the file
                string_detect.highlight_strings_in_pdf(
                    self.input_path, output_pdf, target_strings, 
                    max_workers=self.cores, dpi=self.dpi
                )
                result_path = output_pdf
                
            else:  # batch mode
                # Process all PDFs in the folder
                string_detect.process_all_pdfs(
                    input_folder=self.input_path, 
                    output_folder=self.output_folder,
                    target_file=self.target_path, 
                    max_workers=self.cores, 
                    dpi=self.dpi
                )
                result_path = os.path.join(self.output_folder, "detection_results.csv")
            
            self.finished.emit(True, result_path)
        except Exception as e:
            self.finished.emit(False, str(e))


class CADStringDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.redirector = OutputRedirector()
        self.redirector.output_received.connect(self.update_log)
        self.redirector.start()
        
    def init_ui(self):
        self.setWindowTitle("CAD String Detector")
        self.setMinimumSize(800, 600)
        
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Create a horizontal splitter
        splitter = QSplitter(Qt.Vertical)
        
        # Top panel - Configuration
        config_widget = QWidget()
        config_layout = QVBoxLayout()
        config_widget.setLayout(config_layout)
        
        # File selection section
        file_group = QGroupBox("Input/Output Configuration")
        file_layout = QVBoxLayout()
        
        # Mode selection
        mode_layout = QHBoxLayout()
        self.single_mode_radio = QCheckBox("Process Single File")
        self.single_mode_radio.setChecked(True)
        self.batch_mode_radio = QCheckBox("Process Folder")
        self.single_mode_radio.toggled.connect(self.toggle_mode)
        self.batch_mode_radio.toggled.connect(self.toggle_mode)
        mode_layout.addWidget(self.single_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)
        file_layout.addLayout(mode_layout)
        
        # Input file/folder selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input:"))
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select PDF file or folder...")
        input_layout.addWidget(self.input_path_edit)
        self.browse_input_btn = QPushButton("Browse...")
        self.browse_input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(self.browse_input_btn)
        file_layout.addLayout(input_layout)
        
        # Output folder selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output:"))
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output folder...")
        output_layout.addWidget(self.output_path_edit)
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.browse_output_btn)
        file_layout.addLayout(output_layout)
        
        # Target strings file selection
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Strings:"))
        self.target_path_edit = QLineEdit()
        self.target_path_edit.setPlaceholderText("Select target.txt file...")
        target_layout.addWidget(self.target_path_edit)
        self.browse_target_btn = QPushButton("Browse...")
        self.browse_target_btn.clicked.connect(self.browse_target)
        target_layout.addWidget(self.browse_target_btn)
        file_layout.addLayout(target_layout)
        
        file_group.setLayout(file_layout)
        config_layout.addWidget(file_group)
        
        # Processing parameters section
        param_group = QGroupBox("Processing Parameters")
        param_layout = QVBoxLayout()
        
        # CPU cores selection
        cores_layout = QHBoxLayout()
        cores_layout.addWidget(QLabel("CPU Cores:"))
        self.cores_spin = QSpinBox()
        self.cores_spin.setRange(0, 32)
        self.cores_spin.setValue(16)
        cores_layout.addWidget(self.cores_spin)
        cores_layout.addWidget(QLabel("(0 = disable parallel processing)"))
        cores_layout.addStretch()
        param_layout.addLayout(cores_layout)
        
        # DPI selection
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("DPI Resolution:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(100, 600)
        self.dpi_spin.setValue(400)
        dpi_layout.addWidget(self.dpi_spin)
        dpi_layout.addWidget(QLabel("(higher = more accurate but slower)"))
        dpi_layout.addStretch()
        param_layout.addLayout(dpi_layout)
        
        param_group.setLayout(param_layout)
        config_layout.addWidget(param_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setMinimumHeight(40)
        font = QFont()
        font.setBold(True)
        self.start_btn.setFont(font)
        btn_layout.addWidget(self.start_btn)
        
        config_layout.addLayout(btn_layout)
        
        # Bottom panel - Log output
        log_widget = QWidget()
        log_layout = QVBoxLayout()
        log_widget.setLayout(log_layout)
        
        # Progress section
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        log_layout.addLayout(progress_layout)
        
        # Log output
        log_layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Add widgets to splitter
        splitter.addWidget(config_widget)
        splitter.addWidget(log_widget)
        
        # Set splitter sizes
        splitter.setSizes([400, 200])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def toggle_mode(self):
        # Ensure only one mode is selected
        if self.sender() == self.single_mode_radio and self.single_mode_radio.isChecked():
            self.batch_mode_radio.setChecked(False)
        elif self.sender() == self.batch_mode_radio and self.batch_mode_radio.isChecked():
            self.single_mode_radio.setChecked(False)
        
        # Update UI based on selected mode
        if self.single_mode_radio.isChecked():
            self.browse_input_btn.setText("Browse File...")
        else:
            self.browse_input_btn.setText("Browse Folder...")
    
    def browse_input(self):
        if self.single_mode_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select PDF File", "", "PDF Files (*.pdf)")
            if file_path:
                self.input_path_edit.setText(file_path)
        else:
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Input Folder")
            if folder_path:
                self.input_path_edit.setText(folder_path)
    
    def browse_output(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder")
        if folder_path:
            self.output_path_edit.setText(folder_path)
    
    def browse_target(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Target File", "", "Text Files (*.txt)")
        if file_path:
            self.target_path_edit.setText(file_path)
    
    def start_processing(self):
        # Validate inputs
        if not self.input_path_edit.text():
            QMessageBox.warning(self, "Input Required", "Please select an input file or folder.")
            return
        
        if not self.output_path_edit.text():
            QMessageBox.warning(self, "Output Required", "Please select an output folder.")
            return
        
        if not self.target_path_edit.text():
            QMessageBox.warning(self, "Target Required", "Please select a target strings file.")
            return
        
        if not os.path.exists(self.target_path_edit.text()):
            QMessageBox.warning(self, "File Not Found", "Target strings file not found.")
            return
        
        if self.single_mode_radio.isChecked():
            if not os.path.exists(self.input_path_edit.text()):
                QMessageBox.warning(self, "File Not Found", "Input PDF file not found.")
                return
            mode = 'single'
        else:
            if not os.path.exists(self.input_path_edit.text()):
                QMessageBox.warning(self, "Folder Not Found", "Input folder not found.")
                return
            mode = 'batch'
        
        # Disable UI controls during processing
        self.set_ui_enabled(False)
        self.log_text.clear()
        self.log_text.append("Starting processing...\n")
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start processing in a separate thread
        self.processing_thread = ProcessingThread(
            mode=mode,
            input_path=self.input_path_edit.text(),
            output_folder=self.output_path_edit.text(),
            target_path=self.target_path_edit.text(),
            cores=self.cores_spin.value(),
            dpi=self.dpi_spin.value()
        )
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()
    
    @pyqtSlot(bool, str)
    def processing_finished(self, success, result):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        if success:
            self.log_text.append(f"\nProcessing completed successfully!")
            self.log_text.append(f"Results saved to: {result}")
            QMessageBox.information(
                self, "Processing Complete", 
                f"Processing completed successfully!\nResults saved to: {result}"
            )
        else:
            self.log_text.append(f"\nError during processing: {result}")
            QMessageBox.critical(
                self, "Processing Error", 
                f"An error occurred during processing:\n{result}"
            )
        
        # Re-enable UI controls
        self.set_ui_enabled(True)
    
    def set_ui_enabled(self, enabled):
        # Enable/disable UI controls
        self.input_path_edit.setEnabled(enabled)
        self.output_path_edit.setEnabled(enabled)
        self.target_path_edit.setEnabled(enabled)
        self.browse_input_btn.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)
        self.browse_target_btn.setEnabled(enabled)
        self.cores_spin.setEnabled(enabled)
        self.dpi_spin.setEnabled(enabled)
        self.start_btn.setEnabled(enabled)
        self.single_mode_radio.setEnabled(enabled)
        self.batch_mode_radio.setEnabled(enabled)
    
    @pyqtSlot(str)
    def update_log(self, text):
        self.log_text.moveCursor(QTextCursor.End)
        self.log_text.insertPlainText(text)
        self.log_text.moveCursor(QTextCursor.End)
    
    def closeEvent(self, event):
        if hasattr(self, 'redirector'):
            self.redirector.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = CADStringDetectorApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()