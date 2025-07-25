# 🎯 AI PDF Editor - Professional Edition

A powerful, AI-driven PDF editing application with advanced features including brush erasing, text editing, and OCR text recognition with real-time editing capabilities.

## ✨ Features

### 🖌️ **Brush Erasing Tool**
- **Visual Preview**: Red circle outline that accurately matches the erasing area
- **Adjustable Size**: Dynamic brush size control (5-50 pixels)
- **Real-time Feedback**: Visual glow effect during erasing
- **Precise Control**: Pixel-perfect erasing with background color matching

### 📝 **Advanced Text Editor**
- **Draggable Text**: Click and drag text overlays to position anywhere
- **Professional Styling**: Font family, size, color, bold, italic, underline
- **Real-time Preview**: See changes instantly before embedding
- **Smart Positioning**: Click to embed, right-click to delete

### 🔍 **OCR Text Recognition & Editing**
- **AI-Powered OCR**: Uses EasyOCR for accurate text detection
- **Real-time Editing**: Click detected text to edit in popup
- **Clean Replacement**: Seamless text replacement without borders or backgrounds
- **Smart Merging**: Automatically merges nearby text elements
- **Confidence Filtering**: Only processes high-confidence text detections

### 🎨 **Professional UI**
- **Modern Interface**: Clean, intuitive design with sidebar controls
- **Responsive Layout**: Works on different screen sizes
- **Visual Feedback**: Status messages and progress indicators
- **Multi-page Support**: Navigate between PDF pages seamlessly

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows 10/11 (tested on Windows 10.0.22631)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-PDF-Editor.git
   cd AI-PDF-Editor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python ultimate_pdf_editor_integrated.py
   ```

   Or use the provided batch file:
   ```bash
   run_ultimate_pdf_editor.bat
   ```

5. **Access the application**
   Open your browser and go to: `http://localhost:8000`

## 📁 Project Structure

```
AI_PDF_Editor/
├── ultimate_pdf_editor_integrated.py  # Main application
├── brush_eraser.py                    # Brush erasing module
├── text_editor.py                     # Text editing module
├── ocr_editor.py                      # OCR recognition module
├── templates/
│   └── ultimate_integrated.html       # Web interface
├── requirements.txt                   # Python dependencies
├── run_ultimate_pdf_editor.bat        # Windows batch file
└── README.md                          # This file
```

## 🛠️ Usage Guide

### 1. **Upload PDF**
- Click "Choose File" and select your PDF
- The application will automatically process all pages
- OCR will detect all text elements with confidence scores

### 2. **Brush Erasing**
- Select the "Brush" tool from the sidebar
- Adjust brush size using the slider
- Move mouse over PDF to see red circle preview
- Click to erase content with visual feedback

### 3. **Adding Text**
- Select "Add Text" tool
- Enter text in the input field
- Customize styling (font, size, color, etc.)
- Click on PDF to place draggable text overlay
- Drag to position, then click to embed permanently

### 4. **OCR Text Editing**
- Select "OCR" tool
- Click on any detected text element
- Edit text in the popup dialog
- Changes are applied with clean replacement (no borders)

### 5. **Generate Final PDF**
- Click "Generate PDF" to create the final document
- Download the edited PDF with all modifications

## 🔧 Technical Details

### **Architecture**
- **Backend**: FastAPI with Uvicorn server
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Image Processing**: OpenCV, PIL/Pillow
- **PDF Processing**: PyMuPDF (fitz)
- **OCR Engine**: EasyOCR with confidence scoring
- **Text Rendering**: Custom font rendering with background sampling

### **Key Algorithms**
- **Brush Erasing**: OpenCV circle masking with Gaussian blur
- **Text Positioning**: Coordinate normalization and pixel mapping
- **OCR Merging**: Proximity-based text element merging
- **Background Sampling**: Intelligent color matching for seamless text replacement

### **Performance Features**
- **Lazy Loading**: Pages processed on-demand
- **Memory Management**: Efficient image handling and cleanup
- **Session Management**: UUID-based session tracking
- **Error Recovery**: Graceful error handling with fallbacks

## 📦 Dependencies

### Core Dependencies
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `opencv-python==4.8.1.78` - Image processing
- `Pillow==10.0.1` - Image manipulation
- `PyMuPDF==1.23.8` - PDF processing
- `easyocr==1.7.0` - OCR engine
- `numpy==1.24.3` - Numerical computing

### Development Dependencies
- `python-multipart==0.0.6` - File upload handling
- `jinja2==3.1.2` - Template engine

## 🎯 Advanced Features

### **Smart Text Merging**
The OCR system automatically merges nearby text elements that likely belong together, improving readability and editing experience.

### **Background Color Sampling**
When replacing OCR text, the system samples the surrounding background color to ensure seamless integration without visible borders.

### **Font Size Estimation**
Automatically estimates appropriate font sizes based on the original text's bounding box dimensions.

### **Multi-page Support**
Full support for multi-page PDFs with independent editing per page and unified final output.

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'cv2'**
   - Ensure virtual environment is activated
   - Run: `pip install -r requirements.txt`

2. **PDF generation fails**
   - Check if PDF file is corrupted
   - Ensure sufficient disk space
   - Verify file permissions

3. **OCR not detecting text**
   - Ensure PDF has readable text (not just images)
   - Check image quality for scanned documents
   - Try adjusting confidence threshold

### Performance Tips
- Use SSD storage for faster file operations
- Close other applications to free up memory
- For large PDFs, process pages individually

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EasyOCR** for powerful text recognition
- **PyMuPDF** for robust PDF processing
- **OpenCV** for advanced image manipulation
- **FastAPI** for modern web framework

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Made with ❤️ for professional PDF editing** 
