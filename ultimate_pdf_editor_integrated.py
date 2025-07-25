"""
🎯 ULTIMATE PDF EDITOR - FINAL INTEGRATED VERSION
===============================================
Complete PDF editing solution with brush erasing, text editing, and OCR functionality
"""

import os
import sys
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

# Web framework
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# PDF and image processing
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# Import our custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from brush_eraser import BrushEraser
from text_editor import TextEditor
from ocr_editor import OCREditor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ultimate PDF Editor",
    description="Professional PDF editing with brush erasing, text editing, and OCR",
    version="1.0.0"
)

# Global instances
brush_eraser = BrushEraser()
text_editor = TextEditor()
ocr_editor = OCREditor()

# Session storage
sessions: Dict[str, Dict[str, Any]] = {}

# Create directories
UPLOAD_DIR = Path("temp_files")
STATIC_DIR = Path("static")
TEMPLATE_DIR = Path("templates")
DOWNLOAD_DIR = Path("downloads")

for directory in [UPLOAD_DIR, STATIC_DIR, TEMPLATE_DIR, DOWNLOAD_DIR]:
    directory.mkdir(exist_ok=True)

# Setup static files and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


class PDFSession:
    """Manages a PDF editing session with all operations"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.pdf_document = None
        self.page_images = {}
        self.original_images = {}
        self.current_page = 0
        self.total_pages = 0
        self.filename = ""
        
        # Operation history
        self.operations = []
        
        # Module instances
        self.brush_eraser = BrushEraser()
        self.text_editor = TextEditor()
        self.ocr_editor = OCREditor()
        
    def load_pdf(self, file_path: str, filename: str):
        """Load PDF and convert pages to images"""
        try:
            self.pdf_document = fitz.open(file_path)
            self.total_pages = len(self.pdf_document)
            self.filename = filename
            
            logger.info(f"📄 Loading PDF: {filename} with {self.total_pages} pages")
            
            # Convert all pages to images
            for page_num in range(self.total_pages):
                page = self.pdf_document[page_num]
                
                # Get page as image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to numpy array
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Store original and current images
                self.original_images[page_num] = img.copy()
                self.page_images[page_num] = img.copy()
                
                logger.info(f"📄 Processed page {page_num + 1}, image shape: {img.shape}")
            
            # Run OCR on all pages
            self.run_ocr_all_pages()
            
            logger.info(f"📄 PDF loaded successfully with {self.total_pages} pages")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading PDF: {str(e)}")
            return False
    
    def run_ocr_all_pages(self):
        """Run OCR on all pages"""
        try:
            total_elements = 0
            for page_num in range(self.total_pages):
                if page_num in self.page_images:
                    image = self.page_images[page_num]
                    elements = self.ocr_editor.detect_text(image, page_num)
                    total_elements += len(elements)
                    logger.info(f"🔍 OCR found {len(elements)} text elements on page {page_num}")
            
            logger.info(f"🔍 OCR completed. Total text elements: {total_elements}")
            
        except Exception as e:
            logger.error(f"❌ OCR error: {str(e)}")
    
    def apply_brush_erase(self, page_num: int, x: float, y: float, brush_size: int = 20):
        """Apply brush erase operation"""
        if page_num not in self.page_images:
            return False
            
        try:
            image = self.page_images[page_num]
            erased_image = self.brush_eraser.apply_brush_erase(image, x, y, brush_size)
            self.page_images[page_num] = erased_image
            
            # Record operation
            operation = {
                "type": "brush_erase",
                "page": page_num,
                "x": x,
                "y": y,
                "brush_size": brush_size,
                "timestamp": uuid.uuid4().hex
            }
            self.operations.append(operation)
            
            logger.info(f"🖌️ Brush erase applied to page {page_num} at ({x:.3f}, {y:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Brush erase error: {str(e)}")
            return False
    
    def add_text(self, page_num: int, x: float, y: float, text: str, **styling):
        """Add text to page with styling"""
        try:
            # Extract styling parameters
            font_family = styling.get('font_family', 'Arial')
            font_size = styling.get('font_size', 16)
            color = styling.get('color', '#000000')
            font_weight = styling.get('font_weight', 'normal')
            font_style = styling.get('font_style', 'normal')
            text_decoration = styling.get('text_decoration', 'none')
            
            element_id = self.text_editor.add_text_element(
                text=text,
                x=x,
                y=y,
                page_num=page_num,
                font_family=font_family,
                font_size=font_size,
                color=color,
                font_weight=font_weight,
                font_style=font_style,
                text_decoration=text_decoration
            )
            
            # Render text on image
            if page_num in self.page_images:
                image = self.page_images[page_num]
                rendered_image = self.text_editor.render_text_on_image(image, page_num)
                self.page_images[page_num] = rendered_image
            
            # Record operation
            operation = {
                "type": "add_text",
                "page": page_num,
                "x": x,
                "y": y,
                "text": text,
                "element_id": element_id,
                "styling": styling,
                "timestamp": uuid.uuid4().hex
            }
            self.operations.append(operation)
            
            logger.info(f"📝 Text added: '{text}' at ({x:.3f}, {y:.3f}) on page {page_num}")
            return element_id
            
        except Exception as e:
            logger.error(f"❌ Add text error: {str(e)}")
            return None
    
    def update_ocr_text(self, element_id: str, new_text: str):
        """Update OCR text element with clean rendering"""
        try:
            success = self.ocr_editor.update_element_text(element_id, new_text)
            
            if success:
                # Re-render all pages with clean OCR text replacement
                for page_num in range(self.total_pages):
                    if page_num in self.page_images:
                        # Start with original image
                        base_image = self.original_images[page_num].copy()
                        
                        # Apply all previous operations except OCR
                        base_image = self._apply_non_ocr_operations(base_image, page_num)
                        
                        # Apply clean OCR text replacement (no borders, no backgrounds)
                        final_image = self.ocr_editor.render_clean_text_replacement(base_image, page_num)
                        self.page_images[page_num] = final_image
                
                # Record operation
                operation = {
                    "type": "update_ocr_text",
                    "element_id": element_id,
                    "new_text": new_text,
                    "timestamp": uuid.uuid4().hex
                }
                self.operations.append(operation)
                
                logger.info(f"🔍 OCR text updated: element {element_id} -> '{new_text}'")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ OCR update error: {str(e)}")
            return False
    
    def _apply_non_ocr_operations(self, image: np.ndarray, page_num: int) -> np.ndarray:
        """Apply all non-OCR operations to reconstruct image state"""
        result_image = image.copy()
        
        for op in self.operations:
            # Check if operation has page info and matches current page
            if op.get("page", 0) != page_num:
                continue
                
            if op["type"] == "brush_erase":
                result_image = self.brush_eraser.apply_brush_erase(
                    result_image, op["x"], op["y"], op["brush_size"]
                )
            elif op["type"] == "add_text":
                # Text operations need special handling
                pass
        
        # Apply text editor rendering
        result_image = self.text_editor.render_text_on_image(result_image, page_num)
        
        return result_image
    
    def get_page_image_base64(self, page_num: int) -> str:
        """Get page image as base64 string"""
        if page_num not in self.page_images:
            return ""
            
        try:
            image = self.page_images[page_num]
            
            # Convert to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"❌ Image conversion error: {str(e)}")
            return ""
    
    def generate_final_pdf(self) -> str:
        """Generate final PDF with all modifications"""
        try:
            output_filename = f"ultimate_final_{self.session_id}.pdf"
            output_path = DOWNLOAD_DIR / output_filename
            
            # Create new PDF
            doc = fitz.open()
            
            for page_num in range(self.total_pages):
                try:
                    if page_num in self.page_images:
                        # Start with original image
                        base_image = self.original_images[page_num].copy()
                        
                        # Apply all non-OCR operations
                        base_image = self._apply_non_ocr_operations(base_image, page_num)
                        
                        # Apply clean OCR text replacement (no borders, no backgrounds)
                        final_image = self.ocr_editor.render_clean_text_replacement(base_image, page_num)
                        
                        # Convert to PIL for PDF creation
                        if len(final_image.shape) == 3:
                            pil_image = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
                        else:
                            pil_image = Image.fromarray(final_image)
                        
                        # Save as temporary PNG
                        temp_png = UPLOAD_DIR / f"temp_page_{page_num}_{self.session_id}.png"
                        pil_image.save(str(temp_png), format='PNG', quality=95, optimize=True)
                        
                        # Create PDF page from image
                        page_rect = fitz.Rect(0, 0, pil_image.width, pil_image.height)
                        pdf_page = doc.new_page(width=page_rect.width, height=page_rect.height)
                        pdf_page.insert_image(page_rect, filename=str(temp_png))
                        
                        # Clean up temp file
                        temp_png.unlink(missing_ok=True)
                    else:
                        # If no modified image, use original page
                        original_page = self.pdf_document[page_num]
                        page_rect = original_page.rect
                        pdf_page = doc.new_page(width=page_rect.width, height=page_rect.height)
                        pdf_page.show_pdf_page(page_rect, self.pdf_document, page_num)
                        
                except Exception as page_error:
                    logger.error(f"❌ Error processing page {page_num}: {str(page_error)}")
                    # Create a blank page as fallback
                    pdf_page = doc.new_page(width=595, height=842)  # A4 size
                    pdf_page.insert_text((50, 50), f"Error processing page {page_num + 1}")
            
            # Save final PDF
            doc.save(str(output_path))
            doc.close()
            
            logger.info(f"📄 Final PDF generated: {output_filename} with {len(self.operations)} operations")
            return output_filename
            
        except Exception as e:
            logger.error(f"❌ PDF generation error: {str(e)}")
            return ""


# API Routes

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse("ultimate_integrated.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Create new session
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create session and load PDF
        session = PDFSession(session_id)
        success = session.load_pdf(str(file_path), file.filename)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
        
        # Store session
        sessions[session_id] = session
        
        # Get first page image
        first_page_image = session.get_page_image_base64(0)
        
        # Get OCR state
        ocr_state = session.ocr_editor.get_editor_state()
        
        # Get text editor state
        text_state = session.text_editor.get_editor_state()
        
        return {
            "session_id": session_id,
            "total_pages": session.total_pages,
            "current_page": 0,
            "page_image": first_page_image,
            "ocr_state": ocr_state,
            "text_state": text_state,
            "message": f"📄 PDF loaded successfully! {session.total_pages} pages processed."
        }
        
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/brush-erase/{session_id}")
async def brush_erase(session_id: str, data: dict):
    """Apply brush erase"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    try:
        page_num = data.get("page", 0)
        x = data.get("x", 0.5)
        y = data.get("y", 0.5)
        brush_size = data.get("brush_size", 20)
        
        success = session.apply_brush_erase(page_num, x, y, brush_size)
        
        if success:
            page_image = session.get_page_image_base64(page_num)
            return {
                "success": True,
                "page_image": page_image,
                "message": f"🖌️ Brush erase applied at ({x:.2f}, {y:.2f})"
            }
        else:
            return {"success": False, "message": "Failed to apply brush erase"}
            
    except Exception as e:
        logger.error(f"❌ Brush erase error: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}


@app.post("/add-text/{session_id}")
async def add_text(session_id: str, data: dict):
    """Add text to PDF"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    try:
        page_num = data.get("page", 0)
        x = data.get("x", 0.5)
        y = data.get("y", 0.5)
        text = data.get("text", "New Text")
        styling = data.get("styling", {})
        
        element_id = session.add_text(page_num, x, y, text, **styling)
        
        if element_id:
            page_image = session.get_page_image_base64(page_num)
            text_state = session.text_editor.get_editor_state()
            
            return {
                "success": True,
                "element_id": element_id,
                "page_image": page_image,
                "text_state": text_state,
                "message": f"📝 Text added: '{text}'"
            }
        else:
            return {"success": False, "message": "Failed to add text"}
            
    except Exception as e:
        logger.error(f"❌ Add text error: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}


@app.post("/update-text/{session_id}")
async def update_text(session_id: str, data: dict):
    """Update text element"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    try:
        element_id = data.get("element_id")
        updates = data.get("updates", {})
        
        # Update in text editor
        success = session.text_editor.update_text_element(element_id, **updates)
        
        if success:
            # Re-render affected page
            element = session.text_editor.text_elements.get(element_id)
            if element:
                page_num = element.page_num
                
                # Reconstruct page image
                base_image = session._apply_non_ocr_operations(
                    session.original_images[page_num].copy(), page_num
                )
                session.page_images[page_num] = base_image
                
                page_image = session.get_page_image_base64(page_num)
                text_state = session.text_editor.get_editor_state()
                
                return {
                    "success": True,
                    "page_image": page_image,
                    "text_state": text_state,
                    "message": "📝 Text updated successfully"
                }
        
        return {"success": False, "message": "Failed to update text"}
        
    except Exception as e:
        logger.error(f"❌ Update text error: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}


@app.post("/update-ocr-text/{session_id}")
async def update_ocr_text(session_id: str, data: dict):
    """Update OCR text element"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    try:
        element_id = data.get("element_id")
        new_text = data.get("new_text", "")
        
        success = session.update_ocr_text(element_id, new_text)
        
        if success:
            # Get updated page image
            element = session.ocr_editor.text_elements.get(element_id)
            if element:
                page_num = element.page_num
                page_image = session.get_page_image_base64(page_num)
                ocr_state = session.ocr_editor.get_editor_state()
                
                return {
                    "success": True,
                    "page_image": page_image,
                    "ocr_state": ocr_state,
                    "message": f"🔍 OCR text updated: '{new_text}'"
                }
        
        return {"success": False, "message": "Failed to update OCR text"}
        
    except Exception as e:
        logger.error(f"❌ Update OCR text error: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}


@app.post("/change-page/{session_id}")
async def change_page(session_id: str, data: dict):
    """Change current page"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    try:
        page_num = data.get("page", 0)
        
        if 0 <= page_num < session.total_pages:
            session.current_page = page_num
            page_image = session.get_page_image_base64(page_num)
            
            # Get relevant states for this page
            ocr_elements = session.ocr_editor.get_elements_by_page(page_num)
            text_elements = session.text_editor.get_elements_by_page(page_num)
            
            return {
                "success": True,
                "current_page": page_num,
                "page_image": page_image,
                "ocr_elements": [elem.to_dict() for elem in ocr_elements],
                "text_elements": [elem.to_dict() for elem in text_elements],
                "message": f"📄 Switched to page {page_num + 1}"
            }
        else:
            return {"success": False, "message": "Invalid page number"}
            
    except Exception as e:
        logger.error(f"❌ Change page error: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}


@app.post("/generate-pdf/{session_id}")
async def generate_pdf(session_id: str):
    """Generate final PDF"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    try:
        filename = session.generate_final_pdf()
        
        if filename:
            return {
                "success": True,
                "filename": filename,
                "download_url": f"/download/{session_id}",
                "operations_count": len(session.operations),
                "message": f"📄 PDF generated successfully with {len(session.operations)} operations!"
            }
        else:
            return {"success": False, "message": "Failed to generate PDF"}
            
    except Exception as e:
        logger.error(f"❌ Generate PDF error: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}


@app.get("/download/{session_id}")
async def download_pdf(session_id: str):
    """Download generated PDF"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    filename = f"ultimate_final_{session_id}.pdf"
    file_path = DOWNLOAD_DIR / filename
    
    if file_path.exists():
        return FileResponse(
            path=str(file_path),
            filename=f"edited_{session.filename}",
            media_type="application/pdf"
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/session-info/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "filename": session.filename,
        "total_pages": session.total_pages,
        "current_page": session.current_page,
        "operations_count": len(session.operations),
        "ocr_state": session.ocr_editor.get_editor_state(),
        "text_state": session.text_editor.get_editor_state()
    }


# Cleanup route
@app.post("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up session files"""
    try:
        if session_id in sessions:
            session = sessions[session_id]
            
            # Close PDF document
            if session.pdf_document:
                session.pdf_document.close()
            
            # Remove from sessions
            del sessions[session_id]
            
            # Clean up temp files
            for file_path in UPLOAD_DIR.glob(f"{session_id}_*"):
                file_path.unlink(missing_ok=True)
            
            return {"success": True, "message": "Session cleaned up successfully"}
        else:
            return {"success": False, "message": "Session not found"}
            
    except Exception as e:
        logger.error(f"❌ Cleanup error: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}


if __name__ == "__main__":
    logger.info("🚀 Starting Ultimate PDF Editor")
    logger.info("📝 Features: Brush Erasing, Text Editing, OCR Recognition")
    logger.info("🌐 Access at: http://localhost:8000")
    
    uvicorn.run(
        "ultimate_pdf_editor_integrated:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    ) 