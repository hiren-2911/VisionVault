# main.py
import argparse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import base64
import asyncio
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import traceback
from pathlib import Path
import uuid

from aadhar_processor import process_aadhaar_document
from data_models import AadhaarData
from config_manager import ConfigManager

# Initialize FastAPI app
app = FastAPI(
    title="Aadhaar OCR Processing API",
    description="API for processing Aadhaar cards using OCR and object detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Aadhaar OCR API")
    parser.add_argument(
        '--config-path',
        type=str,
        default='E:/OCR/aadhar_ocr/config/config.json',
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Path to logs directory'
    )
    parser.add_argument(
        '--logger-config-path',
        type=str,
        default='E:/OCR/aadhar_ocr/config/logger.json',
        help='Path to logger configuration JSON file'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run the server on'
    )
    return parser.parse_args()
# Global configuration
args = parse_arguments()
CONFIG_PATH = args.config_path
LOGGER_CONFIG_PATH = args.logger_config_path
LOG_DIR = Path(args.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)
config_manager: ConfigManager = None
base_dir = Path(__file__).parent.resolve()

# Pydantic models for API
class ProcessingRequest(BaseModel):
    filename: str = Field(..., description="Name of the file to process")
    file_content: str = Field(..., description="Base64 encoded file content")
    ref_id: Optional[str] = Field(None, description="Reference ID for tracking")

class ProcessingResponse(BaseModel):
    success: bool = Field(..., description="Processing success status")
    ref_id: str = Field(..., description="Reference ID")
    data: Optional[Dict[str, Any]] = Field(None, description="Processed data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Total processing time")

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global config_manager
    try:
        #logger.info("Starting Aadhaar OCR API...")
        # Initialize configuration to test setup
        config_manager = ConfigManager(
            config_path=CONFIG_PATH,
            logger_config_path=LOGGER_CONFIG_PATH,
            application_name="aadhar_ocr",
            console_log=True,
            log_dir=LOG_DIR
        )
        logger = logging.getLogger(__name__)
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Aadhaar OCR Processing API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/process", response_model=ProcessingResponse)
async def process_aadhaar(request: ProcessingRequest):
    """Process Aadhaar document"""
    ref_id = request.ref_id or str(uuid.uuid4())
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"[{ref_id}] Processing request for file: {request.filename}")
        
        # Validate input
        if not request.filename or not request.file_content:
            raise HTTPException(
                status_code=400,
                detail="Both filename and file_content are required"
            )
        
        # Validate file extension
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        file_ext = Path(request.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Allowed: {allowed_extensions}"
            )
        
        # Process document
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            process_aadhaar_document,
            ref_id,
            request.filename,
            request.file_content,
            config_manager
        )
        
        logger.info(f"[{ref_id}] Processing completed successfully")
        
        return ProcessingResponse(
            success=True,
            ref_id=ref_id,
            data=result,
            processing_time=result.get('processing_time', 0.0) if result else 0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        logger.error(f"[{ref_id}] {error_msg}")
        logger.error(f"[{ref_id}] Traceback: {traceback.format_exc()}")
        
        return ProcessingResponse(
            success=False,
            ref_id=ref_id,
            error_message=error_msg
        )

@app.post("/process-file", response_model=ProcessingResponse)
async def process_aadhaar_file(
    file: UploadFile = File(...),
    ref_id: Optional[str] = Form(None)
):
    """Process Aadhaar document via file upload"""
    ref_id = ref_id or str(uuid.uuid4())
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"[{ref_id}] Processing uploaded file: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file extension
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}"
            )
        
        # Read and encode file
        file_content = await file.read()
        b64_content = base64.b64encode(file_content).decode('utf-8')
        
        # Process document
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            process_aadhaar_document,
            ref_id,
            file.filename,
            b64_content,
            config_manager
        )
        
        logger.info(f"[{ref_id}] File processing completed successfully")
        
        return ProcessingResponse(
            success=True,
            ref_id=ref_id,
            data=result,
            processing_time=result.get('processing_time', 0.0) if result else 0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"File processing failed: {str(e)}"
        logger.error(f"[{ref_id}] {error_msg}")
        
        return ProcessingResponse(
            success=False,
            ref_id=ref_id,
            error_message=error_msg
        )

@app.get("/status/{ref_id}")
async def get_processing_status(ref_id: str):
    """Get processing status (placeholder for async processing)"""
    # This would be useful if you implement async processing with a task queue
    return {
        "ref_id": ref_id,
        "status": "completed",
        "message": "Synchronous processing - check response from /process endpoint"
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
