#!/usr/bin/env python3

import streamlit as st
import os
import csv
import requests
import base64
import json
import time
from pathlib import Path
import logging
from PIL import Image
import io
import pandas as pd
import zipfile
from datetime import datetime
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import replicate
import tempfile
import shutil
import uuid
from typing import List, Dict, Tuple
import re
import mimetypes
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

# Disable SSL warnings if needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging with detailed format and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'seedream_app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# API Keys - Use Streamlit secrets for hosting
REPLICATE_API_KEY = st.secrets["REPLICATE_API_KEY"]
ARK_API_KEY = st.secrets["ARK_API_KEY"]
COMFYUI_SERVER_URL = st.secrets.get("COMFYUI_SERVER_URL", "http://34.142.205.152/comfy")

# Seedream API Configuration
SEEDREAM_API_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"

# Streamlit page configuration
st.set_page_config(
    page_title="Seedream Face Swap Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SSLAdapter(HTTPAdapter):
    """Custom SSL adapter to handle SSL connection issues"""
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

class ExpressionFilter:
    """Filter to remove expressions from prompts"""
    
    def __init__(self):
        # Common facial expressions to filter out
        self.expression_patterns = [
            # Basic expressions
            r'\b(smiling|smile|smiles)\b',
            r'\b(happy|happiness|joyful|cheerful)\b',
            r'\b(sad|sadness|melancholy|sorrowful)\b',
            r'\b(angry|anger|furious|mad)\b',
            r'\b(surprised|surprise|shocked|amazed)\b',
            r'\b(serious|stern|solemn|grave)\b',
            r'\b(laughing|laugh|giggles|chuckling)\b',
            r'\b(crying|tears|weeping)\b',
            r'\b(frowning|frown|scowling)\b',
            r'\b(grinning|grin)\b',
            r'\b(worried|concerned|anxious)\b',
            r'\b(excited|enthusiastic|thrilled)\b',
            r'\b(calm|peaceful|serene|relaxed)\b',
            r'\b(confused|puzzled|bewildered)\b',
            r'\b(disgusted|disgust|repulsed)\b',
            r'\b(fearful|afraid|scared|frightened)\b',
            
            # Expression descriptions
            r'\bwith a [^,.]* expression\b',
            r'\bexpression of [^,.]*\b',
            r'\bfacial expression [^,.]*\b',
            r'\blooking [^,.]*(happy|sad|angry|surprised|serious|confused|excited|worried|calm)\b',
            r'\bappears [^,.]*(happy|sad|angry|surprised|serious|confused|excited|worried|calm)\b',
            r'\bseems [^,.]*(happy|sad|angry|surprised|serious|confused|excited|worried|calm)\b',
            
            # Mouth/smile related
            r'\bbig smile\b',
            r'\bwide smile\b',
            r'\bgentle smile\b',
            r'\bwarm smile\b',
            r'\bbright smile\b',
            r'\bopen mouth\b',
            r'\bmouth open\b',
            
            # Eyes related to expressions
            r'\beyes [^,.]*(sparkling|twinkling|bright|sad|angry|happy)\b',
            r'\b(sparkling|twinkling|bright|sad|angry|happy) eyes\b',
            
            # Emotional states
            r'\bemotional\b',
            r'\bemotion\b',
            r'\bmood\b',
            r'\bfeeling [^,.]*\b',
        ]
        
        # Compile patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.expression_patterns]
    
    def filter_expressions(self, prompt):
        """Remove expressions from the prompt"""
        filtered_prompt = prompt
        
        # Apply each pattern to remove expressions
        for pattern in self.compiled_patterns:
            filtered_prompt = pattern.sub('', filtered_prompt)
        
        # Clean up extra spaces and punctuation
        filtered_prompt = re.sub(r'\s+', ' ', filtered_prompt)  # Multiple spaces to single
        filtered_prompt = re.sub(r'\s*,\s*,\s*', ', ', filtered_prompt)  # Multiple commas
        filtered_prompt = re.sub(r'\s*\.\s*\.\s*', '. ', filtered_prompt)  # Multiple periods
        filtered_prompt = re.sub(r'^[,.\s]+|[,.\s]+$', '', filtered_prompt)  # Leading/trailing punctuation
        
        return filtered_prompt.strip()

class ComfyUIFaceSwapProcessor:
    """Face swap processor using ComfyUI"""
    def __init__(self, server_url="http://34.142.205.152/comfy", max_workers=10):
        self.server_url = server_url
        self.max_workers = max_workers
        self.session = requests.Session()
        
        logger.info(f"Initializing ComfyUI Face Swap Processor with server: {server_url}")
        
        # Progress tracking
        self.total_uploaded = 0
        self.total_processed = 0
        self.total_downloaded = 0
        self.progress_lock = threading.Lock()
        
        # Workflow template for face swap
        self.workflow_template = {
            "4": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["1", 0]
                },
                "class_type": "SaveImage",
                "_meta": {"title": "SaveImage"}
            },
            "2": {
                "inputs": {
                    "input_faces_order": "large-small",
                    "input_faces_index": "0",
                    "detect_gender_input": "no",
                    "source_faces_order": "large-small",
                    "source_faces_index": "0",
                    "detect_gender_source": "no",
                    "console_log_level": 1
                },
                "class_type": "ReActorOptions",
                "_meta": {"title": "ReActorOptions"}
            },
            "6": {
                "inputs": {
                    "image": "input_image.jpg"
                },
                "class_type": "LoadImage",
                "_meta": {"title": "LoadImage"}
            },
            "5": {
                "inputs": {
                    "image": "source_face.jpg"
                },
                "class_type": "LoadImage",
                "_meta": {"title": "LoadImage"}
            },
            "1": {
                "inputs": {
                    "enabled": True,
                    "swap_model": "inswapper_128_fp16.onnx",
                    "facedetection": "retinaface_resnet50",
                    "face_restore_model": "codeformer-v0.1.0.pth",
                    "face_restore_visibility": 0.6,
                    "codeformer_weight": 1,
                    "input_image": ["6", 0],
                    "source_image": ["5", 0],
                    "options": ["2", 0]
                },
                "class_type": "ReActorFaceSwapOpt",
                "_meta": {"title": "ReActorFaceSwapOpt"}
            }
        }
    
    def create_session(self):
        return requests.Session()
    
    def upload_image(self, image_path, filename):
        """Upload image to ComfyUI server"""
        logger.info(f"Uploading image: {filename} from {image_path}")
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f, 'image/jpeg')}
                response = self.session.post(f"{self.server_url}/upload/image", files=files)
                
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded: {filename}")
                    return True
                else:
                    logger.error(f"Failed to upload {filename}: HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error uploading image {filename}: {e}")
            st.error(f"Error uploading image: {e}")
            return False
    
    def submit_workflow(self, input_filename, source_filename, output_prefix):
        """Submit face swap workflow"""
        logger.info(f"Submitting workflow: input={input_filename}, source={source_filename}")
        try:
            import copy
            workflow = copy.deepcopy(self.workflow_template)
            
            # Update filenames
            workflow["6"]["inputs"]["image"] = input_filename
            workflow["5"]["inputs"]["image"] = source_filename
            workflow["4"]["inputs"]["filename_prefix"] = output_prefix
            
            client_id = str(uuid.uuid4())
            payload = {
                "prompt": workflow,
                "client_id": client_id
            }
            
            response = self.session.post(f"{self.server_url}/prompt", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                logger.info(f"Workflow submitted successfully: prompt_id={prompt_id}, client_id={client_id}")
                return prompt_id, client_id
            else:
                logger.error(f"Failed to submit workflow: HTTP {response.status_code}, Response: {response.text}")
                return None, None
            
        except Exception as e:
            logger.error(f"Error submitting workflow: {e}")
            st.error(f"Error submitting workflow: {e}")
            return None, None
    
    def wait_for_completion(self, prompt_id, timeout=300):
        """Wait for workflow completion"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                response = self.session.get(f"{self.server_url}/history/{prompt_id}")
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        return True
                time.sleep(5)
            except Exception as e:
                st.error(f"Error checking status: {e}")
                break
        
        return False
    
    def download_result(self, prompt_id, output_path):
        """Download processed image"""
        try:
            response = self.session.get(f"{self.server_url}/history/{prompt_id}")
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    if "4" in outputs and "images" in outputs["4"]:
                        img_info = outputs["4"]["images"][0]
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        
                        params = {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": "output"
                        }
                        
                        response = self.session.get(f"{self.server_url}/view", params=params)
                        
                        if response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                f.write(response.content)
                            return True
            
            return False
        
        except Exception as e:
            st.error(f"Error downloading result: {e}")
            return False
    
    def process_face_swap(self, input_image_path, source_face_path, output_path):
        """Process a single face swap"""
        try:
            # Generate unique filenames
            input_filename = f"input_{uuid.uuid4()}.jpg"
            source_filename = f"source_{uuid.uuid4()}.jpg"
            
            # Upload images
            if not self.upload_image(input_image_path, input_filename):
                return False
            
            if not self.upload_image(source_face_path, source_filename):
                return False
            
            # Submit workflow
            prompt_id, client_id = self.submit_workflow(
                input_filename, 
                source_filename, 
                f"swapped_{uuid.uuid4()}"
            )
            
            if not prompt_id:
                return False
            
            # Wait for completion
            if not self.wait_for_completion(prompt_id):
                return False
            
            # Download result
            return self.download_result(prompt_id, output_path)
        
        except Exception as e:
            st.error(f"Error in face swap processing: {e}")
            return False

class StreamlitImageGenerator:
    def __init__(self):
        self.progress_bar = None
        self.status_text = None
        self.results_container = None
        self.face_swap_processor = None
        self.expression_filter = ExpressionFilter()
        
        # Thread-local storage for sessions
        self.local = threading.local()
        
    def initialize_face_swap(self, comfyui_server_url):
        """Initialize face swap processor"""
        self.face_swap_processor = ComfyUIFaceSwapProcessor(server_url=comfyui_server_url)
        
    def get_session(self):
        """Get or create a thread-local session for Seedream API"""
        if not hasattr(self.local, 'session'):
            # Create session for this thread
            session = requests.Session()
            
            # Set up retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST"]
            )
            
            # Mount adapters with SSL configuration
            ssl_adapter = SSLAdapter(max_retries=retry_strategy)
            session.mount("https://", ssl_adapter)
            session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
            
            # Set headers
            session.headers.update({
                "Authorization": f"Bearer {ARK_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "SeedrameImageGenerator/1.0",
                "Accept": "application/json",
                "Connection": "keep-alive"
            })
            
            # Configure session settings
            session.verify = False  # Disable SSL verification as fallback
            
            self.local.session = session
            logger.debug(f"Created new session for thread: {threading.current_thread().name}")
        
        return self.local.session
        
    def extract_images_from_zip(self, zip_file_bytes, temp_dir):
        """Extract images from uploaded ZIP file bytes to temporary directory"""
        logger.info(f"Extracting images from ZIP file to: {temp_dir}")
        extracted_files = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        
        try:
            # Create a BytesIO object from the uploaded file bytes
            zip_buffer = io.BytesIO(zip_file_bytes)
            
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                logger.info(f"ZIP contains {len(file_list)} total files")
                
                for file_path in file_list:
                    # Skip directories, hidden files, and system files
                    if (file_path.endswith('/') or 
                        file_path.startswith('.') or 
                        '/__MACOSX' in file_path or
                        file_path.startswith('__MACOSX')):
                        continue
                    
                    # Check if it's a supported image format
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext in supported_extensions:
                        try:
                            # Extract file content
                            file_content = zip_ref.read(file_path)
                            
                            # Get clean filename
                            filename = os.path.basename(file_path)
                            if not filename:  # Skip if no filename
                                continue
                                
                            # Save to temp directory
                            output_path = os.path.join(temp_dir, filename)
                            with open(output_path, 'wb') as f:
                                f.write(file_content)
                            
                            extracted_files.append(Path(output_path))
                            logger.info(f"Extracted image: {filename} ({len(file_content)} bytes)")
                            
                        except Exception as e:
                            logger.warning(f"Failed to extract {file_path}: {e}")
                            continue
                
                logger.info(f"Successfully extracted {len(extracted_files)} images from ZIP")
                return extracted_files
                
        except zipfile.BadZipFile:
            logger.error("Invalid ZIP file format")
            st.error("Invalid ZIP file. Please upload a valid ZIP file containing images.")
            return []
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {e}")
            st.error(f"Error extracting ZIP file: {e}")
            return []
    
    def generate_random_filename(self, length=32):
        """Generate random alphanumeric filename"""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))
    
    def encode_image_to_base64(self, image_path):
        """Convert image file to base64 string with data URI format"""
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                
                # Get MIME type
                mime_type, _ = mimetypes.guess_type(image_path)
                if not mime_type or not mime_type.startswith('image/'):
                    mime_type = 'image/jpeg'  # Default fallback
                
                return f"data:{mime_type};base64,{base64_string}"
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            return None
    
    def get_chatgpt_prompt(self, image_path, replicate_api_key):
        """Get Seedream-friendly prompt from GPT-5 via Replicate"""
        logger.info(f"Getting prompt for image: {image_path}")
        try:
            # Set Replicate API token
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
            logger.info("Replicate API token set successfully")
            
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                logger.error(f"Failed to encode image to base64: {image_path}")
                return None
            
            logger.info("Image encoded to base64 successfully")
            
            # Use Replicate's GPT-5 model
            logger.info("Calling Replicate GPT-5 model...")
            output = replicate.run(
                "openai/gpt-5",
                input={
                    "prompt": "i want to generate this image on seedream, what should be the prompt. you cannot describe the body of the person or the hair. the expression should also be mentioned - this is mandatory, always describe the facial expression (happy, sad, serious, smiling, etc.). the gender can be present too. Only 1 person should be in focus",
                    "messages": [],
                    "image_input": [base64_image],
                    "verbosity": "medium",
                    "reasoning_effort": "minimal"
                }
            )
            
            # Collect streaming output
            result = ""
            for event in output:
                result += str(event)
            
            # Filter out expressions from the final prompt
            filtered_result = self.expression_filter.filter_expressions(result.strip())
            
            logger.info(f"GPT-5 prompt generated and filtered successfully: {filtered_result[:100]}...")
            return filtered_result
            
        except Exception as e:
            logger.error(f"Error getting GPT-5 prompt for {image_path}: {str(e)}")
            st.error(f"Error getting GPT-5 prompt: {str(e)}")
            return None
    
    def generate_seedream_image(self, prompt, selfie_path):
        """Generate image using Seedream API directly (not via Replicate)"""
        session = self.get_session()
        max_retries = 3
        thread_name = threading.current_thread().name
        
        for attempt in range(max_retries):
            try:
                # Encode selfie image to base64
                base64_image = self.encode_image_to_base64(selfie_path)
                if not base64_image:
                    logger.error(f"Failed to encode selfie to base64: {selfie_path}")
                    return None
                
                # Prepare API request for Seedream
                payload = {
                    "model": "seedream-4-0-250828",
                    "prompt": prompt,
                    "image": base64_image,
                    "size": "2304x4096",
                    "sequential_image_generation": "disabled",
                    "stream": False,
                    "response_format": "url",
                    "watermark": False
                }
                
                logger.debug(f"[{thread_name}] Generating Seedream image (Attempt {attempt + 1}/{max_retries})")
                
                # Make API request with longer timeout
                response = session.post(
                    SEEDREAM_API_URL, 
                    json=payload, 
                    timeout=(30, 180),  # (connection_timeout, read_timeout)
                    stream=False
                )
                
                if response.status_code != 200:
                    logger.warning(f"[{thread_name}] Seedream API Error {response.status_code}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        time.sleep(wait_time)
                        continue
                    return None
                
                result = response.json()
                
                if 'data' in result and len(result['data']) > 0:
                    image_url = result['data'][0]['url']
                    logger.info(f"[{thread_name}] Seedream image generated successfully: {image_url}")
                    return image_url
                else:
                    logger.error(f"[{thread_name}] No image data in Seedream API response")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        time.sleep(wait_time)
                        continue
                    return None
                    
            except requests.exceptions.SSLError as e:
                logger.error(f"[{thread_name}] SSL Error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 3)
                    continue
                return None
                    
            except requests.exceptions.ConnectionError as e:
                logger.error(f"[{thread_name}] Connection Error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 3)
                    continue
                return None
                    
            except Exception as e:
                logger.error(f"[{thread_name}] Unexpected error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 3)
                    continue
                return None
        
        return None
    
    def download_image(self, image_url):
        """Download image from URL"""
        try:
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            return None
    
    def process_single_image(self, image_path, selfie_path, replicate_api_key, enable_face_swap):
        """Process a single image with prompt generation and image generation"""
        try:
            logger.info(f"Processing single image: {image_path}")
            
            # Step 1: Get ChatGPT prompt
            logger.info(f"Getting GPT-5 prompt for {image_path}")
            prompt = self.get_chatgpt_prompt(str(image_path), replicate_api_key)
            if not prompt:
                logger.error(f"Failed to get prompt for {image_path}")
                return None
            
            logger.info(f"Filtered prompt generated for {image_path}: {prompt[:100]}...")
            
            # Step 2: Generate image with Seedream API directly
            logger.info(f"Generating Seedream image for {image_path}")
            image_url = self.generate_seedream_image(prompt, selfie_path)
            if not image_url:
                logger.error(f"Failed to generate image for {image_path}")
                return None
            
            logger.info(f"Seedream image generated successfully for {image_path}: {image_url}")
            
            # Step 3: Download generated image
            logger.info(f"Downloading generated image for {image_path}")
            image_content = self.download_image(image_url)
            if not image_content:
                logger.error(f"Failed to download generated image for {image_path}")
                return None
            
            logger.info(f"Image downloaded successfully for {image_path} ({len(image_content)} bytes)")
            
            # Step 4: Face swap (if enabled)
            final_image_content = image_content
            final_filename = self.generate_random_filename() + ".jpg"
            
            if enable_face_swap and self.face_swap_processor:
                logger.info(f"Starting face swap for {image_path}")
                
                # Save generated image temporarily for face swap
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_generated:
                    tmp_generated.write(image_content)
                    generated_image_path = tmp_generated.name
                
                # Perform face swap
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_swapped:
                    swapped_image_path = tmp_swapped.name
                
                logger.info(f"Calling face swap processor for {image_path}")
                face_swap_success = self.face_swap_processor.process_face_swap(
                    generated_image_path, selfie_path, swapped_image_path
                )
                
                if face_swap_success and os.path.exists(swapped_image_path):
                    with open(swapped_image_path, 'rb') as f:
                        final_image_content = f.read()
                    final_filename = self.generate_random_filename() + "_swapped.jpg"
                    logger.info(f"Face swap completed successfully for {image_path}")
                else:
                    logger.warning(f"Face swap failed for {image_path}, using original generated image")
                
                # Clean up temporary files
                try:
                    os.unlink(generated_image_path)
                    if os.path.exists(swapped_image_path):
                        os.unlink(swapped_image_path)
                    logger.debug(f"Cleaned up temporary files for {image_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary files: {e}")
            
            # Step 5: Create result
            result = {
                'original_image': str(image_path),
                'original_name': image_path.name,
                'generated_filename': final_filename,
                'prompt': prompt,
                'image_content': final_image_content,
                'image_url': image_url,
                'face_swapped': enable_face_swap
            }
            
            logger.info(f"Successfully processed {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing single image {image_path}: {str(e)}")
            return None
    
    def process_images(self, input_folder, selfie_file, replicate_api_key, enable_face_swap=False, comfyui_server_url=None):
        """Process all images in the input folder with parallel processing"""
        logger.info(f"Starting parallel image processing. Input folder: {input_folder}, Face swap enabled: {enable_face_swap}")
        
        results = []
        csv_data = []
        
        # Initialize face swap if enabled
        if enable_face_swap and comfyui_server_url:
            logger.info(f"Initializing face swap with server: {comfyui_server_url}")
            self.initialize_face_swap(comfyui_server_url)
        
        # Get list of image files
        supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        image_files = []
        
        for file_path in Path(input_folder).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_files.append(file_path)
        
        logger.info(f"Found {len(image_files)} supported images to process")
        
        if not image_files:
            logger.warning("No supported image files found in the input folder")
            st.warning("No supported image files found in the input folder.")
            return [], []
        
        # Save uploaded selfie temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_selfie:
            tmp_selfie.write(selfie_file.getvalue())
            selfie_path = tmp_selfie.name
        
        # Create progress tracking
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.results_container = st.container()
        
        total_images = len(image_files)
        completed_count = 0
        
        try:
            # Process images in parallel with 10 workers
            logger.info(f"Starting parallel processing with 10 workers for {total_images} images")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(
                        self.process_single_image, 
                        image_path, 
                        selfie_path, 
                        replicate_api_key, 
                        enable_face_swap
                    ): image_path 
                    for image_path in image_files
                }
                
                # Process completed tasks
                for future in as_completed(future_to_image):
                    image_path = future_to_image[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            
                            # Add to CSV data
                            csv_data.append({
                                'original_image': result['original_name'],
                                'generated_filename': result['generated_filename'],
                                'prompt': result['prompt'],
                                'face_swapped': result['face_swapped'],
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            logger.info(f"Successfully completed processing for {image_path.name}")
                        else:
                            logger.warning(f"Failed to process {image_path.name}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {image_path.name}: {str(e)}")
                    
                    # Update progress
                    progress = completed_count / total_images
                    self.progress_bar.progress(progress)
                    self.status_text.text(f"Completed {completed_count}/{total_images} images")
                    
                    # Show result in real-time if successful
                    if 'result' in locals() and result:
                        with self.results_container:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader(f"Original: {result['original_name']}")
                                original_img = Image.open(image_path)
                                st.image(original_img, use_container_width=True)
                            
                            with col2:
                                title_suffix = " (Face Swapped)" if result['face_swapped'] else ""
                                st.subheader(f"Generated: {result['generated_filename']}{title_suffix}")
                                generated_img = Image.open(io.BytesIO(result['image_content']))
                                st.image(generated_img, use_container_width=True)
                                st.text(f"Filtered Prompt: {result['prompt']}")
                            
                            st.divider()
        
        finally:
            # Clean up temporary selfie file
            try:
                os.unlink(selfie_path)
            except:
                pass
        
        self.status_text.text(f"Parallel processing complete! Successfully processed {len(results)}/{total_images} images")
        logger.info(f"Parallel processing completed. Success rate: {len(results)}/{total_images}")
        
        return results, csv_data

def main():
    st.title("🎭 Seedream + Face Swap Generator (Parallel Processing)")
    st.markdown("Generate Seedream images using filtered ChatGPT prompts, with optional face swapping and parallel processing")
    
    # Check if secrets are available
    try:
        # Test if secrets are accessible
        test_replicate_key = REPLICATE_API_KEY
        test_ark_key = ARK_API_KEY
        api_keys_configured = True
    except Exception as e:
        st.error("API keys not configured in Streamlit secrets. Please add REPLICATE_API_KEY and ARK_API_KEY to your secrets.")
        api_keys_configured = False
        return
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Face swap configuration
    st.sidebar.header("Face Swap Settings")
    enable_face_swap = st.sidebar.checkbox(
        "Enable Face Swap",
        value=False,
        help="Swap faces in generated images with your selfie"
    )
    
    # Always define comfyui_server_url
    comfyui_server_url = COMFYUI_SERVER_URL if enable_face_swap else None
    
    # File uploads
    st.sidebar.header("File Inputs")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload individual images", "Upload ZIP file"]
    )
    
    uploaded_images = []
    uploaded_zip = None
    
    if input_method == "Upload individual images":
        uploaded_files = st.sidebar.file_uploader(
            "Upload Images for Processing",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple images to process"
        )
        uploaded_images = uploaded_files if uploaded_files else []
    else:
        uploaded_zip = st.sidebar.file_uploader(
            "Upload ZIP file containing images",
            type=['zip'],
            help="Upload a ZIP file containing images to process"
        )
    
    # Selfie upload
    selfie_file = st.sidebar.file_uploader(
        "Upload Your Selfie",
        type=['jpg', 'jpeg', 'png'],
        help="This selfie will be used as reference for image generation and face swapping"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Instructions")
        
        workflow_steps = [
            "**API keys are configured** in Streamlit secrets",
            "**Upload images** you want to process",
            "**Upload your selfie** that will be used as reference",
            "**Optionally enable face swapping** for more realistic results",
            "**Click 'Start Processing'** to begin parallel processing"
        ]
        
        if enable_face_swap:
            workflow_description = """
            The app will process images in parallel with 10 workers:
            1. Generate prompts from uploaded images using ChatGPT (GPT-5) via Replicate
            2. **Filter out facial expressions** from the generated prompts
            3. Use the filtered prompts with your selfie to create new images via **Seedream API directly**
            4. **Perform face swapping** to replace faces in generated images with your selfie
            5. Generate random 32-character filenames for all outputs
            6. Create a CSV file with all prompts and metadata
            
            **Note**: Expressions like "smiling", "happy", "sad", etc. are automatically removed from prompts.
            **Performance**: Up to 10 images are processed simultaneously for faster results.
            **API**: Uses Seedream API directly (not via Replicate) for image generation.
            """
        else:
            workflow_description = """
            The app will process images in parallel with 10 workers:
            1. Generate prompts from uploaded images using ChatGPT (GPT-5) via Replicate
            2. **Filter out facial expressions** from the generated prompts
            3. Use the filtered prompts with your selfie to create new images via **Seedream API directly**
            4. Generate random 32-character filenames for all outputs
            5. Create a CSV file with all prompts and metadata
            
            **Note**: Expressions like "smiling", "happy", "sad", etc. are automatically removed from prompts.
            **Performance**: Up to 10 images are processed simultaneously for faster results.
            **API**: Uses Seedream API directly (not via Replicate) for image generation.
            """
        
        for i, step in enumerate(workflow_steps, 1):
            st.markdown(f"{i}. {step}")
        
        st.markdown(workflow_description)
        
        # Expression filtering info
        st.info("""
        **🎭 Expression Filtering**: The system automatically removes facial expressions from prompts to ensure neutral faces in generated images. 
        Filtered expressions include: smiling, happy, sad, angry, surprised, serious, laughing, crying, worried, excited, and many more.
        """)
        
        # Parallel processing info
        st.info("""
        **⚡ Parallel Processing**: Uses 10 concurrent workers to process multiple images simultaneously, significantly reducing total processing time.
        """)
        
        # API info
        st.info("""
        **🔌 Direct Seedream API**: Uses Seedream API directly instead of via Replicate for faster and more reliable image generation.
        """)
    
    with col2:
        st.header("Current Status")
        
        # Check if images are available based on input method
        if input_method == "Upload individual images":
            has_images = bool(uploaded_images)
            image_info = f"({len(uploaded_images)} files)" if uploaded_images else ""
        else:
            has_images = bool(uploaded_zip)
            image_info = f"({uploaded_zip.name})" if uploaded_zip else ""
        
        has_selfie = bool(selfie_file)
        
        checks = []
        checks.append(("API Keys Configured", "✅" if api_keys_configured else "❌ Check secrets"))
        
        if input_method == "Upload individual images":
            checks.append(("Images Uploaded", f"✅ {image_info}" if has_images else "❌"))
        else:
            checks.append(("ZIP File Uploaded", f"✅ {image_info}" if has_images else "❌"))
            
        checks.append(("Selfie Uploaded", "✅" if has_selfie else "❌"))
        checks.append(("Parallel Processing", "✅ 10 Workers"))
        checks.append(("Expression Filtering", "✅ Enabled"))
        checks.append(("Seedream API", "✅ Direct API"))
        
        if enable_face_swap:
            checks.append(("ComfyUI Server", "✅" if COMFYUI_SERVER_URL else "❌"))
        
        for check_name, status in checks:
            st.text(f"{check_name}: {status}")
        
        if enable_face_swap:
            st.info("🎭 Face swap enabled - processing will take longer but results will be more realistic")
    
    # Processing section
    st.header("Processing")
    
    required_items = [api_keys_configured, uploaded_images, selfie_file]
    
    if st.button("🚀 Start Parallel Processing", type="primary", disabled=not all(required_items)):
        if not all(required_items):
            st.error("Please provide all required inputs before starting.")
            return
        
        # Create temporary directory for uploaded images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded images to temp directory
            for uploaded_file in uploaded_images:
                file_path = Path(temp_dir) / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
            
            # Process images with parallel processing
            generator = StreamlitImageGenerator()
            results, csv_data = generator.process_images(
                temp_dir, 
                selfie_file, 
                REPLICATE_API_KEY,
                enable_face_swap,
                comfyui_server_url
            )
            
            if results:
                success_message = f"Successfully processed {len(results)} images with parallel processing!"
                if enable_face_swap:
                    success_message += " (with face swapping and expression filtering using direct Seedream API)"
                else:
                    success_message += " (with expression filtering using direct Seedream API)"
                st.success(success_message)
                
                # Create downloadable files section
                st.header("📥 Download Results")
                
                # Enhanced CSV with more details
                if csv_data:
                    csv_df = pd.DataFrame(csv_data)
                    
                    # Display preview of CSV data
                    st.subheader("📊 Processing Report Preview")
                    st.dataframe(csv_df, use_container_width=True)
                    
                    # Enhanced CSV content with metadata
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_header = f"# Seedream Image Generation Report (Parallel Processing - Direct API)\n"
                    csv_header += f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    csv_header += f"# Total images processed: {len(results)}\n"
                    csv_header += f"# Face swap enabled: {enable_face_swap}\n"
                    csv_header += f"# Expression filtering: Enabled\n"
                    csv_header += f"# Parallel workers: 10\n"
                    csv_header += f"# Input method: {input_method}\n"
                    csv_header += f"# Seedream API: Direct API (not via Replicate)\n"
                    csv_header += "#\n"
                    
                    csv_content = csv_header + csv_df.to_csv(index=False)
                    
                    # Download CSV button
                    st.download_button(
                        label="📊 Download CSV Report",
                        data=csv_content,
                        file_name=f"seedream_report_parallel_{timestamp}.csv",
                        mime="text/csv",
                        help="Download detailed report with filtered prompts and metadata"
                    )
                
                # Enhanced ZIP with organized structure
                st.subheader("📦 Generated Images")
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
                    # Add images to ZIP
                    for i, result in enumerate(results):
                        zip_file.writestr(
                            f"generated_images/{result['generated_filename']}",
                            result['image_content']
                        )
                    
                    # Add original images info (if individual uploads)
                    if input_method == "Upload individual images" and uploaded_images:
                        original_info = "# Original Images Reference\n"
                        for i, result in enumerate(results):
                            original_info += f"{result['original_name']} -> {result['generated_filename']}\n"
                        zip_file.writestr("original_reference.txt", original_info)
                    
                    # Add CSV to ZIP
                    if csv_data:
                        zip_file.writestr("processing_report.csv", csv_content)
                    
                    # Add processing summary
                    summary = f"Seedream Image Generation Summary (Parallel Processing - Direct API)\n"
                    summary += f"================================================================\n\n"
                    summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    summary += f"Input method: {input_method}\n"
                    summary += f"Total images processed: {len(results)}\n"
                    summary += f"Face swap enabled: {enable_face_swap}\n"
                    summary += f"Expression filtering: Enabled\n"
                    summary += f"Parallel workers: 10\n"
                    summary += f"Success rate: {(len(results)/len(uploaded_images)*100):.1f}%\n"
                    summary += f"Seedream API: Direct API (not via Replicate)\n\n"
                    summary += "Features:\n"
                    summary += "- Automatic expression filtering from prompts\n"
                    summary += "- Parallel processing with 10 concurrent workers\n"
                    summary += "- Random 32-character filenames\n"
                    summary += "- Direct Seedream API for faster generation\n"
                    summary += f"- Face swapping: {'Enabled' if enable_face_swap else 'Disabled'}\n\n"
                    summary += "Files included:\n"
                    summary += "- generated_images/: All generated images\n"
                    summary += "- processing_report.csv: Detailed report with filtered prompts\n"
                    if input_method == "Upload individual images":
                        summary += "- original_reference.txt: Mapping of original to generated filenames\n"
                    
                    zip_file.writestr("README.txt", summary)
                
                zip_buffer.seek(0)
                
                # Enhanced download button
                zip_label = f"📦 Download Parallel Processing Package ({len(results)} images + reports)"
                if enable_face_swap:
                    zip_label = f"📦 Download Face-Swapped Package ({len(results)} images + reports)"
                
                st.download_button(
                    label=zip_label,
                    data=zip_buffer.getvalue(),
                    file_name=f"seedream_parallel_{timestamp}.zip",
                    mime="application/zip",
                    help="Download ZIP file containing all generated images, CSV report, and documentation"
                )
                
                # Summary statistics
                st.header("📊 Processing Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Images Processed", len(results))
                
                with col2:
                    st.metric("Success Rate", f"{(len(results)/len(uploaded_images)*100):.1f}%")
                
                with col3:
                    st.metric("Parallel Workers", "10")
                
                with col4:
                    face_swap_count = len([r for r in results if r.get('face_swapped')])
                    st.metric("Face Swapped", face_swap_count)
                
                # Additional details
                st.subheader("🔍 Process Details")
                process_info = f"""
                - **Input Method**: {input_method}
                - **Face Swapping**: {'Enabled' if enable_face_swap else 'Disabled'}
                - **Expression Filtering**: Enabled (removes facial expressions from prompts)
                - **Parallel Processing**: 10 concurrent workers
                - **Processing Time**: Completed at {datetime.now().strftime('%H:%M:%S')}
                - **Output Format**: Random 32-character filenames
                - **Prompt API**: Replicate GPT-5 for prompt generation
                - **Image API**: Direct Seedream API (not via Replicate)
                """
                
                if enable_face_swap:
                    process_info += f"\n- **ComfyUI Server**: {COMFYUI_SERVER_URL}"
                
                st.markdown(process_info)
            
            else:
                st.error("No images were successfully processed. Please check your API keys and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with Streamlit, Replicate GPT-5, and Direct Seedream API | **Features**: Expression Filtering + Parallel Processing + Direct API")
    
    # Display current log file info
    if st.sidebar.button("View Log Info"):
        log_files = [f for f in os.listdir('.') if f.startswith('seedream_app_') and f.endswith('.log')]
        if log_files:
            latest_log = max(log_files, key=lambda x: os.path.getctime(x))
            st.sidebar.info(f"Latest log file: {latest_log}")
        else:
            st.sidebar.info("No log files found")

if __name__ == "__main__":
    main()
