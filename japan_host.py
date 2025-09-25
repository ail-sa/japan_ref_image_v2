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

# Change lines 41-43 to:
REPLICATE_API_KEY = st.secrets["REPLICATE_API_KEY"]
ARK_API_KEY = st.secrets["ARK_API_KEY"]
COMFYUI_SERVER_URL = st.secrets.get("COMFYUI_SERVER_URL", "http://34.142.205.152/comfy")

# Streamlit page configuration
st.set_page_config(
    page_title="Seedream Face Swap Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        
    def initialize_face_swap(self, comfyui_server_url):
        """Initialize face swap processor"""
        self.face_swap_processor = ComfyUIFaceSwapProcessor(server_url=comfyui_server_url)
        
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
                return f"data:image/jpeg;base64,{base64_string}"
        except Exception as e:
            st.error(f"Error encoding image {image_path}: {str(e)}")
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
            
            logger.info(f"GPT-5 prompt generated successfully: {result[:100]}...")
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error getting GPT-5 prompt for {image_path}: {str(e)}")
            st.error(f"Error getting GPT-5 prompt: {str(e)}")
            return None
    
    def generate_seedream_image(self, prompt, selfie_path, ark_api_key):
        """Generate image using Seedream API"""
        try:
            # Encode selfie to base64
            base64_image = self.encode_image_to_base64(selfie_path)
            if not base64_image:
                return None
            
            # Prepare API request
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
            
            headers = {
                "Authorization": f"Bearer {ark_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations",
                json=payload,
                headers=headers,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and len(result['data']) > 0:
                    return result['data'][0]['url']
            
            st.error(f"Seedream API Error: {response.status_code} - {response.text}")
            return None
            
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None
    
    def download_image(self, image_url):
        """Download image from URL"""
        try:
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            return response.content
        except Exception as e:
            st.error(f"Error downloading image: {str(e)}")
            return None
    
    def process_images(self, input_folder, selfie_file, replicate_api_key, ark_api_key, enable_face_swap=False, comfyui_server_url=None):
        """Process all images in the input folder"""
        logger.info(f"Starting image processing. Input folder: {input_folder}, Face swap enabled: {enable_face_swap}")
        
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
        
        try:
            for i, image_path in enumerate(image_files):
                step_text = f"Processing {image_path.name} ({i+1}/{total_images})"
                self.status_text.text(step_text)
                logger.info(f"Starting processing for image {i+1}/{total_images}: {image_path.name}")
                
                # Step 1: Get ChatGPT prompt
                self.status_text.text(f"{step_text} - Getting prompt...")
                logger.info(f"Step 1: Getting GPT-5 prompt for {image_path.name}")
                prompt = self.get_chatgpt_prompt(str(image_path), replicate_api_key)
                if not prompt:
                    logger.error(f"Failed to get prompt for {image_path.name}")
                    st.warning(f"Failed to get prompt for {image_path.name}")
                    continue
                
                logger.info(f"Prompt generated for {image_path.name}: {prompt[:100]}...")
                
                # Step 2: Generate image with Seedream
                self.status_text.text(f"{step_text} - Generating image...")
                logger.info(f"Step 2: Generating Seedream image for {image_path.name}")
                image_url = self.generate_seedream_image(prompt, selfie_path, ark_api_key)
                if not image_url:
                    logger.error(f"Failed to generate image for {image_path.name}")
                    st.warning(f"Failed to generate image for {image_path.name}")
                    continue
                
                logger.info(f"Seedream image generated successfully for {image_path.name}: {image_url}")
                
                # Step 3: Download generated image
                self.status_text.text(f"{step_text} - Downloading...")
                logger.info(f"Step 3: Downloading generated image for {image_path.name}")
                image_content = self.download_image(image_url)
                if not image_content:
                    logger.error(f"Failed to download generated image for {image_path.name}")
                    st.warning(f"Failed to download generated image for {image_path.name}")
                    continue
                
                logger.info(f"Image downloaded successfully for {image_path.name} ({len(image_content)} bytes)")
                
                # Step 4: Face swap (if enabled)
                final_image_content = image_content
                final_filename = self.generate_random_filename() + ".jpg"
                
                if enable_face_swap and self.face_swap_processor:
                    self.status_text.text(f"{step_text} - Face swapping...")
                    logger.info(f"Step 4: Starting face swap for {image_path.name}")
                    
                    # Save generated image temporarily for face swap
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_generated:
                        tmp_generated.write(image_content)
                        generated_image_path = tmp_generated.name
                    
                    # Perform face swap
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_swapped:
                        swapped_image_path = tmp_swapped.name
                    
                    logger.info(f"Calling face swap processor for {image_path.name}")
                    face_swap_success = self.face_swap_processor.process_face_swap(
                        generated_image_path, selfie_path, swapped_image_path
                    )
                    
                    if face_swap_success and os.path.exists(swapped_image_path):
                        with open(swapped_image_path, 'rb') as f:
                            final_image_content = f.read()
                        final_filename = self.generate_random_filename() + "_swapped.jpg"
                        logger.info(f"Face swap completed successfully for {image_path.name}")
                    else:
                        logger.warning(f"Face swap failed for {image_path.name}, using original generated image")
                        st.warning(f"Face swap failed for {image_path.name}, using original generated image")
                    
                    # Clean up temporary files
                    try:
                        os.unlink(generated_image_path)
                        if os.path.exists(swapped_image_path):
                            os.unlink(swapped_image_path)
                        logger.debug(f"Cleaned up temporary files for {image_path.name}")
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
                
                results.append(result)
                
                # Add to CSV data
                csv_data.append({
                    'original_image': image_path.name,
                    'generated_filename': final_filename,
                    'prompt': prompt,
                    'face_swapped': enable_face_swap,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update progress
                progress = (i + 1) / total_images
                self.progress_bar.progress(progress)
                
                # Show result in real-time
                with self.results_container:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"Original: {image_path.name}")
                        original_img = Image.open(image_path)
                        st.image(original_img, use_container_width=True)
                    
                    with col2:
                        title_suffix = " (Face Swapped)" if enable_face_swap else ""
                        st.subheader(f"Generated: {final_filename}{title_suffix}")
                        generated_img = Image.open(io.BytesIO(final_image_content))
                        st.image(generated_img, use_container_width=True)
                        st.text(f"Prompt: {prompt}")
                    
                    st.divider()
                
                # Add delay to respect rate limits
                time.sleep(2)
        
        finally:
            # Clean up temporary selfie file
            try:
                os.unlink(selfie_path)
            except:
                pass
        
        self.status_text.text("Processing complete!")
        return results, csv_data

def main():
    st.title("🎭 Seedream + Face Swap Generator")
    st.markdown("Generate Seedream images using ChatGPT prompts, with optional face swapping")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Face swap configuration
    st.sidebar.header("Face Swap Settings")
    enable_face_swap = st.sidebar.checkbox(
        "Enable Face Swap",
        value=False,
        help="Swap faces in generated images with your selfie"
    )
    
    if enable_face_swap:
        comfyui_server_url = COMFYUI_SERVER_URL
    
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
            "**Update API keys** in the script configuration section",
            "**Upload images** you want to process",
            "**Upload your selfie** that will be used as reference",
            "**Optionally enable face swapping** for more realistic results",
            "**Click 'Start Processing'** to begin"
        ]
        
        if enable_face_swap:
            workflow_description = """
            The app will:
            1. Send each uploaded image to ChatGPT for prompt generation
            2. Use the generated prompt with your selfie to create new images via Seedream
            3. **Perform face swapping** to replace faces in generated images with your selfie
            4. Generate random 32-character filenames for all outputs
            5. Create a CSV file with all prompts and metadata
            
            **Note**: Face swapping requires a ComfyUI server and adds processing time.
            """
        else:
            workflow_description = """
            The app will:
            1. Send each uploaded image to ChatGPT for prompt generation
            2. Use the generated prompt with your selfie to create new images via Seedream
            3. Generate random 32-character filenames for all outputs
            4. Create a CSV file with all prompts and metadata
            """
        
        for i, step in enumerate(workflow_steps, 1):
            st.markdown(f"{i}. {step}")
        
        st.markdown(workflow_description)
    
    with col2:
        st.header("Current Status")
        
        # Validation checks
        api_keys_configured = REPLICATE_API_KEY != "your_replicate_api_key_here" and ARK_API_KEY != "your_seedream_api_key_here"
        
        # Check if images are available based on input method
        if input_method == "Upload individual images":
            has_images = bool(uploaded_images)
            image_info = f"({len(uploaded_images)} files)" if uploaded_images else ""
        else:
            has_images = bool(uploaded_zip)
            image_info = f"({uploaded_zip.name})" if uploaded_zip else ""
        
        has_selfie = bool(selfie_file)
        
        checks = []
        checks.append(("API Keys Configured", "✅" if api_keys_configured else "❌ Update in script"))
        
        if input_method == "Upload individual images":
            checks.append(("Images Uploaded", f"✅ {image_info}" if has_images else "❌"))
        else:
            checks.append(("ZIP File Uploaded", f"✅ {image_info}" if has_images else "❌"))
            
        checks.append(("Selfie Uploaded", "✅" if has_selfie else "❌"))
        
        if enable_face_swap:
            checks.append(("ComfyUI Server", "✅" if COMFYUI_SERVER_URL else "❌"))
        
        for check_name, status in checks:
            st.text(f"{check_name}: {status}")
        
        if enable_face_swap:
            st.info("🎭 Face swap enabled - processing will take longer but results will be more realistic")
        
        if not api_keys_configured:
            st.warning("⚠️ Please update REPLICATE_API_KEY and ARK_API_KEY in the script before running")
    
    # Processing section
    st.header("Processing")
    
    required_items = [api_keys_configured, uploaded_images, selfie_file]
    
    if st.button("🚀 Start Processing", type="primary", disabled=not all(required_items)):
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
            
            # Process images
            generator = StreamlitImageGenerator()
            results, csv_data = generator.process_images(
                temp_dir, 
                selfie_file, 
                REPLICATE_API_KEY, 
                ARK_API_KEY,
                enable_face_swap,
                comfyui_server_url
            )
            
            if results:
                success_message = f"Successfully processed {len(results)} images!"
                if enable_face_swap:
                    success_message += " (with face swapping)"
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
                    csv_header = f"# Seedream Image Generation Report\n"
                    csv_header += f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    csv_header += f"# Total images processed: {len(results)}\n"
                    csv_header += f"# Face swap enabled: {enable_face_swap}\n"
                    csv_header += f"# Input method: {input_method}\n"
                    csv_header += "#\n"
                    
                    csv_content = csv_header + csv_df.to_csv(index=False)
                    
                    # Download CSV button
                    st.download_button(
                        label="📊 Download CSV Report",
                        data=csv_content,
                        file_name=f"seedream_report_{timestamp}.csv",
                        mime="text/csv",
                        help="Download detailed report with prompts and metadata"
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
                    summary = f"Seedream Image Generation Summary\n"
                    summary += f"================================\n\n"
                    summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    summary += f"Input method: {input_method}\n"
                    summary += f"Total images processed: {len(results)}\n"
                    summary += f"Face swap enabled: {enable_face_swap}\n"
                    summary += f"Success rate: {(len(results)/len(uploaded_images if input_method == 'Upload individual images' else extracted_files)*100):.1f}%\n\n"
                    summary += "Files included:\n"
                    summary += "- generated_images/: All generated images\n"
                    summary += "- processing_report.csv: Detailed report with prompts\n"
                    if input_method == "Upload individual images":
                        summary += "- original_reference.txt: Mapping of original to generated filenames\n"
                    
                    zip_file.writestr("README.txt", summary)
                
                zip_buffer.seek(0)
                
                # Enhanced download button
                zip_label = f"📦 Download Complete Package ({len(results)} images + reports)"
                if enable_face_swap:
                    zip_label = f"📦 Download Face-Swapped Package ({len(results)} images + reports)"
                
                st.download_button(
                    label=zip_label,
                    data=zip_buffer.getvalue(),
                    file_name=f"seedream_complete_{timestamp}.zip",
                    mime="application/zip",
                    help="Download ZIP file containing all generated images, CSV report, and documentation"
                )
                
                # Summary statistics
                st.header("📊 Processing Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate total input images
                if input_method == "Upload individual images":
                    total_input = len(uploaded_images)
                else:
                    total_input = len(extracted_files)
                
                with col1:
                    st.metric("Images Processed", len(results))
                
                with col2:
                    st.metric("Success Rate", f"{(len(results)/total_input*100):.1f}%")
                
                with col3:
                    st.metric("Total Input Images", total_input)
                
                with col4:
                    face_swap_count = len([r for r in results if r.get('face_swapped')])
                    st.metric("Face Swapped", face_swap_count)
                
                # Additional details
                st.subheader("🔍 Process Details")
                process_info = f"""
                - **Input Method**: {input_method}
                - **Face Swapping**: {'Enabled' if enable_face_swap else 'Disabled'}
                - **Processing Time**: Completed at {datetime.now().strftime('%H:%M:%S')}
                - **Output Format**: Random 32-character filenames
                - **API Used**: Replicate GPT-5 for prompts, Seedream for generation
                """
                
                if enable_face_swap:
                    process_info += f"\n- **ComfyUI Server**: {COMFYUI_SERVER_URL}"
                
                st.markdown(process_info)
            
            else:
                st.error("No images were successfully processed. Please check your API keys and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with Streamlit, Replicate (GPT-5), Seedream, and ComfyUI APIs")
    
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