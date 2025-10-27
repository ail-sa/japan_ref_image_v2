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

# Secure API Configuration - Use environment variables or Streamlit secrets
def get_api_keys():
    """Get API keys from environment variables or Streamlit secrets"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud hosting)
        replicate_key = st.secrets.get("REPLICATE_API_KEY")
        ark_key = st.secrets.get("ARK_API_KEY") 
        comfyui_url = st.secrets.get("COMFYUI_SERVER_URL")
    except (AttributeError, FileNotFoundError):
        # Fall back to environment variables (for local hosting)
        replicate_key = os.getenv("REPLICATE_API_KEY")
        ark_key = os.getenv("ARK_API_KEY")
        comfyui_url = os.getenv("COMFYUI_SERVER_URL")
    
    return replicate_key, ark_key, comfyui_url

# Get API keys
REPLICATE_API_KEY, ARK_API_KEY, COMFYUI_SERVER_URL = get_api_keys()

# Public API Configuration
SEEDREAM_API_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"

# Streamlit page configuration
st.set_page_config(
    page_title="Seedream Face Swap Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for required API keys at startup
def check_api_keys():
    """Check if all required API keys are available"""
    missing_keys = []
    
    if not REPLICATE_API_KEY:
        missing_keys.append("REPLICATE_API_KEY")
    if not ARK_API_KEY:
        missing_keys.append("ARK_API_KEY")
    if not COMFYUI_SERVER_URL:
        missing_keys.append("COMFYUI_SERVER_URL")
    
    if missing_keys:
        st.error(f"""
        ⚠️ **Missing API Keys**
        
        The following environment variables or Streamlit secrets are required:
        {', '.join(missing_keys)}
        
        **For Streamlit Cloud:**
        Add these to your `.streamlit/secrets.toml` file:
        ```
        REPLICATE_API_KEY = "your_replicate_key_here"
        ARK_API_KEY = "your_ark_key_here"  
        COMFYUI_SERVER_URL = "your_comfyui_server_url_here"
        ```
        
        **For local hosting:**
        Set these environment variables:
        ```bash
        export REPLICATE_API_KEY="your_replicate_key_here"
        export ARK_API_KEY="your_ark_key_here"
        export COMFYUI_SERVER_URL="your_comfyui_server_url_here"
        ```
        """)
        return False
    return True

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
    def __init__(self, server_url=None, max_workers=10):
        self.server_url = server_url or COMFYUI_SERVER_URL
        self.max_workers = max_workers
        self.session = requests.Session()
        
        logger.info(f"Initializing ComfyUI Face Swap Processor with server: {self.server_url}")
        
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
            "1": {
                "inputs": {
                    "enabled": True,
                    "swap_model": "inswapper_128.onnx",
                    "facedetection": "retinaface_resnet50",
                    "face_restore_model": "codeformer-v0.1.0.pth",
                    "face_restore_visibility": 1,
                    "codeformer_weight": 0.8,
                    "detect_gender_input": "no",
                    "detect_gender_source": "no",
                    "input_faces_index": "0",
                    "source_faces_index": "0",
                    "console_log_level": 1,
                    "input_image": ["6", 0],
                    "source_image": ["5", 0],
                    "options": ["2", 0]
                },
                "class_type": "ReActorFaceSwap",
                "_meta": {"title": "ReActorFaceSwap"}
            },
            "5": {
                "inputs": {
                    "image": "selfie.jpg",
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {"title": "Load Selfie"}
            },
            "6": {
                "inputs": {
                    "image": "generated.jpg",
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {"title": "Load Generated"}
            }
        }
    
    def upload_image(self, image_path, filename):
        """Upload image to ComfyUI server"""
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f, 'image/jpeg')}
                
                response = self.session.post(
                    f"{self.server_url}/upload/image",
                    files=files,
                    timeout=60
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded {filename}")
                    with self.progress_lock:
                        self.total_uploaded += 1
                    return True
                else:
                    logger.error(f"Failed to upload {filename}: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error uploading {filename}: {str(e)}")
            return False
    
    def run_workflow(self, client_id):
        """Execute the face swap workflow"""
        try:
            # Queue the workflow
            response = self.session.post(
                f"{self.server_url}/prompt",
                json={
                    "prompt": self.workflow_template,
                    "client_id": client_id
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to queue workflow: {response.status_code}")
                return None
            
            prompt_id = response.json()["prompt_id"]
            logger.info(f"Workflow queued with prompt_id: {prompt_id}")
            
            # Wait for completion
            max_wait = 300  # 5 minutes timeout
            wait_time = 0
            
            while wait_time < max_wait:
                history_response = self.session.get(
                    f"{self.server_url}/history/{prompt_id}",
                    timeout=30
                )
                
                if history_response.status_code == 200:
                    history = history_response.json()
                    
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        
                        if "4" in outputs and "images" in outputs["4"]:
                            image_info = outputs["4"]["images"][0]
                            logger.info(f"Workflow completed successfully")
                            with self.progress_lock:
                                self.total_processed += 1
                            return image_info["filename"]
                
                time.sleep(2)
                wait_time += 2
            
            logger.error(f"Workflow timeout after {max_wait} seconds")
            return None
            
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            return None
    
    def download_result(self, filename):
        """Download the result image from ComfyUI server"""
        try:
            response = self.session.get(
                f"{self.server_url}/view",
                params={
                    "filename": filename,
                    "subfolder": "",
                    "type": "output"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully downloaded result: {filename}")
                with self.progress_lock:
                    self.total_downloaded += 1
                return response.content
            else:
                logger.error(f"Failed to download result: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading result: {str(e)}")
            return None
    
    def process_face_swap(self, generated_image_path, selfie_path, output_path):
        """Process a single face swap"""
        client_id = str(uuid.uuid4())
        thread_name = threading.current_thread().name
        
        try:
            logger.info(f"[{thread_name}] Starting face swap process")
            
            # Upload images
            if not self.upload_image(selfie_path, "selfie.jpg"):
                return False
            
            if not self.upload_image(generated_image_path, "generated.jpg"):
                return False
            
            # Run workflow
            result_filename = self.run_workflow(client_id)
            if not result_filename:
                return False
            
            # Download result
            result_content = self.download_result(result_filename)
            if not result_content:
                return False
            
            # Save result
            with open(output_path, 'wb') as f:
                f.write(result_content)
            
            logger.info(f"[{thread_name}] Face swap completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"[{thread_name}] Face swap error: {str(e)}")
            return False

class StreamlitImageGenerator:
    """Enhanced image generator with parallel processing and direct Seedream API"""
    
    def __init__(self):
        self.expression_filter = ExpressionFilter()
        self.face_swap_processor = None
        
        logger.info("Initialized StreamlitImageGenerator with expression filtering")
    
    def initialize_face_swap(self, comfyui_server_url):
        """Initialize face swap processor"""
        try:
            self.face_swap_processor = ComfyUIFaceSwapProcessor(comfyui_server_url)
            logger.info("Face swap processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face swap processor: {str(e)}")
            self.face_swap_processor = None
    
    def get_session(self):
        """Create a session with SSL adapter for handling connection issues"""
        session = requests.Session()
        session.mount('https://', SSLAdapter())
        session.mount('http://', SSLAdapter())
        return session
    
    def encode_image_to_base64(self, image_path):
        """Encode image to base64 for API requests"""
        try:
            with open(image_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_image}"
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            return None
    
    def get_chatgpt_prompt(self, image_path, replicate_api_key):
        """Get Seedream-friendly prompt from GPT-5-nano via Replicate"""
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
            
            # Use Replicate's GPT-5-nano model
            logger.info("Calling Replicate GPT-5-nano model...")
            output = replicate.run(
                "openai/gpt-5-nano",
                input={
                    "prompt": "Analyze this image and create a concise, effective image generation prompt for Seedream. Include key visual elements: setting, lighting, style, colors, clothing, pose. Keep it focused and practical. RULES: 1) No facial expressions 2) No body shape or hair descriptions 3) Gender can be mentioned 4) One person in focus 5) ONLY the prompt, no explanations:",
                    "messages": [],
                    "verbosity": "medium",
                    "image_input": [base64_image],
                    "reasoning_effort": "minimal"
                }
            )
            
            # Collect streaming output
            result = ""
            for event in output:
                result += str(event)
            
            # Clean the result to extract only the prompt
            cleaned_result = result.strip()
            
            # Remove common prefixes that GPT might add
            prefixes_to_remove = [
                "Here's a prompt for Seedream:",
                "Prompt:",
                "Here's the prompt:",
                "Seedream prompt:",
                "Image generation prompt:",
                "I can help you craft a seed generation prompt, but I can't describe or identify a real person in the image. Here's a generic prompt you can adapt for Seedream while keeping the requested constraints (one person in focus, mention facial expression, gender allowed, no body/hair description):",
                "Notes:",
                "- If you want",
                "- You can specify",
                "- Ensure only"
            ]
            
            for prefix in prefixes_to_remove:
                if cleaned_result.startswith(prefix):
                    cleaned_result = cleaned_result[len(prefix):].strip()
            
            # Split by common separators and take the first meaningful part
            separators = ["\n\nNotes:", "\nNotes:", "Notes:", "\n- If you want", "\n- You can specify", "\n- Ensure only"]
            for separator in separators:
                if separator in cleaned_result:
                    cleaned_result = cleaned_result.split(separator)[0].strip()
            
            # Remove any trailing colons or quotes
            cleaned_result = cleaned_result.rstrip(':"').strip()
            
            # Since we've instructed GPT-5-nano not to include expressions, we'll keep all descriptive content
            # Only minimal cleanup - no expression filtering
            final_result = cleaned_result
            
            logger.info(f"GPT-5-nano raw output: {result[:200]}...")
            logger.info(f"GPT-5-nano cleaned prompt (no filtering): {final_result[:100]}...")
            return final_result
            
        except Exception as e:
            logger.error(f"Error getting GPT-5-nano prompt for {image_path}: {str(e)}")
            st.error(f"Error getting GPT-5-nano prompt: {str(e)}")
            return None
    
    def generate_seedream_image(self, prompt, selfie_path):
        """Generate image using Seedream API directly (not via Replicate)"""
        session = self.get_session()
        max_retries = 3
        thread_name = threading.current_thread().name
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[{thread_name}] Attempt {attempt + 1}/{max_retries} - Calling Seedream API directly")
                
                # Encode selfie image
                with open(selfie_path, 'rb') as f:
                    selfie_content = f.read()
                selfie_base64 = base64.b64encode(selfie_content).decode('utf-8')
                
                # Prepare the request payload for Seedream API
                payload = {
                    "model_id": "general_v1.5",
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, distorted, deformed",
                    "image_strength": 0.75,
                    "cfg_scale": 7.5,
                    "steps": 25,
                    "seed": random.randint(1, 1000000),
                    "width": 512,
                    "height": 512,
                    "image": selfie_base64,
                    "use_sr": True,
                    "sr_scale": 2
                }
                
                # Headers for Seedream API
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {ARK_API_KEY}'
                }
                
                logger.info(f"[{thread_name}] Sending request to Seedream API with prompt: {prompt[:50]}...")
                
                response = session.post(
                    SEEDREAM_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                logger.info(f"[{thread_name}] Seedream API response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'data' in result and len(result['data']) > 0:
                        image_url = result['data'][0]['url']
                        logger.info(f"[{thread_name}] Seedream image generated successfully: {image_url}")
                        return image_url
                    else:
                        logger.error(f"[{thread_name}] No image data in Seedream response: {result}")
                        return None
                else:
                    logger.error(f"[{thread_name}] Seedream API error {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.info(f"[{thread_name}] Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"[{thread_name}] Error calling Seedream API (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"[{thread_name}] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        logger.error(f"[{thread_name}] Failed to generate image after {max_retries} attempts")
        return None
    
    def download_image(self, image_url):
        """Download image from URL"""
        session = self.get_session()
        try:
            response = session.get(image_url, timeout=60)
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Failed to download image: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            return None
    
    def generate_random_filename(self, length=32):
        """Generate a random filename"""
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string
    
    def process_single_image(self, image_path, selfie_path, replicate_api_key, enable_face_swap=False):
        """Process a single image with enhanced parallel-safe processing"""
        thread_name = threading.current_thread().name
        
        try:
            logger.info(f"[{thread_name}] Starting processing for {image_path}")
            
            # Step 1: Get ChatGPT prompt
            logger.info(f"Getting GPT-5-nano prompt for {image_path}")
            prompt = self.get_chatgpt_prompt(image_path, replicate_api_key)
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
        
        # Create temporary selfie file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_selfie:
            temp_selfie.write(selfie_file.getvalue())
            selfie_path = temp_selfie.name
        
        try:
            # Get all image files
            input_path = Path(input_folder)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
            image_files = [f for f in input_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            
            logger.info(f"Found {len(image_files)} images to process")
            
            if not image_files:
                logger.warning("No valid image files found")
                return [], []
            
            # Parallel processing with ThreadPoolExecutor
            max_workers = 10
            logger.info(f"Starting parallel processing with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(
                        self.process_single_image,
                        image_file,
                        selfie_path,
                        replicate_api_key,
                        enable_face_swap
                    ): image_file for image_file in image_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_image):
                    image_file = future_to_image[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            
                            # Create CSV row
                            csv_row = {
                                'original_filename': result['original_name'],
                                'generated_filename': result['generated_filename'],
                                'prompt': result['prompt'],
                                'image_url': result['image_url'],
                                'face_swapped': result['face_swapped'],
                                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            csv_data.append(csv_row)
                            
                            logger.info(f"Completed processing: {image_file.name}")
                        else:
                            logger.error(f"Failed to process: {image_file.name}")
                            
                    except Exception as e:
                        logger.error(f"Error processing {image_file.name}: {str(e)}")
            
            logger.info(f"Parallel processing completed. Successfully processed {len(results)}/{len(image_files)} images")
            
        finally:
            # Clean up temporary selfie file
            try:
                os.unlink(selfie_path)
                logger.debug("Cleaned up temporary selfie file")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary selfie file: {e}")
        
        return results, csv_data

def main():
    """Main Streamlit application"""
    st.title("🎭 Seedream Face Swap Generator")
    st.markdown("**Advanced Parallel Processing with Direct Seedream API**")
    
    # Check API keys first
    if not check_api_keys():
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Upload individual images"]
    )
    
    # Face swap option
    enable_face_swap = st.sidebar.checkbox(
        "🔄 Enable Face Swapping",
        value=False,
        help="Use ComfyUI to swap faces in generated images with your selfie"
    )
    
    # ComfyUI server configuration (only show if face swap is enabled)
    comfyui_server_url = None
    if enable_face_swap:
        # Use the configured server URL from secrets/env vars
        comfyui_server_url = COMFYUI_SERVER_URL
        st.sidebar.info(f"🖥️ ComfyUI Server: {comfyui_server_url[:30]}..." if comfyui_server_url else "❌ ComfyUI Server not configured")
    
    # Display workflow information
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Process Overview")
    
    if enable_face_swap:
        workflow_description = """
        The app will process images in parallel with 10 workers:
        1. Generate concise, effective prompts from uploaded images using ChatGPT (GPT-5-nano) via Replicate
        2. **Focus on key visual elements** without facial expressions
        3. Use the optimized prompts with your selfie to create new images via **Seedream API directly**
        4. **Perform face swapping** to replace faces in generated images with your selfie
        5. Generate random 32-character filenames for all outputs
        6. Create a CSV file with all prompts and metadata
        
        **Note**: GPT-5-nano creates focused, practical prompts without facial expressions.
        """
    else:
        workflow_description = """
        The app will process images in parallel with 10 workers:
        1. Generate concise, effective prompts from uploaded images using ChatGPT (GPT-5-nano) via Replicate
        2. **Focus on key visual elements** without facial expressions
        3. Use the optimized prompts with your selfie to create new images via **Seedream API directly**
        4. Generate random 32-character filenames for all outputs
        5. Create a CSV file with all prompts and metadata
        
        **Note**: GPT-5-nano creates focused, practical prompts without facial expressions.
        **Performance**: Up to 10 images are processed simultaneously for faster results.
        **API**: Uses Seedream API directly (not via Replicate) for image generation.
        """
    
    st.sidebar.markdown(workflow_description)
    
    # Expression filtering info
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Features")
    st.sidebar.markdown("""
    - **Parallel Processing**: 10 concurrent workers
    - **Expression Prevention**: No facial expressions in prompts
    - **Direct API**: Seedream API (not via Replicate)
    - **Random Filenames**: 32-character unique names
    - **Comprehensive Reports**: CSV + ZIP downloads
    - **Face Swapping**: Optional ComfyUI integration
    """)
    
    # Main content area
    st.header("📤 Upload Files")
    
    # File uploaders
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🤳 Your Selfie")
        selfie_file = st.file_uploader(
            "Upload your selfie image:",
            type=['jpg', 'jpeg', 'png'],
            help="This image will be used as the face source for generation and swapping"
        )
        
        if selfie_file:
            selfie_image = Image.open(selfie_file)
            st.image(selfie_image, caption="Your uploaded selfie", use_column_width=True)
    
    with col2:
        st.subheader("🖼️ Images to Process")
        if input_method == "Upload individual images":
            uploaded_images = st.file_uploader(
                "Upload images to process:",
                type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp'],
                accept_multiple_files=True,
                help="Upload multiple images that will be used to generate creative prompts"
            )
            
            if uploaded_images:
                st.success(f"✅ {len(uploaded_images)} images uploaded")
                
                # Show preview of uploaded images
                if len(uploaded_images) <= 5:
                    st.write("**Preview:**")
                    preview_cols = st.columns(min(len(uploaded_images), 3))
                    for i, img_file in enumerate(uploaded_images[:3]):
                        with preview_cols[i]:
                            img = Image.open(img_file)
                            st.image(img, caption=img_file.name, use_column_width=True)
                    
                    if len(uploaded_images) > 3:
                        st.info(f"...and {len(uploaded_images) - 3} more images")
                else:
                    st.info(f"📁 {len(uploaded_images)} images ready for processing")
    
    # Processing section
    st.header("🚀 Start Processing")
    
    # Validation and processing button
    if st.button("🎬 Generate Images", type="primary", use_container_width=True):
        # Validate inputs
        required_items = [selfie_file]
        
        if input_method == "Upload individual images":
            required_items.extend([uploaded_images])
        
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
                    success_message += " (with face swapping and optimized prompts using direct Seedream API)"
                else:
                    success_message += " (with optimized prompts using direct Seedream API)"
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
                    csv_header += f"# Optimized prompts: Enabled\n"
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
                        help="Download detailed report with optimized prompts and metadata"
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
                    summary += f"Optimized prompts: Enabled\n"
                    summary += f"Parallel workers: 10\n"
                    summary += f"Success rate: {(len(results)/len(uploaded_images)*100):.1f}%\n"
                    summary += f"Seedream API: Direct API (not via Replicate)\n\n"
                    summary += "Features:\n"
                    summary += "- Concise, effective prompts focused on key visual elements\n"
                    summary += "- Parallel processing with 10 concurrent workers\n"
                    summary += "- Random 32-character filenames\n"
                    summary += "- Direct Seedream API for faster generation\n"
                    summary += f"- Face swapping: {'Enabled' if enable_face_swap else 'Disabled'}\n\n"
                    summary += "Files included:\n"
                    summary += "- generated_images/: All generated images\n"
                    summary += "- processing_report.csv: Detailed report with optimized prompts\n"
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
                - **Expression Handling**: Avoided during prompt generation
                - **Parallel Processing**: 10 concurrent workers
                - **Processing Time**: Completed at {datetime.now().strftime('%H:%M:%S')}
                - **Output Format**: Random 32-character filenames
                - **Prompt API**: Replicate GPT-5-nano for prompt generation
                - **Image API**: Direct Seedream API (not via Replicate)
                """
                
                if enable_face_swap:
                    process_info += f"\n- **ComfyUI Server**: {COMFYUI_SERVER_URL[:30]}..." if COMFYUI_SERVER_URL else "\n- **ComfyUI Server**: Not configured"
                
                st.markdown(process_info)
            
            else:
                st.error("No images were successfully processed. Please check your API keys and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with Streamlit, Replicate GPT-5-nano, and Direct Seedream API | **Features**: Optimized Prompts + Parallel Processing + Direct API")
    
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
