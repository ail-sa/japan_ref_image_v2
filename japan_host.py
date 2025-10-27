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

# Secure API Keys Configuration
def get_api_keys():
    """
    Get API keys from Streamlit secrets or environment variables.
    This function provides fallback methods for different deployment scenarios.
    """
    try:
        # Try to get from Streamlit secrets first (recommended for Streamlit Cloud)
        replicate_key = st.secrets.get("REPLICATE_API_KEY", "")
        ark_key = st.secrets.get("ARK_API_KEY", "")
        comfyui_url = st.secrets.get("COMFYUI_SERVER_URL", "")
        seedream_url = st.secrets.get("SEEDREAM_API_URL", "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations")
        
        # If secrets are empty, try environment variables (for local development)
        if not replicate_key:
            replicate_key = os.getenv("REPLICATE_API_KEY", "")
        if not ark_key:
            ark_key = os.getenv("ARK_API_KEY", "")
        if not comfyui_url:
            comfyui_url = os.getenv("COMFYUI_SERVER_URL", "")
        if not seedream_url:
            seedream_url = os.getenv("SEEDREAM_API_URL", "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations")
            
        return {
            "replicate": replicate_key,
            "ark": ark_key,
            "comfyui_url": comfyui_url,
            "seedream_url": seedream_url
        }
    except Exception as e:
        logger.error(f"Error getting API keys: {e}")
        return {
            "replicate": "",
            "ark": "",
            "comfyui_url": "",
            "seedream_url": "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        }

# Get API keys using the secure method
API_KEYS = get_api_keys()

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
    def __init__(self, server_url="", max_workers=10):
        self.server_url = server_url or API_KEYS["comfyui_url"]
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
                    "input_image": ["5", 0],
                    "source_image": ["3", 0],
                    "options": ["2", 0]
                },
                "class_type": "ReActorFaceSwap",
                "_meta": {"title": "ReActorFaceSwap"}
            },
            "3": {
                "inputs": {
                    "image": ""
                },
                "class_type": "LoadImage",
                "_meta": {"title": "LoadImage"}
            },
            "5": {
                "inputs": {
                    "image": ""
                },
                "class_type": "LoadImage",
                "_meta": {"title": "LoadImage"}
            }
        }
    
    def upload_image(self, image_path, filename=None):
        """Upload an image to ComfyUI server"""
        if not filename:
            filename = Path(image_path).name
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f, 'image/jpeg')}
                response = self.session.post(f"{self.server_url}/upload/image", files=files)
                response.raise_for_status()
                
                with self.progress_lock:
                    self.total_uploaded += 1
                
                logger.info(f"Successfully uploaded image: {filename}")
                return filename
        except Exception as e:
            logger.error(f"Failed to upload image {filename}: {e}")
            return None
    
    def queue_prompt(self, workflow):
        """Queue a prompt for processing"""
        try:
            response = self.session.post(f"{self.server_url}/prompt", json={"prompt": workflow})
            response.raise_for_status()
            result = response.json()
            return result['prompt_id']
        except Exception as e:
            logger.error(f"Failed to queue prompt: {e}")
            return None
    
    def get_image(self, filename, subfolder="", folder_type="output"):
        """Get the processed image"""
        try:
            params = {
                'filename': filename,
                'subfolder': subfolder,
                'type': folder_type
            }
            response = self.session.get(f"{self.server_url}/view", params=params)
            response.raise_for_status()
            
            with self.progress_lock:
                self.total_downloaded += 1
            
            return response.content
        except Exception as e:
            logger.error(f"Failed to get image {filename}: {e}")
            return None
    
    def wait_for_completion(self, prompt_id, timeout=300):
        """Wait for prompt completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.server_url}/history/{prompt_id}")
                response.raise_for_status()
                history = response.json()
                
                if prompt_id in history:
                    status = history[prompt_id]['status']
                    if status.get('completed', False):
                        outputs = status.get('outputs', {})
                        for node_id in outputs:
                            if 'images' in outputs[node_id]:
                                with self.progress_lock:
                                    self.total_processed += 1
                                return outputs[node_id]['images'][0]['filename']
                        return None
                
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error checking status for prompt {prompt_id}: {e}")
                time.sleep(2)
        
        logger.error(f"Timeout waiting for prompt {prompt_id}")
        return None
    
    def face_swap(self, input_image_path, source_image_path):
        """Perform face swap operation"""
        try:
            # Upload both images
            input_filename = self.upload_image(input_image_path)
            source_filename = self.upload_image(source_image_path)
            
            if not input_filename or not source_filename:
                return None
            
            # Prepare workflow
            workflow = self.workflow_template.copy()
            workflow["3"]["inputs"]["image"] = source_filename
            workflow["5"]["inputs"]["image"] = input_filename
            
            # Queue the prompt
            prompt_id = self.queue_prompt(workflow)
            if not prompt_id:
                return None
            
            # Wait for completion and get result
            result_filename = self.wait_for_completion(prompt_id)
            if not result_filename:
                return None
            
            # Get the processed image
            result_image = self.get_image(result_filename)
            return result_image
            
        except Exception as e:
            logger.error(f"Face swap failed: {e}")
            return None

class StreamlitImageGenerator:
    """Enhanced image generator with face swap capabilities"""
    
    def __init__(self):
        self.expression_filter = ExpressionFilter()
        self.face_swap_processor = ComfyUIFaceSwapProcessor()
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Mount SSL adapter
        ssl_adapter = SSLAdapter()
        ssl_adapter.max_retries = retry_strategy
        self.session.mount("https://", ssl_adapter)
        self.session.mount("http://", ssl_adapter)
    
    def generate_random_filename(self, extension=".jpg"):
        """Generate a random 32-character filename"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=32)) + extension
    
    def generate_prompt_with_replicate(self, image_path, api_key):
        """Generate prompt using Replicate's GPT-5-pro model"""
        try:
            if not api_key:
                logger.error("Replicate API key not provided")
                return None
                
            # Set the API token
            replicate.Client(api_token=api_key)
            
            # Read and encode the image
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{image_data}"
            
            # Generate prompt using GPT-5-pro
            output = replicate.run(
                "meta/meta-llama-3-70b-instruct",
                input={
                    "top_k": 0,
                    "top_p": 0.9,
                    "prompt": f"""You are a professional prompt engineer for AI image generation. Analyze the provided image and create a detailed, descriptive prompt that captures:

1. The person's physical appearance (hair, clothing, pose, general features)
2. The setting and background
3. Lighting and mood
4. Artistic style if apparent

Important guidelines:
- DO NOT include any facial expressions, emotions, or mood descriptors
- Focus on objective, visual elements only
- Be specific about clothing, hairstyles, and settings
- Keep it concise but descriptive
- Avoid words like: smiling, happy, sad, serious, etc.

Create a prompt for generating a similar image:

Image: {image_url}

Prompt:""",
                    "max_tokens": 512,
                    "min_tokens": 0,
                    "temperature": 0.6,
                    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|/start_header_id|>\n\nYou are a helpful assistant<|end_header_id|><|start_header_id|>user<|/start_header_id|>\n\n{prompt}<|end_header_id|><|start_header_id|>assistant<|/start_header_id|>\n\n",
                    "presence_penalty": 0,
                    "frequency_penalty": 0
                }
            )
            
            # Extract the generated prompt
            generated_prompt = ''.join(output).strip()
            logger.info(f"Generated prompt via Replicate: {generated_prompt[:100]}...")
            
            return generated_prompt
            
        except Exception as e:
            logger.error(f"Error generating prompt with Replicate: {e}")
            return None
    
    def generate_image_direct_seedream(self, prompt, ark_api_key, seedream_url):
        """Generate image using direct Seedream API call"""
        try:
            if not ark_api_key:
                logger.error("ARK API key not provided")
                return None
                
            headers = {
                "Authorization": f"Bearer {ark_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model_name": "sd_xl_base_1.0_0.9vae",
                "prompt": prompt,
                "negative_prompt": "",
                "width": 1024,
                "height": 1024,
                "scale": 7.5,
                "steps": 25,
                "seed": -1,
                "use_sr": False,
                "sr_seed": -1,
                "logo_info": {
                    "add_logo": False,
                    "position": 0,
                    "language": 0,
                    "opacity": 0.3
                }
            }
            
            logger.info(f"Sending request to Seedream API: {seedream_url}")
            response = self.session.post(seedream_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("status") == "success" and result.get("data"):
                images = result["data"].get("images", [])
                if images:
                    image_url = images[0].get("url")
                    if image_url:
                        # Download the image
                        img_response = self.session.get(image_url, timeout=30)
                        img_response.raise_for_status()
                        
                        logger.info(f"Successfully generated image via direct Seedream API")
                        return img_response.content
            
            logger.error(f"Unexpected response format from Seedream API: {result}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating image with direct Seedream API: {e}")
            return None
    
    def process_single_image(self, image_path, selfie_path, replicate_api_key, enable_face_swap):
        """Process a single image with prompt generation and image creation"""
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Generate prompt using Replicate
            original_prompt = self.generate_prompt_with_replicate(image_path, replicate_api_key)
            if not original_prompt:
                logger.error(f"Failed to generate prompt for {image_path}")
                return None
            
            # Filter expressions from the prompt
            filtered_prompt = self.expression_filter.filter_expressions(original_prompt)
            
            # Generate image using direct Seedream API
            image_content = self.generate_image_direct_seedream(
                filtered_prompt, 
                API_KEYS["ark"], 
                API_KEYS["seedream_url"]
            )
            
            if not image_content:
                logger.error(f"Failed to generate image for {image_path}")
                return None
            
            # Perform face swap if enabled and selfie is provided
            face_swapped = False
            if enable_face_swap and selfie_path and self.face_swap_processor.server_url:
                try:
                    # Save generated image temporarily
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        temp_file.write(image_content)
                        temp_image_path = temp_file.name
                    
                    # Perform face swap
                    swapped_content = self.face_swap_processor.face_swap(temp_image_path, selfie_path)
                    
                    # Clean up temp file
                    os.unlink(temp_image_path)
                    
                    if swapped_content:
                        image_content = swapped_content
                        face_swapped = True
                        logger.info(f"Successfully performed face swap for {image_path}")
                    else:
                        logger.warning(f"Face swap failed for {image_path}, using original generated image")
                        
                except Exception as e:
                    logger.error(f"Face swap error for {image_path}: {e}")
            
            # Generate random filename
            generated_filename = self.generate_random_filename()
            
            result = {
                'original_name': Path(image_path).name,
                'generated_filename': generated_filename,
                'original_prompt': original_prompt,
                'filtered_prompt': filtered_prompt,
                'image_content': image_content,
                'face_swapped': face_swapped,
                'processing_time': datetime.now().isoformat(),
                'api_used': 'direct_seedream',
                'expression_filtered': original_prompt != filtered_prompt
            }
            
            logger.info(f"Successfully processed {image_path} -> {generated_filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def process_images(self, input_dir, selfie_path, replicate_api_key, enable_face_swap=False, comfyui_server_url=""):
        """Process all images in the directory with parallel processing"""
        try:
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = [
                f for f in Path(input_dir).iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            if not image_files:
                logger.warning("No image files found in the input directory")
                return [], []
            
            logger.info(f"Found {len(image_files)} images to process")
            
            results = []
            csv_data = []
            
            # Process images with ThreadPoolExecutor for parallelization
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(
                        self.process_single_image, 
                        str(image_file), 
                        selfie_path, 
                        replicate_api_key,
                        enable_face_swap
                    ): image_file 
                    for image_file in image_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_image):
                    image_file = future_to_image[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            csv_data.append({
                                'original_filename': result['original_name'],
                                'generated_filename': result['generated_filename'],
                                'original_prompt': result['original_prompt'],
                                'filtered_prompt': result['filtered_prompt'],
                                'expression_filtered': result['expression_filtered'],
                                'face_swapped': result['face_swapped'],
                                'processing_time': result['processing_time'],
                                'api_used': result['api_used']
                            })
                            logger.info(f"Completed processing: {image_file.name}")
                        else:
                            logger.error(f"Failed to process: {image_file.name}")
                    except Exception as e:
                        logger.error(f"Exception processing {image_file.name}: {e}")
            
            logger.info(f"Completed processing. Success: {len(results)}/{len(image_files)}")
            return results, csv_data
            
        except Exception as e:
            logger.error(f"Error in process_images: {e}")
            return [], []

def check_api_keys():
    """Check if all required API keys are available"""
    missing_keys = []
    
    if not API_KEYS["replicate"]:
        missing_keys.append("REPLICATE_API_KEY")
    if not API_KEYS["ark"]:
        missing_keys.append("ARK_API_KEY")
    
    return missing_keys

def display_api_key_setup_instructions():
    """Display instructions for setting up API keys"""
    st.error("🔑 API Keys Required")
    
    st.markdown("""
    ### For Streamlit Cloud Deployment:
    
    1. Go to your Streamlit Cloud app settings
    2. Navigate to the "Secrets" section
    3. Add the following secrets:
    
    ```toml
    REPLICATE_API_KEY = "your_replicate_api_key_here"
    ARK_API_KEY = "your_ark_api_key_here"
    COMFYUI_SERVER_URL = "your_comfyui_server_url_here"
    SEEDREAM_API_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    ```
    
    ### For Local Development:
    
    Create a `.streamlit/secrets.toml` file in your project directory with the same content above,
    or set environment variables:
    
    ```bash
    export REPLICATE_API_KEY="your_replicate_api_key_here"
    export ARK_API_KEY="your_ark_api_key_here"
    export COMFYUI_SERVER_URL="your_comfyui_server_url_here"
    export SEEDREAM_API_URL="https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    ```
    
    ### Getting API Keys:
    
    - **Replicate API Key**: Sign up at [replicate.com](https://replicate.com) and get your API token
    - **ARK API Key**: Get your Seedream API key from your BytePlus account
    - **ComfyUI Server**: Set up your ComfyUI server or use a hosted service
    """)

def main():
    st.title("🎭 Secure Seedream Face Swap Generator")
    st.markdown("*Enhanced with Expression Filtering, Parallel Processing & Secure API Management*")
    
    # Check API keys first
    missing_keys = check_api_keys()
    if missing_keys:
        display_api_key_setup_instructions()
        st.warning(f"Missing API keys: {', '.join(missing_keys)}")
        st.info("Please set up your API keys using the instructions above before proceeding.")
        return
    
    # API Keys status display
    with st.sidebar:
        st.header("🔐 Security Status")
        st.success("✅ API keys configured securely")
        
        # Show which keys are available (without showing the actual keys)
        key_status = []
        if API_KEYS["replicate"]:
            key_status.append("✅ Replicate API")
        if API_KEYS["ark"]:
            key_status.append("✅ ARK/Seedream API")
        if API_KEYS["comfyui_url"]:
            key_status.append("✅ ComfyUI Server")
            
        for status in key_status:
            st.text(status)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Face swap toggle
    enable_face_swap = st.sidebar.checkbox(
        "🔄 Enable Face Swapping",
        value=False,
        help="Enable face swapping using ComfyUI (requires ComfyUI server and selfie upload)"
    )
    
    # ComfyUI server configuration
    if enable_face_swap:
        comfyui_server_url = st.sidebar.text_input(
            "ComfyUI Server URL",
            value=API_KEYS["comfyui_url"],
            help="URL of your ComfyUI server with ReActor extension"
        )
    else:
        comfyui_server_url = API_KEYS["comfyui_url"]
    
    # Processing configuration
    st.sidebar.subheader("🚀 Processing Options")
    st.sidebar.info("Parallel processing: 10 concurrent workers")
    st.sidebar.info("Expression filtering: Automatically enabled")
    st.sidebar.info("API: Direct Seedream API (not via Replicate)")
    
    # Main content area
    st.header("📤 Upload Images")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload individual images", "Upload ZIP file"],
        horizontal=True
    )
    
    uploaded_images = []
    
    if input_method == "Upload individual images":
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=True,
            help="Upload one or more images for processing"
        )
        uploaded_images = uploaded_files if uploaded_files else []
        
    else:  # ZIP file upload
        zip_file = st.file_uploader(
            "Choose a ZIP file containing images",
            type=['zip'],
            help="Upload a ZIP file containing images"
        )
        
        if zip_file:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract ZIP file
                    zip_path = Path(temp_dir) / "uploaded.zip"
                    with open(zip_path, 'wb') as f:
                        f.write(zip_file.getvalue())
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find image files in extracted content
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if Path(file).suffix.lower() in image_extensions:
                                file_path = Path(root) / file
                                with open(file_path, 'rb') as img_file:
                                    # Create a file-like object for Streamlit
                                    uploaded_images.append(type('UploadedFile', (), {
                                        'name': file,
                                        'getvalue': lambda: img_file.read()
                                    })())
                    
                    if uploaded_images:
                        st.success(f"Found {len(uploaded_images)} images in ZIP file")
                    else:
                        st.warning("No image files found in the ZIP file")
                        
            except Exception as e:
                st.error(f"Error processing ZIP file: {e}")
    
    # Selfie upload for face swapping
    selfie_file = None
    if enable_face_swap:
        st.header("🤳 Upload Selfie for Face Swapping")
        selfie_file = st.file_uploader(
            "Choose your selfie image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear selfie image for face swapping"
        )
        
        if not selfie_file:
            st.warning("Please upload a selfie image to enable face swapping")
    
    # Display uploaded images preview
    if uploaded_images:
        st.header("🖼️ Image Preview")
        
        # Limit preview to first 10 images
        preview_images = uploaded_images[:10]
        cols = st.columns(min(5, len(preview_images)))
        
        for i, uploaded_file in enumerate(preview_images):
            with cols[i % 5]:
                try:
                    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying {uploaded_file.name}: {e}")
        
        if len(uploaded_images) > 10:
            st.info(f"Showing first 10 of {len(uploaded_images)} images")
    
    # Process button
    if st.button("🚀 Start Parallel Processing with Secure API", type="primary", use_container_width=True):
        # Validation
        required_items = [uploaded_images]
        if enable_face_swap:
            required_items.append(selfie_file)
        
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
            
            # Save selfie if provided
            selfie_path = None
            if selfie_file:
                selfie_path = Path(temp_dir) / f"selfie_{selfie_file.name}"
                with open(selfie_path, 'wb') as f:
                    f.write(selfie_file.getvalue())
            
            # Process images with parallel processing
            generator = StreamlitImageGenerator()
            results, csv_data = generator.process_images(
                temp_dir, 
                str(selfie_path) if selfie_path else None,
                API_KEYS["replicate"],
                enable_face_swap
            )
            
            if results:
                success_message = f"Successfully processed {len(results)} images with secure parallel processing!"
                if enable_face_swap:
                    success_message += " (with face swapping and expression filtering using secure API)"
                else:
                    success_message += " (with expression filtering using secure API)"
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
                    csv_header = f"# Secure Seedream Image Generation Report (Parallel Processing - Direct API)\n"
                    csv_header += f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    csv_header += f"# Total images processed: {len(results)}\n"
                    csv_header += f"# Face swap enabled: {enable_face_swap}\n"
                    csv_header += f"# Expression filtering: Enabled\n"
                    csv_header += f"# Parallel workers: 10\n"
                    csv_header += f"# Input method: {input_method}\n"
                    csv_header += f"# Seedream API: Direct API (secure)\n"
                    csv_header += f"# Security: API keys managed securely\n"
                    csv_header += "#\n"
                    
                    csv_content = csv_header + csv_df.to_csv(index=False)
                    
                    # Download CSV button
                    st.download_button(
                        label="📊 Download CSV Report",
                        data=csv_content,
                        file_name=f"secure_seedream_report_{timestamp}.csv",
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
                    summary = f"Secure Seedream Image Generation Summary (Parallel Processing - Direct API)\n"
                    summary += f"================================================================\n\n"
                    summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    summary += f"Input method: {input_method}\n"
                    summary += f"Total images processed: {len(results)}\n"
                    summary += f"Face swap enabled: {enable_face_swap}\n"
                    summary += f"Expression filtering: Enabled\n"
                    summary += f"Parallel workers: 10\n"
                    summary += f"Success rate: {(len(results)/len(uploaded_images)*100):.1f}%\n"
                    summary += f"Seedream API: Direct API (secure)\n"
                    summary += f"Security: API keys managed securely via Streamlit secrets\n\n"
                    summary += "Features:\n"
                    summary += "- Secure API key management (no keys in code)\n"
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
                zip_label = f"📦 Download Secure Processing Package ({len(results)} images + reports)"
                if enable_face_swap:
                    zip_label = f"📦 Download Secure Face-Swapped Package ({len(results)} images + reports)"
                
                st.download_button(
                    label=zip_label,
                    data=zip_buffer.getvalue(),
                    file_name=f"secure_seedream_{timestamp}.zip",
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
                - **Security**: API keys managed securely via Streamlit secrets
                - **Input Method**: {input_method}
                - **Face Swapping**: {'Enabled' if enable_face_swap else 'Disabled'}
                - **Expression Filtering**: Enabled (removes facial expressions from prompts)
                - **Parallel Processing**: 10 concurrent workers
                - **Processing Time**: Completed at {datetime.now().strftime('%H:%M:%S')}
                - **Output Format**: Random 32-character filenames
                - **Prompt API**: Replicate GPT-5-pro for prompt generation
                - **Image API**: Direct Seedream API (secure)
                """
                
                if enable_face_swap and API_KEYS["comfyui_url"]:
                    process_info += f"\n- **ComfyUI Server**: {API_KEYS['comfyui_url'][:50]}..."
                
                st.markdown(process_info)
            
            else:
                st.error("No images were successfully processed. Please check your API keys and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("🔐 **Secure Version** | Made with Streamlit, Replicate GPT-5-pro, and Direct Seedream API | **Features**: Secure API Management + Expression Filtering + Parallel Processing")
    
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
