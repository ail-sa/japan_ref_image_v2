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
try:
    REPLICATE_API_KEY = st.secrets["REPLICATE_API_KEY"]
    ARK_API_KEY = st.secrets["ARK_API_KEY"]
    COMFYUI_SERVER_URL = st.secrets.get("COMFYUI_SERVER_URL", "http://34.142.205.152/comfy")
except Exception as e:
    st.error("Please configure your API keys in Streamlit secrets. Check the sidebar for instructions.")
    REPLICATE_API_KEY = None
    ARK_API_KEY = None
    COMFYUI_SERVER_URL = "http://34.142.205.152/comfy"

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
            "1": {
                "inputs": {
                    "enabled": True,
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
                    "image": "input.jpg",
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {"title": "Load Input"}
            }
        }
    
    def upload_image(self, image_data, filename):
        """Upload image to ComfyUI server"""
        files = {'image': (filename, io.BytesIO(image_data), 'image/jpeg')}
        
        try:
            response = self.session.post(f"{self.server_url}/upload/image", files=files, timeout=30)
            response.raise_for_status()
            
            with self.progress_lock:
                self.total_uploaded += 1
            
            logger.info(f"Successfully uploaded {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {filename}: {e}")
            return False
    
    def process_face_swap(self, input_image_data, selfie_image_data, input_filename, output_filename):
        """Process face swap for a single image"""
        try:
            # Upload input image
            if not self.upload_image(input_image_data, "input.jpg"):
                return None
            
            # Upload selfie
            if not self.upload_image(selfie_image_data, "selfie.jpg"):
                return None
            
            # Create workflow
            workflow = self.workflow_template.copy()
            
            # Queue the workflow
            response = self.session.post(f"{self.server_url}/prompt", json={"prompt": workflow}, timeout=60)
            response.raise_for_status()
            
            prompt_id = response.json()['prompt_id']
            logger.info(f"Queued face swap for {input_filename} with prompt_id: {prompt_id}")
            
            # Wait for completion and get result
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Check queue status
                queue_response = self.session.get(f"{self.server_url}/queue", timeout=30)
                queue_data = queue_response.json()
                
                # Check if our prompt is still running
                running_prompts = [item[1] for item in queue_data.get('queue_running', [])]
                pending_prompts = [item[1] for item in queue_data.get('queue_pending', [])]
                
                if prompt_id not in running_prompts and prompt_id not in pending_prompts:
                    # Prompt completed, get the result
                    history_response = self.session.get(f"{self.server_url}/history/{prompt_id}", timeout=30)
                    history_data = history_response.json()
                    
                    if prompt_id in history_data:
                        outputs = history_data[prompt_id].get('outputs', {})
                        
                        # Find the saved image
                        for node_id, node_output in outputs.items():
                            if 'images' in node_output:
                                for image_info in node_output['images']:
                                    image_filename = image_info['filename']
                                    
                                    # Download the image
                                    image_url = f"{self.server_url}/view?filename={image_filename}&type=output"
                                    image_response = self.session.get(image_url, timeout=60)
                                    image_response.raise_for_status()
                                    
                                    with self.progress_lock:
                                        self.total_processed += 1
                                        self.total_downloaded += 1
                                    
                                    logger.info(f"Successfully processed face swap for {input_filename}")
                                    return image_response.content
                    
                    logger.error(f"No output found for {input_filename}")
                    return None
                
                time.sleep(2)
            
            logger.error(f"Timeout waiting for face swap completion for {input_filename}")
            return None
            
        except Exception as e:
            logger.error(f"Face swap processing failed for {input_filename}: {e}")
            return None

class StreamlitImageGenerator:
    """Main image generator class for Streamlit"""
    
    def __init__(self):
        self.expression_filter = ExpressionFilter()
        self.face_swap_processor = None
    
    def generate_random_filename(self, length=32):
        """Generate a random filename"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length)) + '.jpg'
    
    def analyze_image_with_replicate(self, image_data, api_key):
        """Use Replicate to analyze image and generate prompt"""
        try:
            # Initialize Replicate client
            client = replicate.Client(api_token=api_key)
            
            # Convert image data to base64
            image_b64 = base64.b64encode(image_data).decode()
            data_uri = f"data:image/jpeg;base64,{image_b64}"
            
            # Use GPT-5-pro for image analysis
            output = client.run(
                "meta/meta-llama-3-70b-instruct",
                input={
                    "image": data_uri,
                    "prompt": "Describe this image in detail for AI image generation. Include clothing, setting, pose, lighting, and style. Be specific and detailed.",
                    "max_tokens": 512,
                    "temperature": 0.7
                }
            )
            
            # Extract the description
            if isinstance(output, list):
                description = ''.join(output)
            else:
                description = str(output)
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Replicate analysis failed: {e}")
            return "high quality portrait, professional photography"
    
    def generate_with_seedream_api(self, prompt, ark_api_key):
        """Generate image using direct Seedream API"""
        try:
            headers = {
                'Authorization': f'Bearer {ark_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": "general_v1.5",
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "response_format": "url"
            }
            
            # Create session with SSL adapter
            session = requests.Session()
            session.mount('https://', SSLAdapter())
            
            response = session.post(SEEDREAM_API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            if 'data' in result and len(result['data']) > 0:
                image_url = result['data'][0]['url']
                
                # Download the image
                img_response = session.get(image_url, timeout=60)
                img_response.raise_for_status()
                
                return img_response.content
            else:
                logger.error("No image data in Seedream API response")
                return None
                
        except Exception as e:
            logger.error(f"Seedream API generation failed: {e}")
            return None
    
    def process_single_image(self, args):
        """Process a single image (for parallel processing)"""
        image_path, selfie_data, replicate_api_key, ark_api_key, enable_face_swap, comfyui_server_url = args
        
        try:
            # Read and analyze the image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            original_name = Path(image_path).name
            
            # Analyze image with Replicate
            prompt = self.analyze_image_with_replicate(image_data, replicate_api_key)
            if not prompt:
                return None
            
            # Filter expressions from prompt
            filtered_prompt = self.expression_filter.filter_expressions(prompt)
            
            # Generate image with Seedream API
            generated_image_data = self.generate_with_seedream_api(filtered_prompt, ark_api_key)
            if not generated_image_data:
                return None
            
            # Apply face swap if enabled
            face_swapped = False
            if enable_face_swap and selfie_data:
                if not self.face_swap_processor:
                    self.face_swap_processor = ComfyUIFaceSwapProcessor(comfyui_server_url)
                
                swapped_data = self.face_swap_processor.process_face_swap(
                    generated_image_data, selfie_data, original_name, "output.jpg"
                )
                
                if swapped_data:
                    generated_image_data = swapped_data
                    face_swapped = True
            
            # Generate random filename
            generated_filename = self.generate_random_filename()
            
            return {
                'original_name': original_name,
                'generated_filename': generated_filename,
                'image_content': generated_image_data,
                'original_prompt': prompt,
                'filtered_prompt': filtered_prompt,
                'face_swapped': face_swapped,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return {
                'original_name': Path(image_path).name,
                'error': str(e),
                'success': False
            }
    
    def process_images(self, image_dir, selfie_file, replicate_api_key, enable_face_swap=False, comfyui_server_url=None):
        """Process all images in directory with parallel processing"""
        
        # Get selfie data if provided
        selfie_data = None
        if enable_face_swap and selfie_file:
            selfie_data = selfie_file.getvalue()
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [
            f for f in Path(image_dir).iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            return [], []
        
        # Prepare arguments for parallel processing
        args_list = [
            (str(img_path), selfie_data, replicate_api_key, ARK_API_KEY, enable_face_swap, comfyui_server_url)
            for img_path in image_files
        ]
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        csv_data = []
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(self.process_single_image, args): args for args in args_list}
            
            completed = 0
            total = len(args_list)
            
            # Process completed tasks
            for future in as_completed(future_to_args):
                result = future.result()
                completed += 1
                
                # Update progress
                progress = completed / total
                progress_bar.progress(progress)
                status_text.text(f"Processing: {completed}/{total} images completed")
                
                if result and result.get('success'):
                    results.append(result)
                    
                    # Add to CSV data
                    csv_data.append({
                        'original_name': result['original_name'],
                        'generated_filename': result['generated_filename'],
                        'original_prompt': result['original_prompt'],
                        'filtered_prompt': result['filtered_prompt'],
                        'face_swapped': result['face_swapped'],
                        'timestamp': datetime.now().isoformat()
                    })
                elif result:
                    # Log error
                    logger.error(f"Failed to process {result.get('original_name', 'unknown')}: {result.get('error', 'unknown error')}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results, csv_data

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🎭 Seedream Face Swap Generator")
    st.markdown("Upload images to generate AI versions with optional face swapping and expression filtering")
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API key status
        if not REPLICATE_API_KEY or not ARK_API_KEY:
            st.error("🔑 API Keys Required")
            st.markdown("""
            **To use this app, you need to configure your API keys in Streamlit secrets:**
            
            1. Go to your Streamlit Cloud dashboard
            2. Click on your app settings
            3. Go to the "Secrets" tab
            4. Add the following secrets:
            
            ```toml
            REPLICATE_API_KEY = "your_replicate_api_key_here"
            ARK_API_KEY = "your_ark_api_key_here"
            COMFYUI_SERVER_URL = "http://34.142.205.152/comfy"  # optional
            ```
            
            **API Key Sources:**
            - Replicate API: Get from [replicate.com](https://replicate.com)
            - ARK API: Get from BytePlus Console
            """)
            
            st.stop()
        else:
            st.success("✅ API Keys Configured")
        
        # Face swap configuration
        st.subheader("🔄 Face Swap Options")
        enable_face_swap = st.checkbox("Enable Face Swapping", value=False)
        
        if enable_face_swap:
            st.info("Upload a selfie below to swap faces in generated images")
            comfyui_server_url = COMFYUI_SERVER_URL
        else:
            comfyui_server_url = None
        
        # Processing info
        st.subheader("📊 Processing Info")
        st.markdown("""
        **Features:**
        - Parallel processing (10 workers)
        - Expression filtering
        - Direct Seedream API
        - Optional face swapping
        - Detailed CSV reports
        """)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Images")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload individual images", "Upload ZIP file"],
            help="Select how you want to provide images for processing"
        )
        
        uploaded_images = []
        
        if input_method == "Upload individual images":
            uploaded_files = st.file_uploader(
                "Choose image files",
                type=['png', 'jpg', 'jpeg', 'webp'],
                accept_multiple_files=True,
                help="Upload multiple image files (PNG, JPG, JPEG, WebP)"
            )
            uploaded_images = uploaded_files if uploaded_files else []
            
        else:  # ZIP file upload
            zip_file = st.file_uploader(
                "Choose ZIP file containing images",
                type=['zip'],
                help="Upload a ZIP file containing image files"
            )
            
            if zip_file:
                # Extract and validate ZIP contents
                try:
                    with zipfile.ZipFile(io.BytesIO(zip_file.getvalue())) as zf:
                        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
                        image_files = [
                            name for name in zf.namelist() 
                            if Path(name).suffix.lower() in image_extensions and not name.startswith('__MACOSX/')
                        ]
                        
                        if image_files:
                            st.success(f"Found {len(image_files)} images in ZIP file")
                            
                            # Create temporary uploaded file objects
                            uploaded_images = []
                            for img_name in image_files[:50]:  # Limit to 50 images
                                img_data = zf.read(img_name)
                                # Create a simple object that mimics UploadedFile
                                class TempFile:
                                    def __init__(self, name, data):
                                        self.name = Path(name).name
                                        self.data = data
                                    def getvalue(self):
                                        return self.data
                                
                                uploaded_images.append(TempFile(img_name, img_data))
                        else:
                            st.error("No valid image files found in ZIP")
                            
                except Exception as e:
                    st.error(f"Error reading ZIP file: {e}")
    
    with col2:
        st.header("🤳 Face Swap Setup")
        
        if enable_face_swap:
            selfie_file = st.file_uploader(
                "Upload your selfie",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear photo of your face for face swapping"
            )
            
            if selfie_file:
                # Display preview
                selfie_image = Image.open(selfie_file)
                st.image(selfie_image, caption="Selfie Preview", width=200)
                
                # Reset file pointer
                selfie_file.seek(0)
        else:
            selfie_file = None
            st.info("Face swapping is disabled. Enable it in the sidebar to upload a selfie.")
    
    # Processing section
    st.header("🚀 Processing")
    
    if uploaded_images:
        st.success(f"Ready to process {len(uploaded_images)} images")
        
        # Show preview of uploaded images
        if len(uploaded_images) <= 5:
            cols = st.columns(min(len(uploaded_images), 5))
            for i, img_file in enumerate(uploaded_images[:5]):
                with cols[i]:
                    try:
                        img = Image.open(io.BytesIO(img_file.getvalue()))
                        st.image(img, caption=img_file.name[:20], width=100)
                    except:
                        st.text(img_file.name[:20])
        else:
            st.info(f"Preview: {uploaded_images[0].name} and {len(uploaded_images)-1} more images")
        
        # Process button
        if st.button("🎯 Start Processing", type="primary", use_container_width=True):
            process_images(uploaded_images, selfie_file, input_method, enable_face_swap, comfyui_server_url)
    else:
        st.info("Please upload images to begin processing")

def process_images(uploaded_images, selfie_file, input_method, enable_face_swap, comfyui_server_url):
    """Process the uploaded images"""
    
    # Validation
    required_items = [
        uploaded_images,
        REPLICATE_API_KEY,
        ARK_API_KEY
    ]
    
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
            - **Prompt API**: Replicate Meta Llama 3 70B for prompt generation
            - **Image API**: Direct Seedream API (not via Replicate)
            """
            
            if enable_face_swap:
                process_info += f"\n- **ComfyUI Server**: {COMFYUI_SERVER_URL}"
            
            st.markdown(process_info)
        
        else:
            st.error("No images were successfully processed. Please check your API keys and try again.")

    # Footer
    st.markdown("---")
    st.markdown("Made with Streamlit, Replicate Meta Llama 3, and Direct Seedream API | **Features**: Expression Filtering + Parallel Processing + Direct API")
    
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
