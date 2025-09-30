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
import queue

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

class ProgressTracker:
    """Thread-safe progress tracker for parallel processing"""
    def __init__(self, total_items):
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.current_stage = {}
        self.lock = threading.Lock()
        self.progress_bar = None
        self.status_text = None
        
    def set_ui_elements(self, progress_bar, status_text):
        self.progress_bar = progress_bar
        self.status_text = status_text
    
    def update_stage(self, item_id, stage):
        with self.lock:
            self.current_stage[item_id] = stage
            self._update_ui()
    
    def complete_item(self, item_id, success=True):
        with self.lock:
            if success:
                self.completed_items += 1
            else:
                self.failed_items += 1
            
            if item_id in self.current_stage:
                del self.current_stage[item_id]
            
            self._update_ui()
    
    def _update_ui(self):
        if self.progress_bar and self.status_text:
            progress = (self.completed_items + self.failed_items) / self.total_items
            self.progress_bar.progress(progress)
            
            active_stages = list(self.current_stage.values())
            status_msg = f"Completed: {self.completed_items}/{self.total_items}"
            if self.failed_items > 0:
                status_msg += f" | Failed: {self.failed_items}"
            if active_stages:
                status_msg += f" | Active: {', '.join(active_stages[:3])}"
                if len(active_stages) > 3:
                    status_msg += f" (+{len(active_stages)-3} more)"
            
            self.status_text.text(status_msg)

class ExpressionFilter:
    """Filter to remove facial expressions from prompts"""
    
    def __init__(self):
        # Common expression keywords to filter out
        self.expression_patterns = [
            r'\b(smiling|smile|smiles)\b',
            r'\b(happy|joyful|cheerful)\b',
            r'\b(sad|sadness|melancholy)\b',
            r'\b(angry|anger|furious|mad)\b',
            r'\b(serious|stern|stoic)\b',
            r'\b(laughing|laugh|giggles)\b',
            r'\b(crying|tears|weeping)\b',
            r'\b(surprised|shock|amazed)\b',
            r'\b(confused|puzzled|bewildered)\b',
            r'\b(excited|enthusiastic|thrilled)\b',
            r'\b(calm|peaceful|serene)\b',
            r'\b(worried|anxious|concerned)\b',
            r'\b(confident|proud|determined)\b',
            r'\b(disappointed|frustrated|annoyed)\b',
            r'\b(expression|facial expression)\b',
            r'\bwith\s+a\s+\w+\s+expression\b',
            r'\bexpressing\s+\w+\b',
            r'\blooking\s+(happy|sad|angry|serious|confused|excited|worried|calm|confident|disappointed)\b'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.expression_patterns]
    
    def filter_expressions(self, prompt):
        """Remove expression-related terms from prompt"""
        filtered_prompt = prompt
        
        for pattern in self.compiled_patterns:
            filtered_prompt = pattern.sub('', filtered_prompt)
        
        # Clean up extra spaces and punctuation
        filtered_prompt = re.sub(r'\s+', ' ', filtered_prompt)  # Multiple spaces to single
        filtered_prompt = re.sub(r',\s*,', ',', filtered_prompt)  # Double commas
        filtered_prompt = re.sub(r'\s*,\s*', ', ', filtered_prompt)  # Space around commas
        filtered_prompt = re.sub(r'^\s*,\s*|\s*,\s*$', '', filtered_prompt)  # Leading/trailing commas
        filtered_prompt = filtered_prompt.strip()
        
        return filtered_prompt

class ComfyUIFaceSwapProcessor:
    """Face swap processor using ComfyUI"""
    def __init__(self, server_url="http://34.142.205.152/comfy", max_workers=10):
        self.server_url = server_url
        self.max_workers = max_workers
        self.session_pool = queue.Queue()
        
        # Initialize session pool
        for _ in range(max_workers):
            session = requests.Session()
            self.session_pool.put(session)
        
        logger.info(f"Initializing ComfyUI Face Swap Processor with server: {server_url}")
        
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
    
    def get_session(self):
        """Get a session from the pool"""
        return self.session_pool.get()
    
    def return_session(self, session):
        """Return a session to the pool"""
        self.session_pool.put(session)
    
    def upload_image(self, image_path, filename, session=None):
        """Upload image to ComfyUI server"""
        logger.info(f"Uploading image: {filename} from {image_path}")
        
        if session is None:
            session = self.get_session()
            return_session = True
        else:
            return_session = False
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f, 'image/jpeg')}
                response = session.post(f"{self.server_url}/upload/image", files=files)
                
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded: {filename}")
                    return True
                else:
                    logger.error(f"Failed to upload {filename}: HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error uploading image {filename}: {e}")
            return False
        finally:
            if return_session:
                self.return_session(session)
    
    def submit_workflow(self, input_filename, source_filename, output_prefix, session=None):
        """Submit face swap workflow"""
        logger.info(f"Submitting workflow: input={input_filename}, source={source_filename}")
        
        if session is None:
            session = self.get_session()
            return_session = True
        else:
            return_session = False
        
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
            
            response = session.post(f"{self.server_url}/prompt", json=payload)
            
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
            return None, None
        finally:
            if return_session:
                self.return_session(session)
    
    def wait_for_completion(self, prompt_id, session=None, timeout=300):
        """Wait for workflow completion"""
        if session is None:
            session = self.get_session()
            return_session = True
        else:
            return_session = False
        
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < timeout:
                try:
                    response = session.get(f"{self.server_url}/history/{prompt_id}")
                    if response.status_code == 200:
                        history = response.json()
                        if prompt_id in history:
                            return True
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error checking status: {e}")
                    break
            
            return False
        finally:
            if return_session:
                self.return_session(session)
    
    def download_result(self, prompt_id, output_path, session=None):
        """Download processed image"""
        if session is None:
            session = self.get_session()
            return_session = True
        else:
            return_session = False
        
        try:
            response = session.get(f"{self.server_url}/history/{prompt_id}")
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
                        
                        response = session.get(f"{self.server_url}/view", params=params)
                        
                        if response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                f.write(response.content)
                            return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error downloading result: {e}")
            return False
        finally:
            if return_session:
                self.return_session(session)
    
    def process_face_swap(self, input_image_path, source_face_path, output_path):
        """Process a single face swap"""
        session = self.get_session()
        try:
            # Generate unique filenames
            input_filename = f"input_{uuid.uuid4()}.jpg"
            source_filename = f"source_{uuid.uuid4()}.jpg"
            
            # Upload images
            if not self.upload_image(input_image_path, input_filename, session):
                return False
            
            if not self.upload_image(source_face_path, source_filename, session):
                return False
            
            # Submit workflow
            prompt_id, client_id = self.submit_workflow(
                input_filename, 
                source_filename, 
                f"swapped_{uuid.uuid4()}",
                session
            )
            
            if not prompt_id:
                return False
            
            # Wait for completion
            if not self.wait_for_completion(prompt_id, session):
                return False
            
            # Download result
            return self.download_result(prompt_id, output_path, session)
        
        except Exception as e:
            logger.error(f"Error in face swap processing: {e}")
            return False
        finally:
            self.return_session(session)

class StreamlitImageGenerator:
    def __init__(self):
        self.progress_tracker = None
        self.results_container = None
        self.face_swap_processor = None
        self.expression_filter = ExpressionFilter()
        self.max_workers = 10
        
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
            
            # Use Replicate's GPT-5 model - keeping expression in prompt for analysis
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
            
            # Filter out expressions using ExpressionFilter
            filtered_prompt = self.expression_filter.filter_expressions(result.strip())
            logger.info(f"Expression-filtered prompt: {filtered_prompt[:100]}...")
            
            return filtered_prompt
            
        except Exception as e:
            logger.error(f"Error getting GPT-5 prompt for {image_path}: {str(e)}")
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
            
            logger.error(f"Seedream API Error: {response.status_code} - {response.text}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
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
    
    def process_single_image(self, image_info, selfie_path, replicate_api_key, ark_api_key, enable_face_swap):
        """Process a single image - designed for parallel execution"""
        image_path, image_name = image_info
        item_id = f"img_{threading.current_thread().ident}_{image_name}"
        
        try:
            # Update progress
            self.progress_tracker.update_stage(item_id, "Getting prompt")
            logger.info(f"Worker processing: {image_name}")
            
            # Step 1: Get ChatGPT prompt (with expression filtering)
            prompt = self.get_chatgpt_prompt(str(image_path), replicate_api_key)
            if not prompt:
                logger.error(f"Failed to get prompt for {image_name}")
                self.progress_tracker.complete_item(item_id, success=False)
                return None
            
            # Step 2: Generate image with Seedream
            self.progress_tracker.update_stage(item_id, "Generating image")
            image_url = self.generate_seedream_image(prompt, selfie_path, ark_api_key)
            if not image_url:
                logger.error(f"Failed to generate image for {image_name}")
                self.progress_tracker.complete_item(item_id, success=False)
                return None
            
            # Step 3: Download generated image
            self.progress_tracker.update_stage(item_id, "Downloading")
            image_content = self.download_image(image_url)
            if not image_content:
                logger.error(f"Failed to download generated image for {image_name}")
                self.progress_tracker.complete_item(item_id, success=False)
                return None
            
            # Step 4: Face swap (if enabled)
            final_image_content = image_content
            final_filename = self.generate_random_filename() + ".jpg"
            
            if enable_face_swap and self.face_swap_processor:
                self.progress_tracker.update_stage(item_id, "Face swapping")
                
                # Save generated image temporarily for face swap
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_generated:
                    tmp_generated.write(image_content)
                    generated_image_path = tmp_generated.name
                
                # Perform face swap
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_swapped:
                    swapped_image_path = tmp_swapped.name
                
                face_swap_success = self.face_swap_processor.process_face_swap(
                    generated_image_path, selfie_path, swapped_image_path
                )
                
                if face_swap_success and os.path.exists(swapped_image_path):
                    with open(swapped_image_path, 'rb') as f:
                        final_image_content = f.read()
                    final_filename = self.generate_random_filename() + "_swapped.jpg"
                    logger.info(f"Face swap completed successfully for {image_name}")
                else:
                    logger.warning(f"Face swap failed for {image_name}, using original generated image")
                
                # Clean up temporary files
                try:
                    os.unlink(generated_image_path)
                    if os.path.exists(swapped_image_path):
                        os.unlink(swapped_image_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary files: {e}")
            
            # Create result
            result = {
                'original_image': str(image_path),
                'original_name': image_name,
                'generated_filename': final_filename,
                'prompt': prompt,
                'image_content': final_image_content,
                'image_url': image_url,
                'face_swapped': enable_face_swap
            }
            
            logger.info(f"Successfully processed: {image_name}")
            self.progress_tracker.complete_item(item_id, success=True)
            
            # Add small delay to respect rate limits
            time.sleep(1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}")
            self.progress_tracker.complete_item(item_id, success=False)
            return None
    
    def process_images(self, input_folder, selfie_file, replicate_api_key, ark_api_key, enable_face_swap=False, comfyui_server_url=None):
        """Process all images using parallel workers"""
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
                image_files.append((file_path, file_path.name))
        
        logger.info(f"Found {len(image_files)} supported images to process")
        
        if not image_files:
            logger.warning("No supported image files found in the input folder")
            st.warning("No supported image files found in the input folder.")
            return [], []
        
        # Save uploaded selfie temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_selfie:
            tmp_selfie.write(selfie_file.getvalue())
            selfie_path = tmp_selfie.name
        
        # Initialize progress tracking
        self.progress_tracker = ProgressTracker(len(image_files))
        progress_bar = st.progress(0)
        status_text = st.empty()
        self.progress_tracker.set_ui_elements(progress_bar, status_text)
        
        self.results_container = st.container()
        
        # Process images in parallel
        successful_results = []
        
        try:
            logger.info(f"Starting parallel processing with {self.max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(
                        self.process_single_image, 
                        image_info, 
                        selfie_path, 
                        replicate_api_key, 
                        ark_api_key, 
                        enable_face_swap
                    ): image_info
                    for image_info in image_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_image):
                    image_info = future_to_image[future]
                    image_path, image_name = image_info
                    
                    try:
                        result = future.result()
                        if result:
                            successful_results.append(result)
                            
                            # Add to CSV data
                            csv_data.append({
                                'original_image': image_name,
                                'generated_filename': result['generated_filename'],
                                'prompt': result['prompt'],
                                'face_swapped': enable_face_swap,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            # Show result in real-time
                            with self.results_container:
                                with st.expander(f"✅ {image_name} → {result['generated_filename']}", expanded=False):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("Original")
                                        original_img = Image.open(image_path)
                                        st.image(original_img, use_container_width=True)
                                    
                                    with col2:
                                        title_suffix = " (Face Swapped)" if enable_face_swap else ""
                                        st.subheader(f"Generated{title_suffix}")
                                        generated_img = Image.open(io.BytesIO(result['image_content']))
                                        st.image(generated_img, use_container_width=True)
                                        
                                    st.text(f"Prompt: {result['prompt']}")
                        else:
                            logger.warning(f"Failed to process: {image_name}")
                            with self.results_container:
                                st.error(f"❌ Failed to process: {image_name}")
                            
                    except Exception as e:
                        logger.error(f"Error getting result for {image_name}: {e}")
                        with self.results_container:
                            st.error(f"❌ Error processing {image_name}: {e}")
        
        finally:
            # Clean up temporary selfie file
            try:
                os.unlink(selfie_path)
            except:
                pass
        
        status_text.text(f"Processing complete! Successfully processed {len(successful_results)}/{len(image_files)} images")
        return successful_results, csv_data

def main():
    st.title("🎭 Seedream + Face Swap Generator (Parallel Processing)")
    st.markdown("Generate Seedream images using ChatGPT prompts with expression filtering and parallel processing")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Processing configuration
    st.sidebar.header("Processing Settings")
    st.sidebar.info("🚀 Using 10 parallel workers for faster processing")
    st.sidebar.info("🎭 Expression filtering: GPT-5 analyzes expressions but they're removed from Seedream prompts")
    
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
        st.header("🔄 Parallel Processing Features")
        
        features = [
            "**10 Parallel Workers** - Process multiple images simultaneously",
            "**Expression Filtering** - GPT-5 analyzes expressions but removes them from Seedream prompts",
            "**Real-time Progress** - Live updates showing worker status and completion",
            "**Thread-safe Processing** - Robust parallel execution with proper error handling",
            "**Rate Limit Management** - Built-in delays to respect API limits",
            "**Memory Efficient** - Session pooling and proper resource cleanup"
        ]
        
        for feature in features:
            st.markdown(f"• {feature}")
        
        st.header("📋 Processing Workflow")
        
        workflow_steps = [
            "**Upload images and selfie**",
            "**Configure parallel processing settings**",
            "**Click 'Start Parallel Processing'**"
        ]
        
        if enable_face_swap:
            workflow_description = """
            **Parallel Processing Pipeline (per worker):**
            1. 🧠 GPT-5 analyzes image and generates detailed prompt (including expressions)
            2. 🎭 Expression filter removes facial expressions from the prompt
            3. 🎨 Seedream generates new image using filtered prompt + your selfie
            4. 🔄 **Face swap replaces faces** in generated image with your selfie
            5. 📁 Random 32-character filename generation
            
            **Note**: Face swapping adds processing time but provides more realistic results.
            """
        else:
            workflow_description = """
            **Parallel Processing Pipeline (per worker):**
            1. 🧠 GPT-5 analyzes image and generates detailed prompt (including expressions)
            2. 🎭 Expression filter removes facial expressions from the prompt
            3. 🎨 Seedream generates new image using filtered prompt + your selfie
            4. 📁 Random 32-character filename generation
            """
        
        for i, step in enumerate(workflow_steps, 1):
            st.markdown(f"{i}. {step}")
        
        st.markdown(workflow_description)
    
    with col2:
        st.header("⚡ Status Dashboard")
        
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
        checks.append(("🔑 API Keys", "✅" if api_keys_configured else "❌ Update in script"))
        
        if input_method == "Upload individual images":
            checks.append(("📁 Images", f"✅ {image_info}" if has_images else "❌"))
        else:
            checks.append(("📦 ZIP File", f"✅ {image_info}" if has_images else "❌"))
            
        checks.append(("🤳 Selfie", "✅" if has_selfie else "❌"))
        
        if enable_face_swap:
            checks.append(("🔄 ComfyUI Server", "✅" if COMFYUI_SERVER_URL else "❌"))
        
        for check_name, status in checks:
            st.text(f"{check_name}: {status}")
        
        # Performance info
        st.subheader("⚡ Performance")
        st.text("🔧 Workers: 10 parallel")
        st.text("🎭 Expression Filtering: ON")
        st.text("🧵 Thread-safe: YES")
        
        if enable_face_swap:
            st.info("🎭 Face swap enabled - realistic results with longer processing time")
        
        if not api_keys_configured:
            st.warning("⚠️ Please update API keys in the script before running")
    
    # Processing section
    st.header("🚀 Parallel Processing")
    
    required_items = [api_keys_configured, has_images, has_selfie]
    
    if st.button("🚀 Start Parallel Processing", type="primary", disabled=not all(required_items)):
        if not all(required_items):
            st.error("Please provide all required inputs before starting.")
            return
        
        # Create temporary directory for uploaded images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Handle different input methods
            if input_method == "Upload individual images":
                # Save uploaded images to temp directory
                for uploaded_file in uploaded_images:
                    file_path = Path(temp_dir) / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                total_input = len(uploaded_images)
            else:
                # Extract ZIP file
                generator = StreamlitImageGenerator()
                extracted_files = generator.extract_images_from_zip(uploaded_zip.getvalue(), temp_dir)
                total_input = len(extracted_files)
                
                if not extracted_files:
                    st.error("No valid images found in the ZIP file.")
                    return
            
            # Process images with parallel workers
            generator = StreamlitImageGenerator()
            results, csv_data = generator.process_images(
                temp_dir, 
                selfie_file, 
                REPLICATE_API_KEY, 
                ARK_API_KEY,
                enable_face_swap,
                comfyui_server_url if enable_face_swap else None
            )
            
            if results:
                success_message = f"🎉 Successfully processed {len(results)}/{total_input} images using parallel processing!"
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
                    csv_header = f"# Seedream Parallel Image Generation Report\n"
                    csv_header += f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    csv_header += f"# Total images processed: {len(results)}/{total_input}\n"
                    csv_header += f"# Face swap enabled: {enable_face_swap}\n"
                    csv_header += f"# Input method: {input_method}\n"
                    csv_header += f"# Parallel workers: {generator.max_workers}\n"
                    csv_header += f"# Expression filtering: Enabled\n"
                    csv_header += "#\n"
                    
                    csv_content = csv_header + csv_df.to_csv(index=False)
                    
                    # Download CSV button
                    st.download_button(
                        label="📊 Download Processing Report",
                        data=csv_content,
                        file_name=f"seedream_parallel_report_{timestamp}.csv",
                        mime="text/csv",
                        help="Download detailed report with prompts and metadata"
                    )
                
                # Enhanced ZIP with organized structure
                st.subheader("📦 Generated Images Package")
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
                    summary = f"Seedream Parallel Image Generation Summary\n"
                    summary += f"==========================================\n\n"
                    summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    summary += f"Input method: {input_method}\n"
                    summary += f"Total images processed: {len(results)}/{total_input}\n"
                    summary += f"Success rate: {(len(results)/total_input*100):.1f}%\n"
                    summary += f"Face swap enabled: {enable_face_swap}\n"
                    summary += f"Parallel workers: {generator.max_workers}\n"
                    summary += f"Expression filtering: Enabled (expressions analyzed but removed from Seedream prompts)\n\n"
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
                - **Processing Mode**: Parallel (10 workers)
                - **Expression Filtering**: Enabled (analyzed but removed from Seedream)
                - **Face Swapping**: {'Enabled' if enable_face_swap else 'Disabled'}
                - **Processing Time**: Completed at {datetime.now().strftime('%H:%M:%S')}
                - **Output Format**: Random 32-character filenames
                - **APIs Used**: Replicate GPT-5 for prompts, Seedream for generation
                """
                
                if enable_face_swap:
                    process_info += f"\n- **ComfyUI Server**: {COMFYUI_SERVER_URL}"
                
                st.markdown(process_info)
            
            else:
                st.error("No images were successfully processed. Please check your API keys and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with Streamlit, Replicate (GPT-5), Seedream, and ComfyUI APIs | **Parallel Processing with Expression Filtering**")
    
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
