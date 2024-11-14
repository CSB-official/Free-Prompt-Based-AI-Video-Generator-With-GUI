#!/usr/bin/env python3
"""
AI Video Generator
A tool for generating videos from text prompts using Stable Diffusion.
Supports AMD GPUs through DirectML acceleration.
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
import imageio
import os
from PIL import Image
import cv2
from tqdm.auto import tqdm
from pathlib import Path
import subprocess
import gradio as gr
from datetime import datetime
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_requirements():
    """
    Install required Python packages for the video generator.
    Handles errors gracefully and provides feedback on installation status.
    """
    
    env_dir = Path("venv")
    requirements = [
        "diffusers==0.31.0",
        "fastapi==0.115.5",
        "gradio==5.5.0",
        "gradio-client==1.4.2",
        "safehttpx==0.1.1",
        "tokenizers==0.20.3",
        "torch-directml==0.2.5.dev240914",
        "torchvision==0.19.1",
        "transformers==4.46.2"
    ]

    # Create virtual environment if it doesn't exist
    if not env_dir.exists():
        logger.info("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)

    pip_executable = env_dir / "bin" / "pip" if os.name != "nt" else env_dir / "Scripts" / "pip.exe"

    # Upgrade pip only if required
    try:
        result = subprocess.run(
            [str(pip_executable), "--version"],
            capture_output=True, text=True, check=True
        )
        installed_version = result.stdout.split()[1]
        if installed_version < "24.3.1":
            logger.info("Upgrading pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        else:
            logger.info("Pip is up to date.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to upgrade pip: {str(e)}. Continuing with existing pip version.")

    # Install each package in the virtual environment
    for package in requirements:
        try:
            subprocess.run([str(pip_executable), "install", package], check=True)
            logger.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error installing {package}: {str(e)}")

class VideoGenerator:
    """
    Main video generation class handling model loading and frame generation.
    Uses Stable Diffusion for image generation and supports AMD GPUs via DirectML.
    """
    
    def __init__(self, cache_dir="./model_cache"):
        """
        Initialize the video generator with model and cache settings.
        
        Args:
            cache_dir (str): Directory to store model cache
        """
        # Setup cache directories
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for model caching
        os.environ['HF_HOME'] = str(self.cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir / "transformers")
        os.environ['DIFFUSERS_CACHE'] = str(self.cache_dir / "diffusers")
        
        logger.info(f"Using cache directory: {self.cache_dir}")

        # Initialize DirectML for AMD GPU support
        self._setup_device()
        
        # Load the Stable Diffusion model
        self._initialize_model()

    def _setup_device(self):
        """Set up DirectML device for AMD GPU support"""
        try:
            import torch_directml
            dml = torch_directml.device()
            self.device = dml
            logger.info("Using DirectML for AMD GPU acceleration")
        except ImportError:
            logger.warning("DirectML not found. Installing required packages...")
            install_requirements()
            import torch_directml
            dml = torch_directml.device()
            self.device = dml
            logger.info("Successfully initialized DirectML")

    def _initialize_model(self):
        """Initialize and load the new Stable Diffusion model"""
        model_id = "stabilityai/stable-diffusion-2-1"  # New model ID
        local_model_path = self.cache_dir / "stable-diffusion-2-1"

        try:
            if self.is_model_cached(local_model_path):
                logger.info("Loading model from local cache...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16,  # Adjust if the new model benefits from it
                    local_files_only=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                logger.info("Downloading model (this will be cached)...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe.save_pretrained(local_model_path)
            
            # Reconfigure scheduler if needed
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.to(self.device)
            logger.info("New model pipeline loaded successfully!")

        except Exception as e:
            logger.error(f"Error initializing new model: {str(e)}")
            raise

    def is_model_cached(self, path: Path) -> bool:
        """
        Check if model files are already cached locally.
        
        Args:
            path (Path): Path to check for model files
            
        Returns:
            bool: True if all required files exist
        """
        if not path.exists():
            return False
        required_files = ['model_index.json', 'scheduler', 'tokenizer', 'unet', 'vae']
        return all((path / file).exists() for file in required_files)

    def generate_frame(self, prompt: str, step: int, num_frames: int, seed: int = None) -> tuple:
        """
        Generate a single frame using the Stable Diffusion model.
        
        Args:
            prompt (str): Text description for image generation
            step (int): Current frame number
            num_frames (int): Total number of frames
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            tuple: (PIL.Image, numpy.ndarray) of the generated frame
        """
        if seed is not None:
            torch.manual_seed(seed + step)
            
        frame_prompt = f"{prompt} . Frame {step + 1} of {num_frames} sequence"
        
        try:
            with torch.no_grad():
                image = self.pipe(
                    frame_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
            
            return image, np.array(image)
            
        except Exception as e:
            logger.error(f"Error generating frame {step + 1}: {str(e)}")
            raise

class VideoGenerationUI:
    """
    Gradio-based user interface for the video generator.
    Provides controls for video generation parameters and live preview.
    """
    
    def __init__(self):
        """Initialize the UI components and video generator"""
        self.generator = VideoGenerator()
        self.output_dir = Path("./generated_videos")
        self.output_dir.mkdir(exist_ok=True)
        self.frames = []

    def generate_video(
        self,
        prompt: str,
        num_frames: int,
        width: int,
        height: int,
        seed: int,
        progress=gr.Progress(track_tqdm=False)
    ) -> tuple:
        """
        Generate video frames with live preview updates.
        
        Args:
            prompt (str): Text description for video
            num_frames (int): Number of frames to generate
            width (int): Frame width
            height (int): Frame height
            seed (int): Random seed
            progress: Gradio progress tracker
            
        Yields:
            tuple: (preview_gallery, video_path, status_text)
        """
        try:
            self.frames = []
            preview_gallery = []
            status_text = "Initializing generation..."
            
            yield preview_gallery, None, status_text
            
            # Generate individual frames
            for i in range(num_frames):
                status_text = f"Generating frame {i + 1}/{num_frames}..."
                yield preview_gallery, None, status_text
                
                pil_image, np_image = self.generator.generate_frame(
                    prompt, i, num_frames, seed
                )
                
                if np_image.shape[0] != height or np_image.shape[1] != width:
                    np_image = cv2.resize(np_image, (width, height))
                
                self.frames.append(np_image)
                preview_gallery.append(pil_image)
                
                if hasattr(torch, 'directml'):
                    torch.directml.empty_cache()
                
                yield preview_gallery, None, f"Completed frame {i + 1}/{num_frames}"
            
            # Process transitions and save video
            status_text = "Processing transitions..."
            yield preview_gallery, None, status_text
            
            video_path = self._save_video_with_transitions(width, height)
            
            status_text = "Video generation complete!"
            yield preview_gallery, video_path, status_text
            
        except Exception as e:
            logger.error(f"Error in video generation: {str(e)}")
            yield [], None, f"Error: {str(e)}"

    def _save_video_with_transitions(self, width: int, height: int) -> str:
        """
        Save the generated frames as a video file with transitions.
        
        Args:
            width (int): Frame width
            height (int): Frame height
            
        Returns:
            str: Path to saved video file
        """
        frames_with_transitions = []
        
        # Add transitions between frames
        for i in range(len(self.frames)):
            frames_with_transitions.append(self.frames[i])
            if i < len(self.frames) - 1:
                transitions = create_frame_transition(
                    self.frames[i],
                    self.frames[i + 1]
                )
                frames_with_transitions.extend(transitions)
        
        # Prepare video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(self.output_dir / f"generated_video_{timestamp}.mp4")
        
        video_array = np.array(frames_with_transitions)
        if video_array.dtype != np.uint8:
            video_array = (video_array * 255).astype(np.uint8)
        
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            12,
            (width, height)
        )
        
        # Write frames
        for frame in video_array:
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        
        writer.release()
        return output_path

    def create_ui(self) -> gr.Blocks:
        """
        Create the Gradio interface with controls and preview.
        
        Returns:
            gr.Blocks: Configured Gradio interface
        """
        with gr.Blocks(title="AI Video Generator") as interface:
            gr.Markdown("# AI Video Generator")
            gr.Markdown("Generate videos from text descriptions using AI")
            
            with gr.Row():
                # Left column: Controls
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe what you want to see in the video...",
                        value="A majestic eagle soaring through a sunset sky"
                    )
                    
                    with gr.Row():
                        num_frames = gr.Slider(
                            minimum=2,
                            maximum=8,
                            value=4,
                            step=1,
                            label="Number of Frames"
                        )
                        seed = gr.Slider(
                            minimum=0,
                            maximum=1000000,
                            value=42,
                            step=1,
                            label="Random Seed"
                        )
                    
                    with gr.Row():
                        width = gr.Slider(
                            minimum=128,
                            maximum=512,
                            value=256,
                            step=64,
                            label="Width"
                        )
                        height = gr.Slider(
                            minimum=128,
                            maximum=512,
                            value=256,
                            step=64,
                            label="Height"
                        )
                    
                    generate_btn = gr.Button("Generate Video", variant="primary")
                    status = gr.Textbox(label="Status")
                
                # Right column: Preview and output
                with gr.Column(scale=2):
                    preview_gallery = gr.Gallery(
                        label="Frame Previews",
                        show_label=True,
                        elem_id="preview_gallery",
                        columns=[2, 2],
                        rows=[2, 2],
                        height=500,
                        object_fit="contain"
                    )
                    video_output = gr.Video(label="Generated Video")
            
            # Connect the generation function
            generate_btn.click(
                fn=self.generate_video,
                inputs=[prompt, num_frames, width, height, seed],
                outputs=[preview_gallery, video_output, status],
                show_progress="hidden"
            )
            
        return interface

def create_frame_transition(frame1: np.ndarray, frame2: np.ndarray, num_transition_frames: int = 2) -> list:
    """
    Create smooth transitions between two frames.
    
    Args:
        frame1 (np.ndarray): First frame
        frame2 (np.ndarray): Second frame
        num_transition_frames (int): Number of transition frames to generate
        
    Returns:
        list: List of transition frames
    """
    transitions = []
    for i in range(num_transition_frames):
        alpha = (i + 1) / (num_transition_frames + 1)
        transition = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        transitions.append(transition)
    return transitions

def main():
    """Main entry point for the application"""
    logger.info("Initializing video generator interface...")
    install_requirements()
    
    try:
        ui = VideoGenerationUI()
        interface = ui.create_ui()
        interface.launch(share=False)
    except Exception as e:
        logger.error(f"Error launching interface: {str(e)}")
        raise

if __name__ == "__main__":
    main()
