# 🎥🤖 AI Video Generator 🌈💫

The AI Video Generator is an advanced Python application that leverages state-of-the-art machine learning techniques to generate dynamic videos from textual descriptions. By harnessing the power of the Stable Diffusion model and integrating DirectML for AMD GPU acceleration, this tool pushes the boundaries of AI-driven content creation.

## 📋 Table of Contents
- [✨ Features](#features)
- [💾 Installation](#installation)
- [🚀 Usage](#usage) 
- [🔧 Technical Overview](#technical-overview)
  - [🖼️ Stable Diffusion Model](#stable-diffusion-model)
  - [💪 DirectML Integration](#directml-integration)
  - [🎞️ Frame Generation](#frame-generation)
  - [📹 Video Processing](#video-processing)
- [🌟 User Interface](#user-interface)
- [🔮 Future Enhancements](#future-enhancements)
- [📜 License](#license)

## ✨ Features
- 🎨 Generate compelling videos from text prompts
- 🧠 Utilizes cutting-edge Stable Diffusion model for high-quality image synthesis
- 🚀 Supports AMD GPUs through DirectML acceleration for improved performance
- 🖥️ Intuitive web-based user interface built with Gradio
- 🎛️ Customizable video parameters including frame count, dimensions, and random seed
- 🔗 Smooth frame transitions for enhanced visual coherence
- ⚡ Efficient model caching and resource management

## 💾 Installation
1. Clone the repository:
   ```
   git clone https://github.com/CSB-official/Free-Prompt-Based-AI-Video-Generator-With-GUI.git
   cd Free-Prompt-Based-AI-Video-Generator-With-GUI
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   Note: The `install_requirements()` function in the code will handle installing the necessary packages automatically.

## 🚀 Usage
To launch the AI Video Generator, run the following command:
```
python ai_video_generator.py
```
This will start the Gradio interface, which you can access through your web browser at the provided URL.

## 🔧 Technical Overview
### 🖼️ Stable Diffusion Model
The AI Video Generator utilizes the Stable Diffusion v1.5 model, a state-of-the-art text-to-image generation model. Stable Diffusion is capable of producing highly detailed and coherent images from textual descriptions. The model is loaded using the `diffusers` library and can be automatically downloaded and cached locally for efficient usage.

### 💪 DirectML Integration
To enable AMD GPU acceleration, the code integrates DirectML, a low-level API for machine learning on DirectX 12 compatible hardware. By leveraging DirectML, the video generator can significantly speed up the image generation process on supported AMD GPUs. The `torch_directml` package is used to initialize and utilize the DirectML device.

### 🎞️ Frame Generation
The core of the video generation process lies in the `generate_frame()` method of the `VideoGenerator` class. This method takes a text prompt, frame number, total frame count, and an optional random seed as input. It then generates a single frame using the Stable Diffusion model by constructing a frame-specific prompt and running the model inference with the specified parameters.

### 📹 Video Processing
The generated frames are processed and combined into a video using OpenCV. The `create_frame_transition()` function creates smooth transitions between consecutive frames by interpolating the pixel values. The resulting frames, along with the transitions, are then written to an MP4 video file using OpenCV's `VideoWriter` class.

## 🌟 User Interface
The AI Video Generator provides an intuitive web-based user interface built with Gradio. The interface allows users to input a text prompt, specify the desired number of frames, video dimensions, and a random seed. It also includes a live preview gallery that displays the generated frames in real-time during the video creation process. Once the video is generated, it can be viewed and downloaded directly from the interface.

## 🔮 Future Enhancements
- 📼 Support for additional video formats and codecs
- 🎥 Integration of more advanced frame interpolation techniques for smoother transitions
- 🔊 Option to generate videos with audio based on text prompts
- 🎬 Implementation of real-time preview and interactive frame adjustment
- 🌈 Exploration of other state-of-the-art text-to-image models for improved quality and diversity
