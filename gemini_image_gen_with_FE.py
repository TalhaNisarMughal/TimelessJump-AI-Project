import streamlit as st
import os
import base64
from pathlib import Path
from openai import OpenAI
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

def refine_prompt(user_prompt):
    """
    Uses GPT-4o-mini API to refine the user's prompt into a professional product photography prompt
    """
    example_prompt = """Based on the uploaded image of the TIMELESS JUMP yellow speed rope, create a high-definition studio product photograph. The background must be pure seamless white (RGB 255, 255, 255). The rope should be neatly coiled to show the full length and yellow colour of the cable, with the elongated handle featuring the 'TIMELESS JUMP' inscription clearly visible and in focus. Use soft, diffused, even lighting (like a professional softbox) to eliminate harsh shadows and reflections, showcasing the smooth, lustrous PVC surface and ergonomic handle texture. The image should be perfectly centred with a square 1:1 aspect ratio and ultra-sharp detail. Please pay attention to specific details like handle and rope and also handle texture and shape etc"""
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": f"""You are a professional product photography prompt engineer specializing in Gemini image generation. Your task is to transform user prompts into highly detailed, narrative-driven prompts optimized for Gemini's image generation API.

CORE PRINCIPLE: Describe the scene in natural language, don't just list keywords. Gemini excels at understanding descriptive paragraphs over disconnected keywords.

PROMPT STRUCTURE FOR PRODUCT PHOTOGRAPHY:
A [shot type] of [product description with specific details], [action/positioning], set in [environment/background]. The scene is illuminated by [lighting description], creating a [mood] atmosphere. Captured with [camera/lens details], emphasizing [key features and textures]. [Specific constraints like aspect ratio, text placement, focus areas].

BEST PRACTICES FROM GEMINI DOCUMENTATION:
1. Photography terminology: Use specific camera angles (wide-angle, macro, 45-degree elevated shot), lens types (85mm portrait lens), lighting setups (three-point softbox, golden hour, diffused natural light)
2. Material & texture specificity: Describe surfaces precisely - "smooth lustrous PVC surface", "ergonomic textured grip", "matte finish"
3. Background precision: Be explicit - "pure seamless white background (RGB 255, 255, 255)", "polished concrete surface", "dark gradient backdrop"
4. Composition details: Specify positioning - "perfectly centered", "bottom-right placement", "coiled neatly to show full length"
5. Text rendering: For any text/branding - "inscription clearly visible and in focus", specify font style if needed
6. Mood and atmosphere: Include emotional tone - "professional and clean", "dynamic and energetic", "premium and luxurious"

EXAMPLE OF PERFECT PROMPT:
"{example_prompt}"

USER'S REQUEST:
"{user_prompt}"

YOUR TASK: Transform the user's request into a detailed, narrative prompt following the structure and best practices above. Focus on describing the complete scene with photography-specific language.

CRITICAL: Return ONLY the refined prompt text as a single descriptive paragraph. No explanations, no preamble, no markdown formatting."""
            }],
            max_tokens=1000
        )
        
        refined = response.choices[0].message.content.strip()
        return refined
    
    except Exception as e:
        return user_prompt


def generate_image(user_prompt, image_paths):
    """
    Main pipeline: refines prompt and generates image using Gemini 3 Pro Image
    
    Args:
        user_prompt: User's description of the desired image
        image_paths: List of paths to reference images (up to 10)
    """
    
    if not user_prompt or not user_prompt.strip():
        return None
    
    if not image_paths or len(image_paths) == 0:
        return None
    
    if len(image_paths) > 10:
        image_paths = image_paths[:10]
    
    refined_prompt = refine_prompt(user_prompt)
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            pass
    
    if not images:
        return None
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        contents = [refined_prompt] + images
        
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio="1:1",
                    image_size="2K"
                )
            )
        )
        
        output_path = f"generated_jump_rope_{int(Path().absolute().stat().st_mtime)}.png"
        
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(output_path)
                return output_path
        
        return None
    
    except Exception as e:
        return None


def get_images_from_folder(folder_path):
    """
    Get all image files from a folder
    
    Args:
        folder_path: Path to folder containing images
    
    Returns:
        List of image file paths
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
    image_paths = []
    
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    if not folder.is_dir():
        return []
    
    for file in folder.iterdir():
        if file.suffix.lower() in supported_formats:
            image_paths.append(str(file))
    
    return sorted(image_paths)[:10]


# Streamlit UI
st.title("Product Image Generator")

prompt = st.text_area("Enter your product description:", height=100)

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            folder_path = "gemini_images"
            image_paths = get_images_from_folder(folder_path)
            
            if image_paths:
                result_path = generate_image(prompt, image_paths)
                
                if result_path:
                    st.image(result_path)
                else:
                    st.error("Failed to generate image")
            else:
                st.error("No reference images found in gemini_images folder")
    else:
        st.warning("Please enter a description")