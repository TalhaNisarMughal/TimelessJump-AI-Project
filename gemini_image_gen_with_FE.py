import streamlit as st
import os
import base64
import uuid
import concurrent.futures
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import logging
from google.genai import types
import random

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# We only need the Gemini Key now
GEMINI_API_KEY = os.getenv("GEMINI_KEY")

def refine_prompt(user_prompt, variation_seed=None):
    """
    Refine user prompts for Gemini 3 Pro Image generation.
    Optimized for product variant generation WITHOUT any text or logos.
    
    Args:
        user_prompt: User's description of the desired product variant
        variation_seed: Optional seed to create prompt variations
    """
    
    reference_images_folder = "gemini_images"
    
    # Add variation elements based on seed
    variation_elements = [
        "with subtle lighting variation",
        "from a slightly different perspective",
        "with alternative composition",
        "with varied depth of field emphasis",
        "with different shadow placement",
        "with adjusted product positioning"
    ]
    
    variation_hint = ""
    if variation_seed is not None:
        random.seed(variation_seed)
        variation_hint = f"\n\nIMPORTANT: Create a unique variation {random.choice(variation_elements)}. Ensure this image is distinct from other generations."
    
    example_prompt = """Based on the uploaded product reference images, create a high-definition studio product photograph of the jump rope. The background must be pure seamless white (RGB 255, 255, 255). The rope should be neatly coiled to show the full length and yellow colour of the cable, with the handle clearly visible and in focus. Use soft, diffused, even lighting (like a professional softbox) to eliminate harsh shadows and reflections, showcasing the smooth, lustrous PVC surface and ergonomic handle texture. The image should be perfectly centred with a square 1:1 aspect ratio and ultra-sharp detail. Focus on the handle shape, grip texture, rope material, and cable design exactly as shown in the reference images. CRITICAL: Generate a completely clean product with NO text, NO logos, NO inscriptions, NO branding marks anywhere on the handle, rope, or any part of the product."""
    
    system_instruction = f"""You are a professional image editing prompt engineer specializing in Gemini 3 Pro Image. Your task is to create EDITING instructions that modify the uploaded reference product images based on the user's requested changes.

###CRITICAL CONTEXT###
- The user has uploaded 7+ REFERENCE IMAGES of their actual product (a jump rope)
- These are REAL product photos showing the exact design, shape, materials, and construction
- Your job is to create an EDITING prompt that MODIFIES these reference images, NOT generate from scratch
- Gemini 3 Pro Image excels at "conversational editing" - understanding reference images and applying requested changes

###REFERENCE-BASED EDITING APPROACH###
The prompt structure should be:
1. Acknowledge the uploaded reference images explicitly
2. Instruct the model to STUDY the reference images to understand the product
3. Request SPECIFIC EDITS/CHANGES while maintaining product consistency
4. Use natural, conversational language (Gemini 3 Pro understands context)

###BEST PRACTICES FROM GEMINI 3 PRO IMAGE DOCUMENTATION###

1. **Start with Reference Context**: 
   - "Based on the uploaded reference images of the jump rope..."
   - "Study the reference product images carefully and..."
   - "Using the provided reference images as the base..."

2. **Conversational Editing Language**:
   - Use phrases like "change the color to...", "make it...", "transform the handles to..."
   - Be direct: "Change the rope and handles to [color/material]"
   - Natural language works best: "Keep the exact shape and design but render it in black"

3. **Maintain Product Consistency**:
   - "Maintain the exact handle shape, grip texture, and rope dimensions from the references"
   - "Keep all physical characteristics identical - only modify the [color/material/finish]"
   - "Preserve the ergonomic design and construction details"

4. **Photography Instructions** (for variation):
   - Specify camera angle: "from a 45-degree elevated angle", "straight-on view", "top-down"
   - Define lighting: "soft diffused studio lighting", "three-point softbox", "natural window light"
   - Background: "pure seamless white background (RGB 255, 255, 255)"

5. **NO TEXT/BRANDING** (Critical):
   - "CRITICAL: Remove ALL text, logos, inscriptions, and branding marks from the product"
   - "Generate a clean version with NO text anywhere on handles or rope"
   - "The product must be completely unmarked - no letters, words, or symbols"

6. **Material & Finish Specifications**:
   - Be specific: "matte black finish", "glossy metallic silver", "brushed aluminum texture"
   - Describe surface: "smooth lustrous coating", "textured grip zones", "reflective surface"

###EXAMPLE OF PERFECT EDITING PROMPT###
"Based on the uploaded reference images, take a close look at the jump rope product to understand its exact handle shape, grip texture, rope design, and overall construction. Now, create a professional studio product photograph where you change the entire product to a sleek matte black color - both the handles and the rope cable should be rendered in uniform black while maintaining every physical detail from the references. The rope should be neatly coiled showing its full length, positioned on a pure seamless white background (RGB 255, 255, 255). Shot from a 45-degree elevated angle with soft, diffused studio lighting using a three-point softbox setup to eliminate harsh shadows. Captured with an 85mm lens at f/5.6 for tack-sharp detail. CRITICAL: Remove ALL text, logos, and branding marks - the product must be completely clean with no inscriptions anywhere."

###USER'S REQUEST###
"{user_prompt}"
{variation_hint}

###YOUR TASK###
Transform the user's request into a conversational EDITING instruction that:
1. References the uploaded images as the BASE/SOURCE
2. Specifies the exact CHANGES to make (color, material, finish)
3. Maintains all PHYSICAL CHARACTERISTICS from references
4. Adds professional photography specifications
5. Includes variation elements if provided
6. Emphasizes NO TEXT/LOGOS removal
7. Uses natural, conversational language

###OUTPUT REQUIREMENTS###
- Return ONLY the editing prompt as a single descriptive paragraph
- No explanations, preamble, or markdown
- Start by acknowledging reference images
- Use conversational editing language ("change to...", "make it...", "transform...")
- Include photography angle and lighting for variation
- End with strong NO TEXT/LOGOS statement"""

    try:
        import os
        import glob
        
        # Prepare content list with system instruction
        contents = [system_instruction]
        
        # Load reference images from folder if provided
        if reference_images_folder and os.path.isdir(reference_images_folder):
            import PIL.Image
            
            # Get all image files from folder
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(reference_images_folder, ext)))
            
            # Sort and limit to 14 images (Gemini 3 Pro max)
            image_files = sorted(image_files)[:14]
            
            if image_files:
                logger.info(f"Loading {len(image_files)} reference images from folder...")
                for idx, img_path in enumerate(image_files):
                    try:
                        img = PIL.Image.open(img_path)
                        contents.append(img)
                        logger.info(f"✓ Loaded: {os.path.basename(img_path)}")
                    except Exception as e:
                        logger.error(f"✗ Error loading {os.path.basename(img_path)}: {e}")
            else:
                logger.warning(f"⚠ No images found in folder: {reference_images_folder}")
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Use Gemini 2.5 Flash with HIGHER temperature for more variation
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=1  # INCREASED from 0.8 for more creative variation
            )
        )
        
        refined = response.text.strip()
        
        # Verify the refined prompt contains NO TEXT instructions
        if "no text" not in refined.lower() and "no logo" not in refined.lower():
            logger.warning("⚠ WARNING: Refined prompt may not contain strong NO TEXT/LOGO instructions!")
        
        logger.info(f"✓ Refined prompt generated (seed: {variation_seed})")
        
        return refined
    
    except Exception as e:
        logger.error(f"Error in prompt refinement: {e}")
        return user_prompt

def generate_image(user_prompt, image_paths, variation_seed=None):
    """
    Main pipeline: refines prompt and generates image using Gemini 3 Pro Image
    Each call gets its OWN refined prompt for variety
    
    Args:
        user_prompt: User's description of the desired image
        image_paths: List of paths to reference images (up to 10)
        variation_seed: Seed for creating unique variations
    """
    
    if not user_prompt or not user_prompt.strip():
        logger.error("Empty user prompt provided")
        return None
    
    if not image_paths or len(image_paths) == 0:
        logger.error("No image paths provided")
        return None
    
    if len(image_paths) > 10:
        image_paths = image_paths[:10]
    
    # IMPORTANT: Each thread refines its own prompt with unique seed for variation
    refined_prompt = refine_prompt(user_prompt, variation_seed=variation_seed)
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            pass
    
    if not images:
        logger.error("No images could be loaded")
        return None
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        contents = [refined_prompt] + images
        
        logger.info(f"Generating image with seed {variation_seed}...")
        logger.info(f"Refined prompt preview: {contents}...")
        
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size="1K"
                )
            )
        )
        
        # Use UUID for unique filenames
        output_path = f"generated_jump_rope_{uuid.uuid4().hex[:8]}.png"
        
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(output_path)
                logger.info(f"✓ Saved image: {output_path}")
                return output_path
        
        logger.error("No image data in response")
        return None
    
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

def generate_multiple_images(user_prompt, image_paths, count=3):
    """
    Generates multiple images simultaneously using threads.
    Each thread will refine the prompt independently with unique seeds for variation.
    """
    
    if not user_prompt or not image_paths:
        logger.error("Invalid input for multiple image generation")
        return []
    
    results = []
    
    # Use ThreadPoolExecutor with unique seeds for each thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        # Each thread gets a unique seed to ensure variation in refinement
        futures = [
            executor.submit(generate_image, user_prompt, image_paths, variation_seed=i) 
            for i in range(count)
        ]
        
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.info(f"✓ Image {idx + 1}/{count} completed: {result}")
                else:
                    logger.warning(f"✗ Image {idx + 1}/{count} failed to generate")
            except Exception as e:
                logger.error(f"✗ Thread {idx + 1} raised exception: {e}")
                
    logger.info(f"Generated {len(results)}/{count} images successfully")
    return results

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




# --- Streamlit UI ---
st.set_page_config(page_title="Product AI Generator", layout="wide")
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
st.title("Product Image Generator (Gemini Powered)")

prompt = st.text_area("Enter your product description:", height=100)

folder_path = "gemini_images"
image_paths = get_images_from_folder(folder_path)

if not image_paths:
    st.error("No reference images found in 'gemini_images' folder.")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Generate Single Image"):
        if prompt and image_paths:
            with st.spinner("Refining prompt and generating single image..."):
                result_path = generate_image(prompt, image_paths)
                
                if result_path:
                    st.session_state.generated_images = [result_path]  # Store in session state
                else:
                    st.error("Failed to generate image.")
        elif not prompt:
            st.warning("Please enter a description.")
    
    # Display stored image
    if len(st.session_state.generated_images) == 1:
        st.image(st.session_state.generated_images[0], caption="Generated Image")
        with open(st.session_state.generated_images[0], "rb") as file:
            st.download_button(
                label="Download Image",
                data=file,
                file_name=st.session_state.generated_images[0],
                mime="image/png"
            )
with col2:
    if st.button("Generate Multiple (3x)"):
        if prompt and image_paths:
            with st.spinner("Generating 3 images simultaneously... this is faster with threads!"):
                result_paths = generate_multiple_images(prompt, image_paths, count=3)
                
                if result_paths:
                    st.session_state.generated_images = result_paths  # Store in session state
                else:
                    st.error("Failed to generate images.")
        elif not prompt:
            st.warning("Please enter a description.")
    
    # Display stored images
    if len(st.session_state.generated_images) == 3:
        img_cols = st.columns(3)
        
        for idx, path in enumerate(st.session_state.generated_images):
            with img_cols[idx]:
                st.image(path)
                with open(path, "rb") as file:
                    st.download_button(
                        label=f"Download #{idx+1}",
                        data=file,
                        file_name=path,
                        mime="image/png",
                        key=f"dl_btn_{idx}"
                    )