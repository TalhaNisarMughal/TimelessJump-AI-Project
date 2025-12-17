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
import random

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_KEY")

def refine_prompt(user_prompt, variation_number=None):
    """
    Refine user prompts for Gemini 3 Pro Image generation with MAXIMUM CONSISTENCY.
    Based on official Google documentation best practices.
    
    Args:
        user_prompt: User's description of the desired product variant
        variation_number: Optional number for MINIMAL variations
    """
    
    reference_images_folder = "gemini_images"
    
    # CRITICAL DOCUMENTATION FINDING: Gemini 3 prefers natural language over complex templates
    # Source: Official Gemini docs emphasize "Be descriptive, not repetitive"
    
    # Minimal variations - ONLY if absolutely necessary
    # Keep these extremely subtle to maintain consistency
    minimal_variations = [
        "",  # No variation - baseline
        "with 2% softer diffused lighting",
        "with camera position shifted 3 degrees clockwise",
        "with ambient lighting reduced by 5%",
        "with focus depth increased by 0.1 stops"
    ]
    
    variation_instruction = ""
    if variation_number is not None and variation_number > 0:
        var_idx = variation_number % len(minimal_variations)
        if minimal_variations[var_idx]:
            variation_instruction = f"\n\nMINOR ADJUSTMENT: {minimal_variations[var_idx]}. This is the ONLY change allowed."
    
    # IMPROVED: Direct, natural language system instruction
    # Based on official docs: "Gemini 3 prefers direct, efficient answers"
    # Remove XML bloat, focus on clear natural instructions
    system_instruction = f"""You are an expert prompt engineer for Gemini 3 Pro Image editing. Your role is to create prompts that maintain EXACT physical fidelity to uploaded reference images while applying user-requested changes.

CONTEXT:
The user has uploaded reference images of a jump rope product. Your task is to create an editing prompt that preserves every physical detail while applying only the requested modifications.

CRITICAL FOCUS AREAS FOR JUMP ROPE PRODUCTS:

1. HANDLE DETAILS (highest priority):
   - Exact handle shape, length, diameter, and ergonomic curves
   - Precise grip texture patterns (ridges, dimples, knurling depth)
   - Handle cap design and attachment points
   - Weight distribution and proportions
   - Surface finish characteristics (matte, glossy, textured areas)

2. TEXT AND BRANDING ON HANDLES:
   - Any text, logos, or brand names printed/embossed on handles
   - Font style, size, and placement of text
   - Text legibility and accuracy - spell exactly as shown
   - Logo colors, proportions, and positioning
   - Printing method appearance (screen print, embossed, debossed)

3. ROPE/CABLE CHARACTERISTICS:
   - Exact rope diameter and material type (PVC, steel cable, braided, etc.)
   - Cable texture and surface pattern (smooth, braided, twisted)
   - Rope coil arrangement and natural curve
   - Connection points where rope meets handles
   - Cable color, sheen, and transparency (if applicable)

USER'S REQUEST:
{user_prompt}
Everything else must remain identical to the references.{variation_instruction}

WHAT TO PRESERVE:
Everything except what the user explicitly requests to change. This includes:
- All handle dimensions and shapes
- Every text element and logo 
- Rope construction and diameter
- Grip patterns and surface textures
- Product proportions and scale
- Connection mechanisms

CRITICAL - MAKE SURE GEMINI 3 PRO IMAGE GEN FOLLOWS THIS:
-The logo and text direction should remain the same as it is inside the reference image.
-Explicitly tell the model and emphasize on reference images.
-Tell it to have a super good look at the reference images before generating.

WHAT TO CHANGE:
Only the specific attributes mentioned in the user's request, such as:
- Color changes (apply to specified parts only)
- Material finish changes (maintain same shape/texture depth)
- Surface texture modifications (keep same physical form)

PHOTOGRAPHY SPECIFICATIONS (for consistency):
- Camera: 45-degree elevated angle, 85mm lens, f/5.6 aperture
- Lighting: Three-point softbox setup (key 45° left, fill 45° right at 50%, rim from behind), 5600K color temperature
- Background: Pure seamless white (RGB 255,255,255)
- Composition: Product centered, occupying 70% of frame height
- Focus: Sharp throughout entire product, especially handles and any text

TEXT RENDERING RULES:
- If handles have text/logos, preserve spelling and layout EXACTLY
- Render all text legibly and sharply in focus
- Maintain original font characteristics unless user requests changes
- Keep logo proportions and colors accurate to reference

OUTPUT FORMAT:
Write a single, descriptive paragraph that:
1. Begins with "Based on the uploaded reference images of the jump rope, carefully study the handle design including [specific details], the rope cable construction showing [details], and any text or branding elements..."
2. States ONLY the requested changes explicitly
3. Emphasizes preservation of handles, text/logos, and rope characteristics
4. Includes complete photography specifications
5. Uses natural, descriptive language (not keyword lists)
6. Ends with "Ensure the product is completely clean with no additional text, watermarks, or labels beyond what exists in the original design."

Remember: Gemini 3 Pro Image excels at text rendering and detail preservation. Be explicit about maintaining existing text and logo elements - they are key product identifiers that must remain consistent. """

    try:
        import os
        import glob
        
        contents = [system_instruction]
        
        # Load reference images (max 14 for Gemini 3 Pro)
        if reference_images_folder and os.path.isdir(reference_images_folder):
            import PIL.Image
            
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(reference_images_folder, ext)))
            
            # Limit to 14 images (official Gemini 3 Pro limit)
            image_files = sorted(image_files)[:14]
            
            if image_files:
                logger.info(f"Loading {len(image_files)} reference images...")
                for img_path in image_files:
                    try:
                        img = PIL.Image.open(img_path)
                        contents.append(img)
                        logger.info(f"✓ Loaded: {os.path.basename(img_path)}")
                    except Exception as e:
                        logger.error(f"✗ Error loading {img_path}: {e}")
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # CRITICAL FIX: Use gemini-2.5-flash for prompt refinement
        # (Gemini 3 Pro Image is for the actual image generation, not prompt refinement)
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.5,  # Lower for more consistency
                top_p=0.85,       # Reduced for deterministic outputs
                top_k=30          # Tighter token selection
            )
        )
        
        refined = response.text.strip()
        
        # Validation checks
        if "no text" not in refined.lower() and "no logo" not in refined.lower():
            logger.warning("⚠ Adding explicit no text/logo instruction")
            refined += " Ensure the product is completely clean with no text, logos, or watermarks visible."
        
        if "reference images" not in refined.lower():
            logger.warning("⚠ Prompt missing reference to uploaded images")
        
        logger.info(f"✓ Refined prompt generated (variation: {variation_number})")
        logger.debug(f"Prompt preview: {refined[:200]}...")
        
        return refined
    
    except Exception as e:
        logger.error(f"Error in prompt refinement: {e}")
        # Fallback: return enhanced user prompt
        return f"Based on the uploaded reference images, {user_prompt}. Maintain all physical characteristics exactly as shown. Professional studio photography, clean product."


def generate_image(user_prompt, image_paths, variation_number=None, base_seed=42, resolution="1K", aspect_ratio="16:9"):
    """
    Generate image using Gemini 3 Pro Image with CONSISTENCY controls
    
    Args:
        user_prompt: User's description of the desired image
        image_paths: List of paths to reference images (up to 10)
        variation_number: Number for creating subtle variations (0, 1, 2...)
        base_seed: Base seed for reproducibility (same seed = similar results)
        resolution: Image resolution (1K, 2K, 4K)
        aspect_ratio: Aspect ratio (16:9, 1:1, etc.)
    """
    
    if not user_prompt or not user_prompt.strip():
        logger.error("Empty user prompt provided")
        return None
    
    if not image_paths or len(image_paths) == 0:
        logger.error("No image paths provided")
        return None
    
    if len(image_paths) > 10:
        image_paths = image_paths[:10]
    
    # CRITICAL: Generate refined prompt with controlled variation
    refined_prompt = refine_prompt(user_prompt, variation_number=variation_number)
    
    # Load reference images
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
        
        # IMPORTANT: Reference images MUST be included in generation contents
        contents = [refined_prompt] + images
        
        # Calculate deterministic seed for this variation
        if variation_number is not None:
            generation_seed = base_seed + variation_number
        else:
            generation_seed = base_seed
        
        logger.info(f"Generating image (variation: {variation_number}, seed: {generation_seed})...")
        logger.info(f"Refined prompt length: {refined_prompt}")
        
        # CRITICAL: Add consistency parameters
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                temperature=1.0,  # Keep at 1.0 (Google's recommendation for Gemini 3)
                # Note: Gemini API may not expose seed parameter directly
                # but thinking_level helps with consistency
                  # Use high reasoning for better accuracy
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=resolution
                )
            )
        )
        
        # Use UUID for unique filenames
        variation_suffix = f"_v{variation_number}" if variation_number is not None else ""
        output_path = f"generated_jump_rope_{uuid.uuid4().hex[:8]}{variation_suffix}.png"
        
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


def generate_multiple_images(user_prompt, image_paths, count=3, base_seed=42, resolution="1K", aspect_ratio="16:9"):
    """
    Generates multiple images with CONTROLLED variation.
    Each image uses the same base prompt with minor controlled tweaks.
    """
    
    if not user_prompt or not image_paths:
        logger.error("Invalid input for multiple image generation")
        return []
    
    results = []
    
    # IMPROVED: Use ThreadPoolExecutor with controlled variation numbers
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        # Each thread gets a unique variation number (0, 1, 2...)
        futures = [
            executor.submit(
                generate_image, 
                user_prompt, 
                image_paths, 
                variation_number=i,
                base_seed=base_seed,
                resolution=resolution,
                aspect_ratio=aspect_ratio
            ) 
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
    """Get all image files from a folder"""
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
st.set_page_config(page_title="Product Image Generator", layout="wide")

if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

if 'base_seed' not in st.session_state:
    st.session_state.base_seed = 42

st.title("🎯 Product Image Generator")
st.caption("Transform your product with AI")

# Image settings
col_settings1, col_settings2 = st.columns(2)

with col_settings1:
    resolution = st.selectbox(
        "Resolution",
        options=["1K", "2K", "4K"],
        index=0,
        help="Higher resolution = better quality but slower generation"
    )

with col_settings2:
    aspect_ratio_options = {
        "Square (1:1)": "1:1",
        "Landscape (16:9)": "16:9",
        "Portrait (9:16)": "9:16",
        "Widescreen (21:9)": "21:9",
        "Standard (4:3)": "4:3"
    }
    
    aspect_ratio_display = st.selectbox(
        "Aspect Ratio",
        options=list(aspect_ratio_options.keys()),
        index=1,
        help="Choose the shape of your output image"
    )
    aspect_ratio = aspect_ratio_options[aspect_ratio_display]

st.divider()

prompt = st.text_area(
    "Describe what you want to change:", 
    height=100,
    placeholder="Example: Change the rope and handles to matte black finish"
)

folder_path = "gemini_images"
image_paths = get_images_from_folder(folder_path)

if not image_paths:
    logger.error("⚠️ No reference images found in 'gemini_images' folder.")
else:
    logger.error(f"✅ {len(image_paths)} reference images loaded")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🎨 Generate Single Image", type="primary", use_container_width=True):
        if prompt and image_paths:
            with st.spinner("Generating image..."):
                result_path = generate_image(
                    prompt, 
                    image_paths,
                    variation_number=None,
                    base_seed=st.session_state.base_seed,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio
                )
                
                if result_path:
                    st.session_state.generated_images = [result_path]
                    st.success("✅ Image generated!")
                else:
                    st.error("❌ Failed to generate image.")
        elif not prompt:
            st.warning("⚠️ Please enter a description.")
    
    # Display stored image
    if len(st.session_state.generated_images) == 1:
        st.image(st.session_state.generated_images[0], caption="Generated Image")
        with open(st.session_state.generated_images[0], "rb") as file:
            st.download_button(
                label="📥 Download Image",
                data=file,
                file_name=st.session_state.generated_images[0],
                mime="image/png",
                use_container_width=True
            )

with col2:
    if st.button("🎨 Generate 3 Variations", type="primary", use_container_width=True):
        if prompt and image_paths:
            with st.spinner("Generating 3 variations..."):
                result_paths = generate_multiple_images(
                    prompt, 
                    image_paths, 
                    count=3,
                    base_seed=st.session_state.base_seed,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio
                )
                
                if result_paths:
                    st.session_state.generated_images = result_paths
                    st.success(f"✅ Generated {len(result_paths)} images!")
                else:
                    st.error("❌ Failed to generate images.")
        elif not prompt:
            st.warning("⚠️ Please enter a description.")
    
    # Display stored images
    if len(st.session_state.generated_images) == 3:
        img_cols = st.columns(3)
        
        for idx, path in enumerate(st.session_state.generated_images):
            with img_cols[idx]:
                st.image(path, caption=f"Variation {idx+1}")
                with open(path, "rb") as file:
                    st.download_button(
                        label=f"📥 #{idx+1}",
                        data=file,
                        file_name=path,
                        mime="image/png",
                        key=f"dl_btn_{idx}",
                        use_container_width=True
                    )