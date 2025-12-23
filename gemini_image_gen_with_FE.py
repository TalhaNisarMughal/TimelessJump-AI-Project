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
import shutil

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_KEY")

def refine_prompt(user_prompt, variation_number=None, selected_color=None):
    """
    Refine user prompts for Gemini 3 Pro Image generation with MAXIMUM CONSISTENCY.
    Based on official Google documentation best practices.
    
    Args:
        user_prompt: User's description of the desired product variant
        variation_number: Optional number for MINIMAL variations
        selected_color: Hex color code selected by user (e.g., "#FF5733")
    """
    
    reference_images_folder = "gemini_images"
    
    # Add color instruction if color is provided
    color_instruction = ""
    if selected_color:
        color_instruction = f"\n\nCOLOR SPECIFICATION: Apply the color {selected_color} (hex code) to the parts mentioned in the user's request. Ensure accurate color matching to this exact hex value."
    
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
Everything else must remain identical to the references.{color_instruction}{variation_instruction}

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
        color_fallback = f" Use color {selected_color}." if selected_color else ""
        return f"Based on the uploaded reference images, {user_prompt}.{color_fallback} Maintain all physical characteristics exactly as shown. Professional studio photography, clean product."


def generate_image(user_prompt, image_paths, variation_number=None, base_seed=42, resolution="1K", aspect_ratio="16:9", selected_color=None):
    """
    Generate image using Gemini 3 Pro Image with CONSISTENCY controls
    
    Args:
        user_prompt: User's description of the desired image
        image_paths: List of paths to reference images (up to 10)
        variation_number: Number for creating subtle variations (0, 1, 2...)
        base_seed: Base seed for reproducibility (same seed = similar results)
        resolution: Image resolution (1K, 2K, 4K)
        aspect_ratio: Aspect ratio (16:9, 1:1, etc.)
        selected_color: Hex color code selected by user
    """
    
    if not user_prompt or not user_prompt.strip():
        logger.error("Empty user prompt provided")
        return None
    
    if not image_paths or len(image_paths) == 0:
        logger.error("No image paths provided")
        return None
    
    if len(image_paths) > 10:
        image_paths = image_paths[:10]
    
    # CRITICAL: Generate refined prompt with controlled variation and color
    refined_prompt = refine_prompt(user_prompt, variation_number=variation_number, selected_color=selected_color)
    
    # FIX: Create temporary copies of images for this thread to avoid file conflicts
    temp_images = []
    thread_id = uuid.uuid4().hex[:8]
    
    for idx, path in enumerate(image_paths):
        try:
            # Create a temporary copy with unique name
            temp_path = f"temp_{thread_id}_{idx}_{os.path.basename(path)}"
            shutil.copy2(path, temp_path)
            
            # Load from the temporary copy
            img = Image.open(temp_path)
            temp_images.append(img)
            
            # Clean up the temporary file immediately after loading
            try:
                os.remove(temp_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            pass
    
    if not temp_images:
        logger.error("No images could be loaded")
        return None
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # IMPORTANT: Reference images MUST be included in generation contents
        contents = [refined_prompt] + temp_images
        
        # Calculate deterministic seed for this variation
        if variation_number is not None:
            generation_seed = base_seed + variation_number
        else:
            generation_seed = base_seed
        
        logger.info(f"Generating image (variation: {variation_number}, seed: {generation_seed})...")
        logger.info(f"Refined prompt length: {len(refined_prompt)} chars")
        
        # CRITICAL: Add consistency parameters
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                temperature=1.0,  # Keep at 1.0 (Google's recommendation for Gemini 3)
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


def generate_image_with_chat(user_prompt, image_paths, client=None, chat_session=None, resolution="1K", aspect_ratio="16:9", selected_color=None):
    """
    Generate/edit image using Gemini 3 Pro Image with multi-turn chat support.
    EXPLICITLY loads and sends all reference images to the model.
    
    Args:
        user_prompt: User's description of desired changes
        image_paths: List of paths to reference images (only used on first turn)
        client: Shared genai.Client instance (MUST be provided for multi-turn)
        chat_session: Existing chat session (None for first turn)
        resolution: Image resolution (1K, 2K, 4K)
        aspect_ratio: Aspect ratio (16:9, 1:1, etc.)
        selected_color: Hex color code selected by user
    
    Returns:
        tuple: (output_path, client, chat_session) - path to saved image, client, and chat session for next turn
    """
    
    if not user_prompt or not user_prompt.strip():
        logger.error("❌ Empty user prompt provided")
        return None, client, None
    
    try:
        # Create client ONCE if not provided
        if client is None:
            logger.info("🔧 Creating new persistent client")
            client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Create new chat session if this is the first turn
        if chat_session is None:
            logger.info("="*80)
            logger.info("🆕 INITIALIZING NEW MULTI-TURN CHAT SESSION")
            logger.info("="*80)
            
            if not image_paths or len(image_paths) == 0:
                logger.error("❌ No image paths provided for initial generation")
                return None, client, None
            
            # Limit to 10 images
            if len(image_paths) > 10:
                logger.warning(f"⚠️ Limiting from {len(image_paths)} to 10 reference images")
                image_paths = image_paths[:10]
            
            logger.info(f"📂 Attempting to load {len(image_paths)} reference images:")
            
            # EXPLICITLY LOAD ALL REFERENCE IMAGES
            loaded_images = []
            for idx, path in enumerate(image_paths, 1):
                try:
                    # Check if file exists
                    if not os.path.exists(path):
                        logger.error(f"  ❌ Image {idx}: File not found at {path}")
                        continue
                    
                    # Check file size
                    file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
                    logger.info(f"  📸 Image {idx}: {os.path.basename(path)} ({file_size:.2f} MB)")
                    
                    # Load the image using PIL
                    img = Image.open(path)
                    
                    # Log image properties
                    logger.info(f"      ✓ Loaded: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}")
                    
                    # Add to list
                    loaded_images.append(img)
                    
                except Exception as e:
                    logger.error(f"  ❌ Image {idx}: Failed to load - {str(e)}")
                    import traceback
                    logger.error(f"      {traceback.format_exc()}")
            
            if not loaded_images:
                logger.error("❌ CRITICAL: No images could be loaded successfully!")
                return None, client, None
            
            logger.info(f"✅ Successfully loaded {len(loaded_images)}/{len(image_paths)} reference images")
            logger.info("-"*80)
            
            # Create chat with config
            logger.info("🔧 Creating chat session with configuration:")
            logger.info(f"   Model: gemini-3-pro-image-preview")
            logger.info(f"   Resolution: {resolution}")
            logger.info(f"   Aspect Ratio: {aspect_ratio}")
            logger.info(f"   Temperature: 1.0")
            
            chat_session = client.chats.create(
                model="gemini-3-pro-image-preview",
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    temperature=1.0,
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution
                    )
                )
            )
            logger.info("✅ Chat session created successfully")
            logger.info("-"*80)
            
            # Build the refined prompt
            color_instruction = ""
            if selected_color:
                color_instruction = f" Apply the color {selected_color} to the specified parts."
            
            refined_prompt = f"""
<critical_reference_analysis>
PRIORITY INSTRUCTIONS - Handle & Logo Fidelity:
You have {len(loaded_images)} reference images. ONE of these images is a DEDICATED LOGO REFERENCE showing brand text and graphics in isolation.

MANDATORY REQUIREMENTS:
1. HANDLE DESIGN (HIGHEST PRIORITY):
   - Exact dimensions: 6.3 inches (16 cm) length × 0.9 inches (23 mm) diameter
   - Preserve every surface texture, grip pattern, and material finish
   - Maintain exact handle shape and ergonomic contours
   
2. LOGO & TEXT PLACEMENT (CRITICAL - DO NOT DEVIATE):
   - Study the standalone logo reference image with EXTREME attention
   - Reproduce logo positioning EXACTLY as shown on handles
   - Maintain identical font styles, sizes, and letter spacing
   - Preserve all text orientations and placement angles
   - Keep logo colors, gradients, and opacity PRECISELY as reference
   - Replicate any embossing, engraving, or texture on text/logos
   - Maintain spatial relationship between multiple logos if present

3. BRANDING ELEMENTS:
   - Copy every detail: wordmarks, symbols, icons, decorative elements
   - Preserve text hierarchy and visual weight
   - Maintain any shadows, highlights, or dimensional effects on logos
   - Keep exact contrast ratios between text and handle surface

4. ROPE/CABLE CONSTRUCTION:
   - Material type and texture from reference
   - Cable thickness and construction style
   - Connection points to handles
   - Dont make it too long or too short
</critical_reference_analysis>

<user_request>
{user_prompt}{color_instruction}
</user_request>

<generation_specifications>
Subject: Professional jump rope product featuring handles (6.3" L × 0.9" dia.) with EXACT branding from reference

Composition & Framing:
- Three-quarter product view at 45-degree angle
- Handles prominently displayed to showcase logo placement
- Product centered in frame with optimal logo visibility
- Ensure all branded surfaces are clearly visible and readable

Visual Style: High-end commercial product photography

Camera & Lighting Setup:
- 85mm lens equivalent for natural proportions
- Three-point studio lighting optimized for text/logo clarity
- Key light positioned to eliminate glare on text surfaces
- Fill light to reveal logo details in shadow areas
- Rim lighting to accentuate handle edges and text dimensionality
- Clean white background (RGB 255, 255, 255) for maximum contrast

Technical Constraints:
1. Handle fidelity is NON-NEGOTIABLE - dimensions must be 6.3" × 0.9" diameter
2. Logo/text placement is ABSOLUTE PRIORITY - replicate with 100% accuracy
3. Study the dedicated logo reference image before generating
4. Preserve ALL branding elements exactly as photographed
5. Apply ONLY user-requested changes - everything else stays identical
6. Output must be pristine: no watermarks, no additional text overlays
7. Maintain professional product photography standards throughout

Reasoning Mode: Enable "Thinking" mode to verify logo accuracy before final generation
</generation_specifications>
"""
            
            logger.info("📝 Refined prompt created:")
            logger.info(f"   Length: {len(refined_prompt)} characters")
            logger.info(f"   Preview: {refined_prompt[:200]}...")
            logger.info("-"*80)
            
            # CRITICAL: Build message content with prompt FIRST, then ALL images
            message_content = [refined_prompt] + loaded_images
            
            logger.info("📤 SENDING FIRST MESSAGE TO MODEL:")
            logger.info(f"   Content parts: 1 text prompt + {len(loaded_images)} images")
            logger.info(f"   Total message parts: {len(message_content)}")
            
            # Verify each part
            for idx, part in enumerate(message_content):
                if isinstance(part, str):
                    logger.info(f"   Part {idx}: TEXT ({len(part)} chars)")
                elif isinstance(part, Image.Image):
                    logger.info(f"   Part {idx}: IMAGE ({part.size[0]}x{part.size[1]}, {part.mode})")
                else:
                    logger.info(f"   Part {idx}: {type(part)}")
            
            logger.info("-"*80)
            logger.info("⏳ Waiting for model response...")
            
        else:
            # Subsequent turns: just send the edit instruction
            logger.info("="*80)
            logger.info("🔄 CONTINUING EXISTING CHAT SESSION (MULTI-TURN EDIT)")
            logger.info("="*80)
            
            message_content = user_prompt
            if selected_color:
                message_content = f"{user_prompt} Apply color {selected_color}."
            
            logger.info(f"📤 Sending edit instruction: {message_content}")
            logger.info("   (Reference images maintained via chat context)")
            logger.info("-"*80)
            logger.info("⏳ Waiting for model response...")
        
        # Send message and get response
        response = chat_session.send_message(message_content)
        logger.info("✅ Response received from model")
        logger.info("-"*80)
        
        # Parse and save the generated image
        logger.info("🔍 Parsing response parts:")
        image_found = False
        
        for idx, part in enumerate(response.parts):
            logger.info(f"   Part {idx}: {type(part).__name__}")
            
            if part.inline_data is not None:
                logger.info(f"      ✓ Found inline image data!")
                image_found = True
                
                try:
                    # Extract and save image
                    image = part.as_image()
                    output_path = f"generated_jump_rope_{uuid.uuid4().hex[:8]}.png"
                    image.save(output_path)
                    
                    # Verify saved file
                    saved_size = os.path.getsize(output_path) / (1024 * 1024)
                    logger.info(f"💾 Image saved successfully:")
                    logger.info(f"   Path: {output_path}")
                    
                    return output_path, client, chat_session
                    
                except Exception as e:
                    logger.error(f"❌ Failed to save image: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            elif hasattr(part, 'text') and part.text:
                logger.info(f"      Text content: {part.text[:100]}...")
        
        if not image_found:
            logger.error("❌ NO IMAGE DATA FOUND IN RESPONSE")
            logger.error("   This suggests the model did not generate an image.")
            logger.error("   Possible causes:")
            logger.error("   1. Reference images were not properly received by model")
            logger.error("   2. Prompt was unclear or conflicting")
            logger.error("   3. Model encountered an error during generation")
            logger.info("="*80)
        
        return None, client, chat_session
    
    except Exception as e:
        logger.error("="*80)
        logger.error(f"❌ CRITICAL ERROR IN CHAT GENERATION")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("-"*80)
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        logger.info("="*80)
        return None, client, None
        
def generate_multiple_images(user_prompt, image_paths, count=3, base_seed=42, resolution="1K", aspect_ratio="16:9", selected_color=None):
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
                aspect_ratio=aspect_ratio,
                selected_color=selected_color
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

# Initialize session state variables
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

if 'base_seed' not in st.session_state:
    st.session_state.base_seed = 42

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None

if 'client' not in st.session_state:
    st.session_state.client = None

if 'current_image' not in st.session_state:
    st.session_state.current_image = None

if 'edit_history' not in st.session_state:
    st.session_state.edit_history = []

if 'show_edit_field' not in st.session_state:
    st.session_state.show_edit_field = False

if 'selected_color' not in st.session_state:
    st.session_state.selected_color = None

st.title("🎯 Product Image Generator")
st.caption("Transform your product with AI")

# Color Picker
st.subheader("🎨 Color Selection")
col_color1, col_color2 = st.columns([3, 1])

with col_color1:
    selected_color = st.color_picker("Pick a color for your product", "#FF5733")
    st.session_state.selected_color = selected_color

st.caption(f"Selected Color: {selected_color}")

st.divider()

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
    placeholder="Example: Change the rope and handles to the selected color with matte finish"
)

folder_path = "gemini_images"
image_paths = get_images_from_folder(folder_path)

if not image_paths:
    st.warning("⚠️ No reference images found in 'gemini_images' folder.")
    logger.error("⚠️ No reference images found in 'gemini_images' folder.")
else:
    logger.info(f"✅ {len(image_paths)} reference images loaded")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🎨 Generate Single Image", type="primary", use_container_width=True):
        if not st.session_state.selected_color:
            st.warning("⚠️ Please select a color first.")
        elif prompt and image_paths:
            with st.spinner("Generating image..."):
                logger.info("="*60)
                logger.info("🚀 SINGLE IMAGE GENERATION STARTED")
                logger.info(f"📝 User prompt: {prompt}")
                logger.info(f"🎨 Selected color: {st.session_state.selected_color}")
                logger.info(f"🖼️ Reference images: {len(image_paths)}")
                logger.info(f"📐 Resolution: {resolution}, Aspect: {aspect_ratio}")
                
                result_path, updated_client, new_chat = generate_image_with_chat(
                    prompt, 
                    image_paths,
                    client=None,
                    chat_session=None,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    selected_color=st.session_state.selected_color
                )
                
                if result_path:
                    st.session_state.generated_images = [result_path]
                    st.session_state.current_image = result_path
                    st.session_state.chat_session = new_chat
                    st.session_state.edit_history = [prompt]
                    st.session_state.client = updated_client
                    st.session_state.show_edit_field = False
                    logger.info("✅ GENERATION SUCCESSFUL")
                    logger.info(f"💾 Image saved to: {result_path}")
                    logger.info(f"🔗 Chat session initialized for multi-turn editing")
                    logger.info("="*60)
                    st.success("✅ Image generated!")
                    st.rerun()
                else:
                    logger.error("❌ GENERATION FAILED")
                    logger.info("="*60)
                    st.error("❌ Failed to generate image.")
        elif not prompt:
            st.warning("⚠️ Please enter a description.")
            logger.warning("⚠️ User attempted generation without prompt")
    
    # Display stored single image with edit functionality
    if len(st.session_state.generated_images) == 1:
        # Create columns for thumbs down button and image
        button_col, image_col = st.columns([0.1, 0.9])
        
        with button_col:
            if st.button("👎", key="thumbs_down", help="Edit this image"):
                st.session_state.show_edit_field = not st.session_state.show_edit_field
                st.rerun()
        
        with image_col:
            st.image(st.session_state.generated_images[0], caption="Generated Image")
        
        # Show edit field if thumbs down was clicked
        if st.session_state.show_edit_field:
            edit_prompt = st.text_input(
                "What would you like to change?",
                placeholder="Example: Make the handles silver",
                key="edit_input"
            )
            
            edit_col1, edit_col2 = st.columns([1, 1])
            
            with edit_col1:
                if st.button("✏️ Apply Edit", type="primary", use_container_width=True):
                    if not st.session_state.selected_color:
                        st.warning("⚠️ Please select a color first.")
                    elif edit_prompt:
                        with st.spinner("Editing image..."):
                            logger.info("="*60)
                            logger.info("✏️ MULTI-TURN EDIT STARTED")
                            logger.info(f"📝 Edit instruction: {edit_prompt}")
                            logger.info(f"🎨 Selected color: {st.session_state.selected_color}")
                            logger.info(f"🔄 Edit number: {len(st.session_state.edit_history) + 1}")
                            
                            result_path, updated_client, updated_chat = generate_image_with_chat(
                                edit_prompt,
                                image_paths=None,
                                client=st.session_state.client,
                                chat_session=st.session_state.chat_session,
                                resolution=resolution,
                                aspect_ratio=aspect_ratio,
                                selected_color=st.session_state.selected_color
                            )
                            
                            if result_path:
                                st.session_state.generated_images = [result_path]
                                st.session_state.current_image = result_path
                                st.session_state.chat_session = updated_chat
                                st.session_state.client = updated_client
                                st.session_state.edit_history.append(edit_prompt)
                                st.session_state.show_edit_field = False
                                logger.info("✅ EDIT SUCCESSFUL")
                                logger.info(f"💾 Edited image saved to: {result_path}")
                                logger.info("="*60)
                                st.success("✅ Image edited!")
                                st.rerun()
                            else:
                                logger.error("❌ EDIT FAILED")
                                logger.info("="*60)
                                st.error("❌ Failed to edit image.")
                    else:
                        st.warning("⚠️ Please enter edit instructions.")
            
            with edit_col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_edit_field = False
                    st.rerun()
        
        # Show edit history if exists
        if st.session_state.edit_history and len(st.session_state.edit_history) > 0:
            with st.expander(f"📜 Edit History ({len(st.session_state.edit_history)} edits)"):
                for idx, edit in enumerate(st.session_state.edit_history, 1):
                    st.text(f"{idx}. {edit}")
        
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
        if not st.session_state.selected_color:
            st.warning("⚠️ Please select a color first.")
        elif prompt and image_paths:
            with st.spinner("Generating 3 variations..."):
                logger.info("="*60)
                logger.info("🎨 MULTI-VARIATION GENERATION STARTED")
                logger.info(f"📝 User prompt: {prompt}")
                logger.info(f"🎨 Selected color: {st.session_state.selected_color}")
                logger.info(f"🔢 Variations: 3")
                
                result_paths = generate_multiple_images(
                    prompt, 
                    image_paths, 
                    count=3,
                    base_seed=st.session_state.base_seed,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    selected_color=st.session_state.selected_color
                )
                
                if result_paths:
                    st.session_state.generated_images = result_paths
                    # Reset single image session when generating variations
                    st.session_state.chat_session = None
                    st.session_state.client = None
                    st.session_state.current_image = None
                    st.session_state.edit_history = []
                    st.session_state.show_edit_field = False
                    logger.info(f"✅ GENERATED {len(result_paths)}/3 VARIATIONS")
                    logger.info("="*60)
                    st.success(f"✅ Generated {len(result_paths)} images!")
                    st.rerun()
                else:
                    logger.error("❌ VARIATION GENERATION FAILED")
                    logger.info("="*60)
                    st.error("❌ Failed to generate images.")
        elif not prompt:
            st.warning("⚠️ Please enter a description.")
            logger.warning("⚠️ User attempted variation generation without prompt")
    
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