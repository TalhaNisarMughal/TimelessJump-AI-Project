import streamlit as st
import os
import base64
import uuid
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import logging
import random
import shutil
import io
from io import BytesIO
import threading

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_KEY")
image_lock = threading.Lock()
image_content = []
count = 0
# def load_images(images_path):
#     global count
#     if count==0:
#         for image_path in images_path:
#             img = Image.open(image_path)
#             img.load()
#             image_content.append(img)
#             logger.info(f"✓ Loaded: {os.path.basename(image_path)}")

#         if len(image_content) > 0:
#             count = 1
image_content_parts = []
def load_images(images_path):
    global count
    if count == 0:
        for image_path in images_path:
            with Image.open(image_path) as img:
                # 1. Force Load
                img.load()
                
                # 2. Resize if too large (CRITICAL for multi-thread reliability)
                # Gemini doesn't need 4K. 1024px is plenty and 4x faster to upload.
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # 3. Convert to immutable BYTES immediately
                buf = BytesIO()
                # Default to PNG, or preserve original format if available
                fmt = img.format if img.format else 'PNG'
                img.save(buf, format=fmt)
                byte_data = buf.getvalue()
                
                # 4. Store as a Gemini-ready Part dictionary
                image_content_parts.append({
                    'mime_type': f'image/{fmt.lower()}',
                    'data': byte_data
                })
                
            logger.info(f"✓ Loaded & Optimized: {os.path.basename(image_path)}")

        if len(image_content_parts) > 0:
            count = 1

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
        color_instruction = f"\n\nCOLOR SPECIFICATION: Apply the color {selected_color} (hex code) to the Main product. Ensure accurate color matching to this exact hex value.Ensure that this color is not applied to anything other then the Main Object unless specified by the user"
    
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
    if variation_number is not None :
        var_idx = variation_number % len(minimal_variations)
        if minimal_variations[var_idx]:
            variation_instruction = f"\n\nMINOR ADJUSTMENT: {minimal_variations[var_idx]}. This is the ONLY change allowed."
    
    # IMPROVED: Direct, natural language system instruction
    # Based on official docs: "Gemini 3 prefers direct, efficient answers"
    # Remove XML bloat, focus on clear natural instructions
    system_instruction = f"""You are an expert prompt engineer for Gemini 3 Pro Image editing. Your role is to create prompts that maintain EXACT physical fidelity to uploaded reference images while applying user-requested changes.

CONTEXT:
The user has uploaded reference images of a product. Your task is to create an editing prompt that preserves every physical detail while applying only the requested modifications.

CRITICAL FOCUS AREAS FOR PRODUCT PRESERVATION:

1. PRIMARY COMPONENTS (highest priority):
   - Exact shape, dimensions, and proportions of all major parts
   - Precise surface texture patterns (ridges, grooves, patterns, embossing)
   - Component connections and attachment points
   - Weight distribution and structural proportions
   - Surface finish characteristics (matte, glossy, textured, polished areas)

2. TEXT AND BRANDING ELEMENTS:
   - Any text, logos, or brand names printed/embossed/engraved on the product
   - Font style, size, and placement of all text elements
   - Text legibility and accuracy - spell exactly as shown
   - Logo colors, proportions, and positioning
   - Printing/marking method appearance (screen print, embossed, debossed, laser-etched)

3. MATERIAL AND SURFACE CHARACTERISTICS:
   - Exact material type and appearance for each component
   - Surface texture and pattern details
   - Color gradients, sheen, and transparency (if applicable)
   - Natural curves, bends, or structural features
   - Join lines, seams, and connection mechanisms

4. DETAILED FEATURES:
   - Small components, buttons, switches, or functional elements
   - Decorative details and ornamental features
   - Any stitching, weaving, or joining patterns
   - Hardware elements (screws, rivets, clasps, etc.)
   - Unique identifying characteristics

USER'S REQUEST:
{user_prompt}

Everything else must remain identical to the references.{color_instruction}{variation_instruction}

WHAT TO PRESERVE:
Everything except what the user explicitly requests to change. This includes:
- All component dimensions and shapes
- Every text element and logo
- Material construction and textures
- Surface patterns and finish details
- Product proportions and scale
- Connection mechanisms and hardware
- Structural features and details

CRITICAL - MAKE SURE GEMINI 3 PRO IMAGE GEN FOLLOWS THIS:
- The logo and text direction should remain the same as in the reference images
- Explicitly emphasize reference images as the source of truth
- Tell the model to carefully study the reference images before generating
- Put heavy emphasis that the product should match this exact specification: {color_instruction}
- If information about background color is not provided, choose a white background

WHAT TO CHANGE:
Only the specific attributes mentioned in the user's request, such as:
- Color changes (apply to specified parts only)
- Material finish changes (maintain same shape/texture depth)
- Surface texture modifications (keep same physical form)
- Style variations (while preserving core identity)

PHOTOGRAPHY SPECIFICATIONS (for consistency):
- Camera: 45-degree elevated angle, 85mm lens, f/5.6 aperture
- Lighting: Three-point softbox setup (key 45° left, fill 45° right at 50%, rim from behind), 5600K color temperature
- Background: Pure seamless white (RGB 255,255,255)
- Composition: Product centered, occupying 70% of frame height
- Focus: Sharp throughout entire product, especially any text elements and fine details

TEXT RENDERING RULES:
- If the product has text/logos, preserve spelling and layout EXACTLY
- Render all text legibly and sharply in focus
- Maintain original font characteristics unless user requests changes
- Keep logo proportions and colors accurate to reference

OUTPUT FORMAT:
Write a single, descriptive paragraph that:
1. Begins with "Based on the uploaded reference images of the [product name/type], carefully study [specific key components and features visible in the images], the [material/construction details], and any text or branding elements..."
2. States ONLY the requested changes explicitly
3. Emphasizes preservation of all physical details, text/logos, and material characteristics
4. Includes complete photography specifications
5. Uses natural, descriptive language (not keyword lists)
6. Ends with "Ensure the product is completely clean with no additional text, watermarks, or labels beyond what exists in the original design."

Remember: Gemini 3 Pro Image excels at text rendering and detail preservation. Be explicit about maintaining existing text and logo elements - they are key product identifiers that must remain consistent across all variations.
"""
    try:
        import os
        import glob
        try:
            local_images = []
            with image_lock:
                # Create a fresh copy of the actual IMAGE OBJECT, not just the list
                for img in image_content_parts:
                    local_images.append(img.copy())
            contents = [system_instruction] + local_images
            logger.info("Content loaded with images")
        except :
            contents = [system_instruction]
            logger.info("Content loaded without images")

        logger.info("Data sent to 2.5 pro\n")
        logger.info(contents)
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # CRITICAL FIX: Use gemini-2.5-flash for prompt refinement
        # (Gemini 3 Pro Image is for the actual image generation, not prompt refinement)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.5,  # Lower for more consistency
                    top_p=0.85,       # Reduced for deterministic outputs
                    top_k=30          # Tighter token selection
                )
            )

        except Exception as e:
            logger.warning(f"Image failed, retrying text-only: {e}")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[system_instruction],
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
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # IMPORTANT: Reference images MUST be included in generation contents
        
    # Create a fresh copy of the actual IMAGE OBJECT, not just the list
        api_image_parts = []
        with image_lock:
            for img_dict in image_content_parts:
                api_image_parts.append(
                    types.Part.from_bytes(
                        data=img_dict['data'],
                        mime_type=img_dict['mime_type']
                    )
                )
        contents = [refined_prompt] + api_image_parts
        
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
                temperature=0.5, 
                image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution
                ) # Keep at 1.0 (Google's recommendation for Gemini 3)
            )
        )
        
        # model="gemini-3-pro-image-preview",
        # config=types.GenerateContentConfig(
        #     response_modalities=['TEXT', 'IMAGE'],
        #     temperature=0.5,
        #     image_config=types.ImageConfig(
        #         aspect_ratio=aspect_ratio,
        #         image_size=resolution
        #     )
        # )
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
    

def generate_image_without_prompt_generation(refined_prompt, aspect_ratio,provided_image,client=None, chat_session=None, resolution="1K"):
    
    # CRITICAL: Build message content with prompt FIRST, then ALL images

    logger.info("🔧 Creating new persistent client")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
# Create a fresh copy of the actual IMAGE OBJECT, not just the list
    
    message_content = [refined_prompt] + provided_image

    chat_session = client.chats.create(
        model="gemini-3-pro-image-preview",
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
            temperature=0.0,
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
    
    
    
    logger.info("📤 SENDING FIRST MESSAGE TO MODEL:")
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

    response = chat_session.send_message(message_content)
    logger.info("✅ Response received from model")
    logger.info("-"*80)
    
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
                
                # CLEANUP: Close all loaded reference images
                if loaded_images:
                    logger.info(f"🧹 Closing {len(loaded_images)} reference images...")
                    for img in loaded_images:
                        try:
                            img.close()
                        except Exception as cleanup_error:
                            logger.warning(f"   ⚠️ Error closing image: {cleanup_error}")
                    logger.info("   ✅ All reference images closed")
                
                return {"image_path": output_path, "client":  client,"chat_session": chat_session,"aspect_ratio" : aspect_ratio}
                
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
    
    # Parse and save the generated image
    logger.info("🔍 Parsing response parts:")
    
    # CLEANUP: Close images even if generation failed
    return None , client, chat_session
def generate_multiple_images_with_single_prompt(
    user_prompt,
    aspect_ratios,
    resolution="1K",
    selected_color=None
):
    variation_1 = None
    refined_prompt = refine_prompt(user_prompt, variation_1, selected_color)

    api_image_parts = []
    with image_lock:
        for img_dict in image_content_parts:
            api_image_parts.append(
                types.Part.from_bytes(
                    data=img_dict['data'],
                    mime_type=img_dict['mime_type']
                )
            )
    message_content = [refined_prompt] + api_image_parts
    logger.info("🔧 Creating new persistent client")
    client = genai.Client(api_key=GEMINI_API_KEY)

    chat_session = client.chats.create(
        model="gemini-3-pro-image-preview",
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
            temperature=0.0,
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratios[0],
                image_size=resolution
            )
        )
    )
    response = chat_session.send_message(message_content)

    logger.info("✅ Chat session created successfully")
    logger.info("-"*80)

    # Initialize image registry
    image_registry = {}
    first_image = None
    image_found = False

    # Process first generated image
    for idx, part in enumerate(response.parts):
        logger.info(f"   Part {idx}: {type(part).__name__}")
        
        if part.inline_data is not None:
            logger.info(f"      ✓ Found inline image data!")
            image_found = True
            
            try:
                # Extract and save image
                image = part.as_image()
                if isinstance(image, Image.Image):
                    first_image_pil = image.copy()  # Create a copy
                else:
                    # If it's not a PIL Image, convert it
                    first_image_pil = Image.open(io.BytesIO(part.inline_data.data))
                  # Store for later use
                output_path = f"generated_jump_rope_{uuid.uuid4().hex[:8]}.png"
                image.save(output_path)
                
                # Verify saved file
                saved_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"💾 Image saved successfully:")
                logger.info(f"   Path: {output_path}")
                
                # Add first image to registry (index 0)
                image_registry[0] = {
                    "image_path": output_path,
                    "client": client,
                    "chat_session": chat_session,
                    "aspect_ratio": aspect_ratios[0]
                }
                
                # CLEANUP: Close all loaded reference images
                if loaded_images:
                    logger.info(f"🧹 Closing {len(loaded_images)} reference images...")
                    for img in loaded_images:
                        try:
                            img.close()
                        except Exception as cleanup_error:
                            logger.warning(f"   ⚠️ Error closing image: {cleanup_error}")
                    logger.info("   ✅ All reference images closed")
                
                break  # Only process first image
                
            except Exception as e:
                logger.error(f"❌ Failed to save image: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

    # Remove first aspect ratio since we already generated it
    aspect_ratios.pop(0)

    # Generate additional variations if needed
    if len(aspect_ratios) > 0 and first_image_pil is not None:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    generate_image_without_prompt_generation,
                    "recreate the exact image",
                    aspect_ratios[i],
                    [first_image_pil.copy()],  # Pass as list to match expected format
                    None,           # client (created internally)
                    None,           # chat_session (created internally)
                    resolution
                ): i + 1  # Start from index 1 since 0 is the first image
                for i in range(len(aspect_ratios))
            }

            for future, image_id in futures.items():
                try:
                    result = future.result()

                    if result is None:
                        continue

                    image_registry[image_id] = {
                        "image_path": result["image_path"],
                        "client": result["client"],
                        "chat_session": result["chat_session"]
                    }
                except Exception as e:
                    logger.error(f"❌ Failed to generate variation {image_id}: {str(e)}")

    return image_registry

def generate_image_with_chat(user_prompt, image_paths, aspect_ratio,client=None, chat_session=None, resolution="1K", selected_color=None):
    """
    Generate/edit image using Gemini 3 Pro Image with multi-turn chat support.
    EXPLICITLY loads and sends all reference images to the model using BytesIO.
    
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
  # Track loaded images for cleanup
    
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
            
            # Create chat with config
            logger.info("🔧 Creating chat session with configuration:")
            logger.info(f"   Model: gemini-3-pro-image-preview")
            logger.info(f"   Resolution: {resolution}")
            logger.info(f"   Aspect Ratio: {aspect_ratios}")
            logger.info(f"   Temperature: 1.0")
            variation_1=None
            refined_prompt = refine_prompt(user_prompt,variation_1,selected_color)
            # CRITICAL: Build message content with prompt FIRST, then ALL images
            with image_lock:
                thread_images = image_content.copy()
            message_content = [refined_prompt] + thread_images

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
            
            
            
            logger.info("📤 SENDING FIRST MESSAGE TO MODEL:")
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
                    
                    # CLEANUP: Close all loaded reference images
                    if loaded_images:
                        logger.info(f"🧹 Closing {len(loaded_images)} reference images...")
                        for img in loaded_images:
                            try:
                                img.close()
                            except Exception as cleanup_error:
                                logger.warning(f"   ⚠️ Error closing image: {cleanup_error}")
                        logger.info("   ✅ All reference images closed")
                    
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
        
        # Parse and save the generated image
        logger.info("🔍 Parsing response parts:")
        
        # CLEANUP: Close images even if generation failed
        return None , client, chat_session
    
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
        
        # CLEANUP: Close images even on exception
        return None, client, None
    



def generate_multiple_images(user_prompt, image_paths, aspect_ratios, count=3, base_seed=42, resolution="1K", selected_color=None):
    """
    Generates exactly `count` images (default 3) by cycling through available aspect ratios.
    
    Examples:
    - 1 aspect ratio selected → generates 3 images all with that ratio
    - 2 aspect ratios selected → generates 3 images (ratio1, ratio2, ratio1)
    - 3 aspect ratios selected → generates 3 images (ratio1, ratio2, ratio3)
    """
    
    if not user_prompt or not image_paths:
        logger.error("Invalid input for multiple image generation")
        return []
    
    if not aspect_ratios or len(aspect_ratios) == 0:
        logger.error("No aspect ratios provided")
        return []
    
    # Cycle through aspect ratios to reach desired count
    extended_aspect_ratios = []
    for i in range(count):
        extended_aspect_ratios.append(aspect_ratios[i % len(aspect_ratios)])
    
    logger.info(f"Generating {count} images with aspect ratios: {extended_aspect_ratios}")
    logger.info(f"Original aspect ratios selected: {aspect_ratios}")
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(
                generate_image, 
                user_prompt, 
                image_paths, 
                variation_number=i,
                base_seed=base_seed,
                resolution=resolution,
                aspect_ratio=extended_aspect_ratios[i],
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


# Alternative approach: Generate exactly 3 images by repeating aspect ratios
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
    st.session_state.show_edit_field = {}  # Changed from False

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
    st.write("**Aspect Ratio**")
    aspect_ratios = []
    
    if st.checkbox("Square (1:1)", key="square_ratio"):
        aspect_ratios.append("1:1")
    
    if st.checkbox("Vertical (9:16)", key="vertical_ratio"):
        aspect_ratios.append("9:16")
    
    if st.checkbox("Horizontal (16:9)", key="horizontal_ratio"):
        aspect_ratios.append("16:9")
st.divider()

folder_options = {
    "PVC Jump Ropes": "gemini_images",
    "Hercule S1": "Hercule",
    "JAB 101": "JAB_101",
    "Speed Rope": "Speed_Rope",
    "B&W Strap Miri" : "B&W_Strap_Miri"
}

# Create the dropdown with default set to "Gemini Images"
selected_folder_name = st.selectbox(
    "Select Image Folder",
    options=list(folder_options.keys()),
    index=0  # This sets "Gemini Images" as default (first option)
)

# Get the corresponding folder path
folder_path = folder_options[selected_folder_name]

# Display the selected path
st.info(f"Loading images from: {folder_path}")

# Load images from the selected folder
# These functions will be called whenever the dropdown selection changes
image_paths = get_images_from_folder(folder_path)
loaded_images = load_images(image_paths)

st.divider()
if len(aspect_ratios) < 1:
    aspect_ratios.append("16:9")
prompt = st.text_area(
    "Describe what you want to change:", 
    height=100,
    placeholder="Example: Change the rope and handles to the selected color with matte finish"
)


if not image_paths:
    st.warning("⚠️ No reference images found in 'gemini_images' folder.")
    logger.error("⚠️ No reference images found in 'gemini_images' folder.")
else:
    logger.info(f"✅ {len(image_paths)} reference images loaded")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🎨 Generate Images", type="primary", use_container_width=True):
        if not st.session_state.selected_color:
            st.warning("⚠️ Please select a color first.")
        elif prompt and image_paths:
            with st.spinner("Generating images..."):
                logger.info("="*60)
                logger.info("🚀 MULTIPLE IMAGE GENERATION STARTED")
                logger.info(f"📝 User prompt: {prompt}")
                logger.info(f"🎨 Selected color: {st.session_state.selected_color}")
                logger.info(f"🖼️ Reference images: {len(image_paths)}")
                logger.info(f"📐 Resolution: {resolution}, Aspect ratios: {aspect_ratios}")
                logger.info(f"🔢 Number of images to generate: {len(aspect_ratios)}")
                
                # Returns dictionary with image_id as key
                image_registry = generate_multiple_images_with_single_prompt(
                    prompt,
                    aspect_ratios=aspect_ratios,
                    resolution=resolution,
                    selected_color=st.session_state.selected_color
                )

                load_images(image_paths)
                
                if image_registry:
                    # Store the registry in session state
                    st.session_state.image_registry = image_registry
                    st.session_state.edit_history = {img_id: [prompt] for img_id in image_registry.keys()}
                    st.session_state.show_edit_field = {img_id: False for img_id in image_registry.keys()}
                    
                    logger.info("✅ GENERATION SUCCESSFUL")
                    logger.info(f"💾 Generated {len(image_registry)} images")
                    for img_id, data in image_registry.items():
                        logger.info(f"   Image {img_id}: {data['image_path']}")
                    logger.info("="*60)
                    st.success(f"✅ Generated {len(image_registry)} images!")
                    st.rerun()
                else:
                    logger.error("❌ GENERATION FAILED")
                    logger.info("="*60)
                    st.error("❌ Failed to generate images.")
        elif not prompt:
            st.warning("⚠️ Please enter a description.")
            logger.warning("⚠️ User attempted generation without prompt")
    
    # Display all generated images with individual edit functionality
    if st.session_state.get('image_registry'):
        st.markdown("---")
        st.subheader("Generated Images")
        
        # Create columns for images (max 3 images side by side)
        num_images = len(st.session_state.image_registry)
        cols = st.columns(num_images)
        
        for idx, (img_id, img_data) in enumerate(st.session_state.image_registry.items()):
            with cols[idx]:
                # Thumbs down button for this specific image
                if st.button("👎", key=f"thumbs_down_{img_id}", help=f"Edit image {img_id + 1}"):
                    st.session_state.show_edit_field[img_id] = not st.session_state.show_edit_field[img_id]
                    st.rerun()
                
                # Display the image
                st.image(img_data['image_path'], caption=f"Image {img_id + 1}", use_container_width=True)
                
                # Show edit field if thumbs down was clicked for this image
                if st.session_state.show_edit_field.get(img_id, False):
                    st.markdown(f"**Edit Image {img_id + 1}:**")
                    
                    edit_prompt = st.text_input(
                        "What would you like to change?",
                        placeholder="Example: Make the handles silver",
                        key=f"edit_input_{img_id}"
                    )
                    
                    edit_btn_col1, edit_btn_col2 = st.columns([1, 1])
                    
                    with edit_btn_col1:
                        if st.button("✏️ Apply", type="primary", use_container_width=True, key=f"apply_edit_{img_id}"):
                            if not st.session_state.selected_color:
                                st.warning("⚠️ Please select a color first.")
                            elif edit_prompt:
                                with st.spinner(f"Editing image {img_id + 1}..."):
                                    logger.info("="*60)
                                    logger.info(f"✏️ MULTI-TURN EDIT STARTED (Image {img_id})")
                                    logger.info(f"📝 Edit instruction: {edit_prompt}")
                                    logger.info(f"🎨 Selected color: {st.session_state.selected_color}")
                                    logger.info(f"🔄 Edit number: {len(st.session_state.edit_history[img_id]) + 1}")
                                    
                                    result_path, updated_client, updated_chat = generate_image_with_chat(
                                        edit_prompt,
                                        image_paths=None,
                                        client=img_data['client'],
                                        chat_session=img_data['chat_session'],
                                        resolution=resolution,
                                        aspect_ratio=aspect_ratios[img_id],
                                        selected_color=st.session_state.selected_color
                                    )

                                    load_images(image_paths)
                                    
                                    if result_path:
                                        # Update the specific image in registry
                                        st.session_state.image_registry[img_id]['image_path'] = result_path
                                        st.session_state.image_registry[img_id]['client'] = updated_client
                                        st.session_state.image_registry[img_id]['chat_session'] = updated_chat
                                        st.session_state.edit_history[img_id].append(edit_prompt)
                                        if 'show_edit_field' not in st.session_state:
                                            st.session_state.show_edit_field = {}
                                        
                                        logger.info("✅ EDIT SUCCESSFUL")
                                        logger.info(f"💾 Edited image saved to: {result_path}")
                                        logger.info("="*60)
                                        st.success(f"✅ Image {img_id + 1} edited!")
                                        st.rerun()
                                    else:
                                        logger.error("❌ EDIT FAILED")
                                        logger.info("="*60)
                                        st.error(f"❌ Failed to edit image {img_id + 1}.")
                            else:
                                st.warning("⚠️ Please enter edit instructions.")
                    
                    with edit_btn_col2:
                        if st.button("❌ Cancel", use_container_width=True, key=f"cancel_edit_{img_id}"):
                            st.session_state.show_edit_field[img_id] = False
                            st.rerun()
                
                # Show edit history if exists (MOVED INSIDE THE LOOP)
                if st.session_state.edit_history.get(img_id) and len(st.session_state.edit_history[img_id]) > 0:
                    with st.expander(f"📜 Edit History ({len(st.session_state.edit_history[img_id])} edits)"):
                        for edit_idx, edit in enumerate(st.session_state.edit_history[img_id], 1):
                            st.text(f"{edit_idx}. {edit}")
                
                # Download button for this specific image (MOVED INSIDE THE LOOP)
                with open(img_data['image_path'], "rb") as file:
                    st.download_button(
                        label="📥 Download",
                        data=file,
                        file_name=f"image_{img_id + 1}_{os.path.basename(img_data['image_path'])}",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_{img_id}"
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
                    aspect_ratios=aspect_ratios, 
                    count=3,
                    base_seed=st.session_state.base_seed,
                    resolution=resolution,
                    selected_color=st.session_state.selected_color
                )
                load_images(image_paths)
                if result_paths:
                    st.session_state.generated_images = result_paths
                    # Reset single image session when generating variations
                    st.session_state.chat_session = None
                    st.session_state.client = None
                    st.session_state.current_image = None
                    st.session_state.edit_history = [] 
                    st.session_state.show_edit_field = {}
                    st.session_state.image_registry = {}
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