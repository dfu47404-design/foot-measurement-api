# main.py - FOOT MEASUREMENT API WITH ALL FIXES
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import jwt
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(
    title="Foot Measurement API",
    description="Measure foot size from images using A4 paper as reference",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_URL = "postgresql://fypuser:me9tXj1XwolsOfChIupvOU1OqZQHvwab@dpg-d4pue7c9c44c73b2ie6g-a.singapore-postgres.render.com/fypdb_jozr"

# JWT Secret Key
SECRET_KEY = os.getenv("SECRET_KEY", "VUx4cXpLX2tZNTJldVRwRV9MYXJNX2RrSmpvTzJfVXo=")
ALGORITHM = "HS256"
security = HTTPBearer()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# ========== DATABASE FUNCTIONS ==========
def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user_id"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        user_id = payload.get("user_id")
        if user_id is None:
            user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token: user_id not found")
        
        try:
            user_id = int(user_id)
        except:
            pass
        
        # Verify user exists in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = %s AND is_active = TRUE", (user_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=401, detail="User not found or inactive")
        conn.close()
        
        return user_id
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

# ========== IMPROVED IMAGE PROCESSING FUNCTIONS ==========
def detect_a4_paper_simple(image):
    """
    IMPROVED A4 Paper Detection
    """
    original = image.copy()
    height, width = image.shape[:2]
    
    print(f"DEBUG: Image size: {width}x{height} pixels")
    
    # 1. Resize if too large (improves performance)
    max_size = 1200
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)  # Cast to int
        new_height = int(height * scale)  # Cast to int
        image = cv2.resize(image, (new_width, new_height))
        height, width = image.shape[:2]
        print(f"DEBUG: Resized to: {width}x{height} pixels")
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # 5. Dilate edges to connect broken lines
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 6. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"DEBUG: Found {len(contours)} contours")
    
    if not contours:
        print("DEBUG: No contours found, trying adaptive thresholding")
        # Try adaptive thresholding as fallback
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"DEBUG: Adaptive thresholding found {len(contours)} contours")
    
    if not contours:
        return None
    
    # Sort by area and take top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        image_area = height * width
        
        # Skip too small or too large contours
        if area < image_area * 0.1:  # Less than 10% of image
            continue
        if area > image_area * 0.95:  # More than 95% of image
            continue
        
        print(f"DEBUG: Contour {i}: area={area} ({area/image_area*100:.1f}% of image)")
        
        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for quadrilateral
        if len(approx) == 4:
            # Get bounding rectangle - cv2.boundingRect returns (x, y, w, h) as integers
            x, y, w, h = cv2.boundingRect(approx)
            
            # Calculate aspect ratio
            aspect_ratio = w / float(h)
            print(f"DEBUG: Quadrilateral: {w}x{h}, aspect={aspect_ratio:.3f}")
            
            # A4 aspect ratio: 21/29.7 = 0.707 (allow 0.6-0.8)
            if 0.6 <= aspect_ratio <= 0.8:
                print(f"DEBUG: ✓ A4 paper detected! Aspect ratio: {aspect_ratio:.3f}")
                
                # Draw for debugging
                debug_img = original.copy()
                cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
                cv2.imwrite("a4_debug.jpg", debug_img)
                print(f"DEBUG: Saved A4 detection debug image")
                
                return approx
    
    # If no quadrilateral found, use largest contour's bounding box
    print(f"DEBUG: No quadrilateral found, using largest contour")
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        print(f"DEBUG: Using contour bounding box: {w}x{h}")
        
        # Create quadrilateral from bounding box - ensure int32
        approx = np.array([
            [int(x), int(y)],
            [int(x + w), int(y)],
            [int(x + w), int(y + h)],
            [int(x), int(y + h)]
        ], dtype=np.int32)
        
        return approx
    
    return None

def detect_foot_simple(image):
    """
    IMPROVED Foot Detection for various skin tones
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define multiple skin color ranges
    skin_ranges = [
        # Light to medium skin tones
        ([0, 20, 70], [20, 255, 255]),
        # Medium to dark skin tones  
        ([0, 10, 60], [25, 255, 255]),
        # Wider range for varying lighting
        ([0, 15, 50], [30, 255, 255]),
    ]
    
    combined_mask = None
    
    for lower, upper in skin_ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("DEBUG: No foot contours found in skin detection")
        return None, None
    
    # Find the largest contour (likely the foot)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Skip if contour is too small
    if area < 5000:  # Minimum area threshold
        print(f"DEBUG: Foot contour too small: {area} pixels")
        return None, None
    
    # Create mask
    foot_mask = np.zeros_like(combined_mask)
    cv2.drawContours(foot_mask, [largest_contour], -1, 255, -1)
    
    print(f"DEBUG: Foot detected with area: {area} pixels")
    
    return foot_mask, largest_contour

def measure_foot_simple(image):
    """
    IMPROVED Foot Measurement
    """
    height, width = image.shape[:2]
    
    print(f"MEASUREMENT DEBUG: Processing {width}x{height} image")
    
    # Detect A4 paper
    a4_contour = detect_a4_paper_simple(image)
    
    if a4_contour is None:
        print("MEASUREMENT DEBUG: No A4 detected, estimating from image size")
        
        # If no A4 detected, use image dimensions to estimate
        # Assume typical A4 fills most of the image
        if width > height:  # Landscape orientation
            a4_width_pixels = int(width * 0.8)  # Cast to int
        else:  # Portrait orientation
            a4_width_pixels = int(width * 0.9)  # Cast to int
        
        pixels_per_cm = a4_width_pixels / 21.0
        roi = image
        
        print(f"MEASUREMENT DEBUG: Estimated A4 width: {a4_width_pixels} pixels")
        
    else:
        # Get A4 paper dimensions from contour - boundingRect returns ints
        x, y, w, h = cv2.boundingRect(a4_contour)
        # Ensure they are ints
        x, y, w, h = int(x), int(y), int(w), int(h)
        a4_width_pixels = w
        
        print(f"MEASUREMENT DEBUG: A4 detected: {w}x{h} pixels")
        
        # A4 actual width is 21.0 cm
        pixels_per_cm = a4_width_pixels / 21.0
        
        # Crop to A4 region with some padding - ensure all are ints
        padding = 10
        x1 = int(max(0, x - padding))
        y1 = int(max(0, y - padding))
        x2 = int(min(width, x + w + padding))
        y2 = int(min(height, y + h + padding))
        roi = image[y1:y2, x1:x2]  # Now all indices are integers
    
    print(f"MEASUREMENT DEBUG: Pixels per cm: {pixels_per_cm:.2f}")
    
    # Detect foot in ROI
    foot_mask, foot_contour = detect_foot_simple(roi)
    
    if foot_mask is None or foot_contour is None:
        return None, "Foot not detected. Make sure foot is clearly visible on A4 paper."
    
    # Get foot bounding box
    y_coords, x_coords = np.where(foot_mask > 0)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None, "Could not determine foot boundaries."
    
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    # Calculate foot length (heel to toe) - these are ints from np.where
    foot_length_pixels = int(max_y - min_y)  # Cast to int
    foot_length_cm = foot_length_pixels / pixels_per_cm
    
    # Calculate foot width
    foot_width_pixels = int(max_x - min_x)  # Cast to int
    foot_width_cm = foot_width_pixels / pixels_per_cm
    
    print(f"MEASUREMENT DEBUG: Foot bbox: {foot_width_pixels}x{foot_length_pixels} pixels")
    print(f"MEASUREMENT DEBUG: Foot size: {foot_width_cm:.1f} x {foot_length_cm:.1f} cm")
    
    # Validate foot size (adult feet typically 20-32 cm)
    if foot_length_cm < 15 or foot_length_cm > 35:
        return None, f"Unrealistic foot length ({foot_length_cm:.1f}cm). Normal range: 15-35cm."
    
    return round(foot_length_cm, 2), None

def cm_to_uk_size(foot_length_cm):
    """
    Convert cm to UK shoe size using provided chart
    UK Men's Shoe Size Chart (Pakistan = UK)
    Foot Length (cm) -> UK Size
    24.5 -> 6
    25.4 -> 7
    26.2 -> 8
    27.0 -> 9
    27.9 -> 10
    28.8 -> 11
    29.6 -> 13
    """
    # Define size chart based on your data
    size_chart = [
        (24.0, 24.5, 5.5),
        (24.5, 25.0, 6.0),
        (25.0, 25.4, 6.5),
        (25.4, 25.8, 7.0),
        (25.8, 26.2, 7.5),
        (26.2, 26.6, 8.0),
        (26.6, 27.0, 8.5),
        (27.0, 27.4, 9.0),
        (27.4, 27.9, 9.5),
        (27.9, 28.3, 10.0),
        (28.3, 28.8, 10.5),
        (28.8, 29.2, 11.0),
        (29.2, 29.6, 12.0),
        (29.6, 30.0, 13.0),
        (30.0, 30.5, 14.0),
        (30.5, 31.0, 15.0),
    ]
    
    # Special cases from your chart
    if foot_length_cm >= 29.6:
        # From your chart: 29.6 = 13
        return 13.0
    elif foot_length_cm >= 28.8:
        return 11.0
    elif foot_length_cm >= 27.9:
        return 10.0
    elif foot_length_cm >= 27.0:
        return 9.0
    elif foot_length_cm >= 26.2:
        return 8.0
    elif foot_length_cm >= 25.4:
        return 7.0
    elif foot_length_cm >= 24.5:
        return 6.0
    
    # Use chart for other sizes
    for min_len, max_len, size in size_chart:
        if min_len <= foot_length_cm < max_len:
            return size
    
    # Fallback formula if not in chart
    if foot_length_cm < 24.0:
        return round((foot_length_cm - 22.0) / 0.4 + 4, 0.5)
    else:
        return round((foot_length_cm - 24.0) / 0.4 + 6, 0.5)

def validate_foot_measurement(foot_length_cm):
    """Validate foot measurement"""
    # Adult foot typically 20-32 cm
    if foot_length_cm < 18:
        return False, "Foot too small (likely child's foot or detection error)"
    elif foot_length_cm > 32:
        return False, "Foot too large (likely measurement error)"
    return True, "Valid"

def process_image_measurement(image_bytes, user_id):
    """Process image measurement (runs in thread pool)"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image file"}
        
        # Check image dimensions - img.shape returns ints, safe
        if img.shape[0] < 300 or img.shape[1] < 300:
            return {"error": "Image too small. Minimum 300x300 pixels."}
        
        print(f"Processing image {img.shape[1]}x{img.shape[0]}")
        
        # Measure foot length
        foot_length_cm, error = measure_foot_simple(img)
        
        if error:
            return {"error": error}
        
        # Validate measurement
        is_valid, validation_msg = validate_foot_measurement(foot_length_cm)
        if not is_valid:
            return {"error": validation_msg}
        
        # Convert to UK size
        uk_size = cm_to_uk_size(foot_length_cm)
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get previous foot size
        cursor.execute("SELECT foot_size FROM users WHERE id = %s", (user_id,))
        previous_result = cursor.fetchone()
        previous_size = previous_result[0] if previous_result else None
        
        # Update foot_size
        cursor.execute("""
            UPDATE users 
            SET foot_size = %s, updated_at = %s
            WHERE id = %s
            RETURNING first_name, last_name, email
        """, (foot_length_cm, datetime.now(), user_id))
        
        updated_user = cursor.fetchone()
        conn.commit()
        conn.close()
        
        if not updated_user:
            return {"error": "User not found"}
        
        # Calculate confidence based on measurement validity
        if 22 <= foot_length_cm <= 30:  # Most common adult foot range
            confidence = 0.95
        elif 20 <= foot_length_cm < 22 or 30 < foot_length_cm <= 32:
            confidence = 0.75
        else:
            confidence = 0.5
        
        return {
            "user_id": user_id,
            "user_name": f"{updated_user[0]} {updated_user[1]}",
            "foot_length_cm": foot_length_cm,
            "previous_foot_size_cm": previous_size,
            "recommended_uk_size": uk_size,
            "confidence": round(confidence, 2),
            "measurement_time": datetime.now().isoformat(),
            "message": "Foot measurement successfully updated"
        }
        
    except Exception as e:
        print(f"Processing error: {e}")
        return {"error": str(e)}

# ========== API ENDPOINTS ==========
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Foot Measurement API",
        "version": "1.0.0",
        "description": "Measure foot size from images using A4 paper as reference",
        "endpoints": {
            "measure_foot": "POST /measure-foot (requires JWT token)",
            "get_user_footsize": "GET /user/{user_id}/foot-size (requires JWT token)",
            "health": "GET /health",
            "health_full": "GET /health/full"
        },
        "note": "Use FYP Auth API (https://fyp-auth-api.onrender.com) for signup/login"
    }

@app.get("/health")
async def health_check():
    """Fast health check without database query"""
    return {
        "status": "healthy",
        "api": "Foot Measurement API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health/full")
async def health_check_full():
    """Full health check with DB"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        conn.close()
        return {
            "status": "healthy",
            "database": "connected",
            "users_count": user_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/measure-foot")
async def measure_foot(
    file: UploadFile = File(...),
    user_id: int = Depends(verify_token)
):
    """
    Measure foot from uploaded image and update database
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG)")
        
        # Read image
        contents = await file.read()
        
        # Run image processing in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            process_image_measurement, 
            contents, user_id
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Measurement failed: {str(e)}")

@app.get("/user/{user_id}/foot-size")
async def get_user_footsize(
    user_id: int,
    current_user_id: int = Depends(verify_token)
):
    """
    Get user's current foot size measurement
    """
    if current_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied. You can only access your own data.")
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                id, first_name, last_name, email,
                foot_size, updated_at as last_measurement,
                age, weight, purpose
            FROM users 
            WHERE id = %s
        """, (user_id,))
        
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user['foot_size']:
            uk_size = cm_to_uk_size(user['foot_size'])
            user['recommended_uk_size'] = uk_size
        
        return user
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user data: {str(e)}")
    finally:
        if conn:
            conn.close()

@app.get("/test/image")
async def test_image_processing():
    """Test endpoint for image processing"""
    test_images = ["foot.png", "test.jpg", "sample.jpg"]
    
    results = []
    for img_name in test_images:
        if os.path.exists(img_name):
            try:
                img = cv2.imread(img_name)
                if img is not None:
                    # Test A4 detection
                    a4_contour = detect_a4_paper_simple(img)
                    
                    # Test foot detection
                    foot_mask, _ = detect_foot_simple(img)
                    
                    # Try measurement
                    foot_length, error = measure_foot_simple(img) if foot_mask is not None else (None, "No foot")
                    
                    results.append({
                        "image": img_name,
                        "size": f"{img.shape[1]}x{img.shape[0]}",
                        "a4_detected": a4_contour is not None,
                        "foot_detected": foot_mask is not None,
                        "foot_length_cm": foot_length,
                        "error": error
                    })
            except Exception as e:
                results.append({
                    "image": img_name,
                    "error": str(e)
                })
    
    return {
        "test_results": results,
        "available_images": [f for f in test_images if os.path.exists(f)],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True
    )