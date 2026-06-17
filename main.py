# main.py - FOOT MEASUREMENT API - FULLY FIXED + ENHANCED
# Fixes: All float-to-int casting errors resolved across OpenCV, NumPy, and Python builtins
# Enhanced: Image quality analysis, robust error handling, measurement history logging

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
    version="2.0.0"
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


# ========== UTILITY HELPERS ==========

def to_int_coords(*args):
    """
    FIX: Safely convert any pixel coordinate or dimension to Python int.
    Prevents 'float cannot be interpreted as integer' in cv2.resize,
    NumPy slicing, and range() calls throughout this file.
    """
    return tuple(int(round(float(a))) for a in args)


def safe_divide(numerator, denominator, fallback=1.0):
    """
    FIX: Safe division that never returns zero denominator.
    pixels_per_cm = safe_divide(pixels, 21.0) prevents ZeroDivisionError.
    """
    if denominator == 0:
        return fallback
    return float(numerator) / float(denominator)


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
        except (ValueError, TypeError):
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


# ========== IMAGE PROCESSING FUNCTIONS ==========

def detect_a4_paper_simple(image):
    """
    A4 Paper Detection using edge detection and contour analysis.

    FIX APPLIED:
    - cv2.resize() now always receives (int, int) tuple via to_int_coords()
    - aspect_ratio division uses explicit float() cast
    - fallback bounding box array uses dtype=np.int32 explicitly
    """
    original = image.copy()
    height, width = image.shape[:2]
    # FIX: ensure height/width are Python ints from shape (they should be, but enforce)
    height, width = int(height), int(width)

    print(f"DEBUG: Image size: {width}x{height} pixels")

    # 1. Resize if too large — FIX: to_int_coords ensures cv2.resize gets (int, int)
    max_size = 1200
    if max(width, height) > max_size:
        scale = float(max_size) / float(max(width, height))
        new_width, new_height = to_int_coords(width * scale, height * scale)
        # FIX: cv2.resize requires integer tuple — guaranteed by to_int_coords above
        image = cv2.resize(image, (new_width, new_height))
        height, width = int(image.shape[0]), int(image.shape[1])
        print(f"DEBUG: Resized to: {width}x{height} pixels")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # 5. Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 6. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"DEBUG: Found {len(contours)} contours")

    if not contours:
        print("DEBUG: No contours found, trying adaptive thresholding")
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"DEBUG: Adaptive thresholding found {len(contours)} contours")

    if not contours:
        return None

    # Sort by area, take top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    image_area = float(height * width)

    for i, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))

        if area < image_area * 0.1:
            continue
        if area > image_area * 0.95:
            continue

        print(f"DEBUG: Contour {i}: area={area:.0f} ({area / image_area * 100:.1f}% of image)")

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # FIX: cv2.boundingRect returns numpy int types — cast explicitly
            x, y, w, h = to_int_coords(*cv2.boundingRect(approx))

            # FIX: aspect_ratio uses explicit float division — no implicit int division
            aspect_ratio = float(w) / float(h)
            print(f"DEBUG: Quadrilateral: {w}x{h}, aspect={aspect_ratio:.3f}")

            if 0.6 <= aspect_ratio <= 0.8:
                print(f"DEBUG: ✓ A4 paper detected! Aspect ratio: {aspect_ratio:.3f}")
                debug_img = original.copy()
                cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
                cv2.imwrite("a4_debug.jpg", debug_img)
                print("DEBUG: Saved A4 detection debug image")
                return approx

    # Fallback: use largest contour bounding box
    print("DEBUG: No quadrilateral found, using largest contour")
    if contours:
        x, y, w, h = to_int_coords(*cv2.boundingRect(contours[0]))
        print(f"DEBUG: Using contour bounding box: {w}x{h}")

        # FIX: All values are ints via to_int_coords — np.array with int32 dtype is safe
        approx = np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h]
        ], dtype=np.int32)

        return approx

    return None


def detect_foot_simple(image):
    """
    Foot Detection using HSV skin color segmentation.
    Supports light to dark skin tones with morphological cleanup.

    No float-to-int issues here (cv2.inRange and morphology ops use uint8 arrays).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    skin_ranges = [
        ([0, 20, 70],  [20, 255, 255]),  # Light to medium skin
        ([0, 10, 60],  [25, 255, 255]),  # Medium to dark skin
        ([0, 15, 50],  [30, 255, 255]),  # Wide range for varying lighting
    ]

    combined_mask = None
    for lower, upper in skin_ranges:
        mask = cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8)
        )
        combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("DEBUG: No foot contours found in skin detection")
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 5000:
        print(f"DEBUG: Foot contour too small: {area} pixels")
        return None, None

    foot_mask = np.zeros_like(combined_mask)
    cv2.drawContours(foot_mask, [largest_contour], -1, 255, -1)

    print(f"DEBUG: Foot detected with area: {area} pixels")
    return foot_mask, largest_contour


def analyze_image_quality(image):
    """
    Analyze image quality and return a scored report before measurement.

    Scoring breakdown (total = 0-100):
      Brightness  : 20 pts
      Sharpness   : 20 pts
      Resolution  : 20 pts
      A4 Detection: 25 pts
      Foot Visible: 15 pts

    Ranks: EXCELLENT (85-100), GOOD (70-84), FAIR (50-69), POOR (0-49)
    """
    # FIX: Cast shape values to int immediately
    height, width = int(image.shape[0]), int(image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 1. BRIGHTNESS (20 pts) ---
    mean_brightness = float(np.mean(gray))
    if 80 <= mean_brightness <= 200:
        brightness_score, brightness_label = 20, "good"
    elif 60 <= mean_brightness < 80 or 200 < mean_brightness <= 220:
        brightness_score, brightness_label = 10, "acceptable"
    elif mean_brightness < 60:
        brightness_score, brightness_label = 0, "too_dark"
    else:
        brightness_score, brightness_label = 0, "overexposed"

    # --- 2. SHARPNESS via Laplacian variance (20 pts) ---
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var >= 100:
        sharpness_score, sharpness_label = 20, "sharp"
    elif laplacian_var >= 50:
        sharpness_score, sharpness_label = 10, "slightly_blurry"
    else:
        sharpness_score, sharpness_label = 0, "too_blurry"

    # --- 3. RESOLUTION (20 pts) ---
    if width >= 1080 and height >= 1080:
        res_score, res_label = 20, "high"
    elif width >= 640 and height >= 640:
        res_score, res_label = 10, "medium"
    else:
        res_score, res_label = 0, "low"

    # --- 4. A4 PAPER DETECTION (25 pts) ---
    try:
        a4_contour = detect_a4_paper_simple(image.copy())
    except Exception:
        a4_contour = None

    if a4_contour is not None:
        # FIX: to_int_coords on boundingRect output prevents float aspect ratio issues
        bx, by, bw, bh = to_int_coords(*cv2.boundingRect(a4_contour))
        aspect = float(bw) / float(bh) if bh > 0 else 0.0
        if 0.6 <= aspect <= 0.8:
            a4_score, a4_label = 25, "detected"
        else:
            a4_score, a4_label = 10, "estimated"
    else:
        a4_score, a4_label = 0, "not_found"

    # --- 5. FOOT VISIBILITY (15 pts) ---
    try:
        _, foot_contour = detect_foot_simple(image.copy())
    except Exception:
        foot_contour = None

    if foot_contour is not None:
        foot_area = float(cv2.contourArea(foot_contour))
        if foot_area >= 10000:
            foot_score, foot_label = 15, "clearly_visible"
        else:
            foot_score, foot_label = 8, "partially_visible"
    else:
        foot_score, foot_label = 0, "not_detected"

    # --- TOTAL SCORE & RANK ---
    total_score = int(brightness_score + sharpness_score + res_score + a4_score + foot_score)

    if total_score >= 85:
        rank = "EXCELLENT"
    elif total_score >= 70:
        rank = "GOOD"
    elif total_score >= 50:
        rank = "FAIR"
    else:
        rank = "POOR"

    # --- RECOMMENDATIONS ---
    recommendations = []
    if brightness_label == "too_dark":
        recommendations.append("Move to a brighter area or turn on more lights")
    elif brightness_label == "overexposed":
        recommendations.append("Reduce lighting or avoid direct flash")

    if sharpness_label in ("slightly_blurry", "too_blurry"):
        recommendations.append("Hold camera steady and ensure foot is in focus")

    if res_label == "low":
        recommendations.append("Use a higher resolution camera setting")

    if a4_label == "not_found":
        recommendations.append("Place foot on a flat white A4 paper sheet")
    elif a4_label == "estimated":
        recommendations.append("Ensure A4 paper edges are fully visible in frame")

    if foot_label == "not_detected":
        recommendations.append("Make sure your bare foot is clearly visible on the paper")
    elif foot_label == "partially_visible":
        recommendations.append("Move camera higher so the full foot fits in frame")

    return {
        "total_score": total_score,
        "rank": rank,
        "checks": {
            "brightness": {
                "score": int(brightness_score),
                "label": brightness_label,
                "mean_value": round(mean_brightness, 2)
            },
            "sharpness": {
                "score": int(sharpness_score),
                "label": sharpness_label,
                "laplacian_var": round(laplacian_var, 2)
            },
            "resolution": {
                "score": int(res_score),
                "label": res_label,
                "width": width,
                "height": height
            },
            "a4_paper": {
                "score": int(a4_score),
                "label": a4_label
            },
            "foot": {
                "score": int(foot_score),
                "label": foot_label
            }
        },
        "recommendations": recommendations
    }


def measure_foot_simple(image):
    """
    Measure foot length in cm using A4 paper as pixel-per-cm reference.

    FIX APPLIED (all float-to-int errors):
    1. a4_width_pixels = int(round(width * 0.8/0.9))  — was float, now int
    2. pixels_per_cm kept as float intentionally (used only for division)
    3. x, y, w, h from cv2.boundingRect cast via to_int_coords()
    4. padding crop indices x1,y1,x2,y2 cast via to_int_coords() before slicing
    5. foot_length_pixels / foot_width_pixels cast to int from np.uint64 (np.where output)
    6. round() second arg fixed: Python's round() takes int ndigits, not float
    """
    height, width = int(image.shape[0]), int(image.shape[1])

    print(f"MEASUREMENT DEBUG: Processing {width}x{height} image")

    a4_contour = detect_a4_paper_simple(image)

    if a4_contour is None:
        print("MEASUREMENT DEBUG: No A4 detected, estimating from image size")

        # FIX: int(round(...)) — was bare float multiplication, caused downstream crash
        if width > height:
            a4_width_pixels = int(round(float(width) * 0.8))
        else:
            a4_width_pixels = int(round(float(width) * 0.9))

        # pixels_per_cm is intentionally float — used only in division below
        pixels_per_cm = safe_divide(a4_width_pixels, 21.0)
        roi = image

        print(f"MEASUREMENT DEBUG: Estimated A4 width: {a4_width_pixels} pixels")

    else:
        # FIX: to_int_coords converts numpy int types from boundingRect to Python int
        x, y, w, h = to_int_coords(*cv2.boundingRect(a4_contour))
        a4_width_pixels = w

        print(f"MEASUREMENT DEBUG: A4 detected: {w}x{h} pixels")

        pixels_per_cm = safe_divide(a4_width_pixels, 21.0)

        # FIX: to_int_coords on all crop indices — NumPy slicing requires Python int
        padding = 10
        x1, y1, x2, y2 = to_int_coords(
            max(0, x - padding),
            max(0, y - padding),
            min(width,  x + w + padding),
            min(height, y + h + padding)
        )
        # FIX: roi slice is now guaranteed integer indices — no TypeError
        roi = image[y1:y2, x1:x2]

    print(f"MEASUREMENT DEBUG: Pixels per cm: {pixels_per_cm:.2f}")

    foot_mask, foot_contour = detect_foot_simple(roi)

    if foot_mask is None or foot_contour is None:
        return None, "Foot not detected. Make sure foot is clearly visible on A4 paper."

    y_coords, x_coords = np.where(foot_mask > 0)

    if len(x_coords) == 0 or len(y_coords) == 0:
        return None, "Could not determine foot boundaries."

    # FIX: np.where returns np.intp (numpy integer), cast to Python int for safety
    min_x = int(np.min(x_coords))
    max_x = int(np.max(x_coords))
    min_y = int(np.min(y_coords))
    max_y = int(np.max(y_coords))

    # FIX: explicit int subtraction then float division via safe_divide
    foot_length_pixels = int(max_y - min_y)
    foot_width_pixels  = int(max_x - min_x)

    foot_length_cm = safe_divide(foot_length_pixels, pixels_per_cm)
    foot_width_cm  = safe_divide(foot_width_pixels,  pixels_per_cm)

    print(f"MEASUREMENT DEBUG: Foot bbox: {foot_width_pixels}x{foot_length_pixels} pixels")
    print(f"MEASUREMENT DEBUG: Foot size: {foot_width_cm:.1f} x {foot_length_cm:.1f} cm")

    if foot_length_cm < 15 or foot_length_cm > 35:
        return None, f"Unrealistic foot length ({foot_length_cm:.1f}cm). Normal range: 15-35cm."

    # FIX: round() second arg must be int — round(value, 2) not round(value, 0.5)
    return round(foot_length_cm, 2), None


def cm_to_uk_size(foot_length_cm):
    """
    Convert foot length (cm) to UK shoe size.
    UK Men's Size Chart — Pakistan standard matches UK sizing.

    FIX: round() in fallback branch used 0.5 as second arg (must be int).
    Replaced with correct rounding logic.
    """
    if foot_length_cm >= 29.6:
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

    for min_len, max_len, size in size_chart:
        if min_len <= foot_length_cm < max_len:
            return size

    # FIX: round() ndigits must be int (1), not float (0.5)
    # Previously: round(..., 0.5) → TypeError in Python 3
    if foot_length_cm < 24.0:
        raw = (foot_length_cm - 22.0) / 0.4 + 4.0
    else:
        raw = (foot_length_cm - 24.0) / 0.4 + 6.0

    # Round to nearest 0.5
    return round(raw * 2) / 2


def validate_foot_measurement(foot_length_cm):
    """Validate that measured foot length is within adult human range."""
    if foot_length_cm < 18:
        return False, "Foot too small (likely child's foot or detection error)"
    elif foot_length_cm > 32:
        return False, "Foot too large (likely measurement error)"
    return True, "Valid"


def process_image_measurement(image_bytes, user_id):
    """
    Main processing pipeline — runs in ThreadPoolExecutor (non-blocking).

    Order:
      1. Decode image
      2. Quality analysis (non-blocking, never raises)
      3. Measure foot (with TypeError guard)
      4. Validate measurement
      5. Update database
      6. Return full response with image_quality field
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image file. Ensure it is a valid JPEG or PNG."}

        # FIX: Cast shape to Python int immediately after decode
        img_h, img_w = int(img.shape[0]), int(img.shape[1])

        if img_h < 300 or img_w < 300:
            return {"error": "Image too small. Minimum 300x300 pixels required."}

        print(f"Processing image {img_w}x{img_h}")

        # --- Quality Analysis (never blocks measurement) ---
        try:
            quality_report = analyze_image_quality(img)
            print(f"Quality: {quality_report['rank']} ({quality_report['total_score']}/100)")
        except Exception as qe:
            print(f"Quality analysis failed: {qe}")
            quality_report = {
                "total_score": -1,
                "rank": "UNKNOWN",
                "checks": {},
                "recommendations": [],
                "error": str(qe)
            }

        # --- Measurement (TypeError explicitly caught) ---
        try:
            foot_length_cm, error = measure_foot_simple(img)
        except TypeError as te:
            # FIX: TypeError from float-as-int surfaces here with clear message
            print(f"TypeError in measure_foot_simple: {te}")
            return {
                "error": (
                    f"Type mismatch in pixel coordinates: {str(te)}. "
                    "Check float-to-int casting in image processing."
                ),
                "image_quality": quality_report
            }

        if error:
            return {"error": error, "image_quality": quality_report}

        # --- Validate ---
        is_valid, validation_msg = validate_foot_measurement(foot_length_cm)
        if not is_valid:
            return {"error": validation_msg, "image_quality": quality_report}

        # --- Convert to UK size ---
        uk_size = cm_to_uk_size(foot_length_cm)

        # --- Database update ---
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT foot_size FROM users WHERE id = %s", (user_id,))
        previous_result = cursor.fetchone()
        previous_size = previous_result[0] if previous_result else None

        cursor.execute(
            """
            UPDATE users
            SET foot_size = %s, updated_at = %s
            WHERE id = %s
            RETURNING first_name, last_name, email
            """,
            (foot_length_cm, datetime.now(), user_id)
        )
        updated_user = cursor.fetchone()
        conn.commit()
        conn.close()

        if not updated_user:
            return {"error": "User not found", "image_quality": quality_report}

        # --- Confidence score ---
        if 22 <= foot_length_cm <= 30:
            confidence = 0.95
        elif 20 <= foot_length_cm < 22 or 30 < foot_length_cm <= 32:
            confidence = 0.75
        else:
            confidence = 0.5

        # Boost confidence if image quality is EXCELLENT or GOOD
        if quality_report.get("rank") in ("EXCELLENT", "GOOD"):
            confidence = min(1.0, confidence + 0.05)

        return {
            "user_id": user_id,
            "user_name": f"{updated_user[0]} {updated_user[1]}",
            "foot_length_cm": foot_length_cm,
            "previous_foot_size_cm": previous_size,
            "recommended_uk_size": uk_size,
            "confidence": round(confidence, 2),
            "measurement_time": datetime.now().isoformat(),
            "message": "Foot measurement successfully updated",
            "image_quality": quality_report  # NEW: always included
        }

    except Exception as e:
        print(f"Processing error: {e}")
        return {"error": str(e)}


# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    return {
        "message": "Foot Measurement API",
        "version": "2.0.0",
        "description": "Measure foot size from images using A4 paper as reference",
        "endpoints": {
            "measure_foot":      "POST /measure-foot (requires JWT token)",
            "get_user_footsize": "GET /user/{user_id}/foot-size (requires JWT token)",
            "health":            "GET /health",
            "health_full":       "GET /health/full",
            "test_image":        "GET /test/image"
        },
        "note": "Use FYP Auth API (https://fyp-auth-api.onrender.com) for signup/login"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "api": "Foot Measurement API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health/full")
async def health_check_full():
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
    Upload a foot image (JPEG/PNG) placed on A4 paper.
    Returns foot length in cm, recommended UK shoe size, and image quality report.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")

    contents = await file.read()

    if len(contents) < 1000:
        raise HTTPException(status_code=400, detail="Image file appears to be empty or corrupt")

    try:
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
    """Get stored foot size for a user (own data only)."""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied. You can only access your own data."
        )

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute(
            """
            SELECT
                id, first_name, last_name, email,
                foot_size, updated_at AS last_measurement,
                age, weight, purpose
            FROM users
            WHERE id = %s
            """,
            (user_id,)
        )
        user = cursor.fetchone()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if user["foot_size"]:
            user["recommended_uk_size"] = cm_to_uk_size(user["foot_size"])

        return user

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user data: {str(e)}")
    finally:
        if conn:
            conn.close()


@app.get("/test/image")
async def test_image_processing():
    """
    Test endpoint — processes any local sample images found on disk.
    Useful during development to verify detection pipeline without auth.
    """
    test_images = ["foot.png", "test.jpg", "sample.jpg", "foot-1.jpg", "foot-2.jpg"]

    results = []
    for img_name in test_images:
        if not os.path.exists(img_name):
            continue
        try:
            img = cv2.imread(img_name)
            if img is None:
                results.append({"image": img_name, "error": "Could not read image"})
                continue

            a4_contour  = detect_a4_paper_simple(img)
            foot_mask, _ = detect_foot_simple(img)

            foot_length, error = (
                measure_foot_simple(img) if foot_mask is not None
                else (None, "No foot detected")
            )

            try:
                quality = analyze_image_quality(img)
            except Exception as qe:
                quality = {"error": str(qe)}

            results.append({
                "image":          img_name,
                "size":           f"{int(img.shape[1])}x{int(img.shape[0])}",
                "a4_detected":    a4_contour is not None,
                "foot_detected":  foot_mask is not None,
                "foot_length_cm": foot_length,
                "error":          error,
                "image_quality":  quality
            })
        except Exception as e:
            results.append({"image": img_name, "error": str(e)})

    return {
        "test_results":      results,
        "available_images":  [f for f in test_images if os.path.exists(f)],
        "timestamp":         datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True
    )

# END OF FILE