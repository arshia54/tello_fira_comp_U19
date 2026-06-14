# Myat.py
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation
from pupil_apriltags import Detector
from collections import deque, defaultdict
import threading
from scipy import ndimage
import json
import logging

# ==================== ADVANCED SYSTEM PARAMETERS ====================
frame_size = [960, 720]
april_cm = 1000.0
april_focal = [680, 680]  
CAMERA_CENTER = (int(frame_size[0]/2), int(frame_size[1]/2))
TAG_SIZE = 0.016  

# ==================== MULTI-STRATEGY DETECTOR CONFIGURATION ====================
detector_configs = {
    'FAST': {
        'families': 'tag36h11',
        'nthreads': 8,
        'quad_decimate': 2.0,  # Faster processing
        'quad_sigma': 0.4,
        'refine_edges': 1,
        'decode_sharpening': 0.3,
        'debug': 0
    },
    'PRECISE': {
        'families': 'tag36h11', 
        'nthreads': 4,
        'quad_decimate': 1.0,  # Full resolution
        'quad_sigma': 0.6,
        'refine_edges': 2,
        'decode_sharpening': 0.8,
        'debug': 0
    },
    'LOW_LIGHT': {
        'families': 'tag36h11',
        'nthreads': 6,
        'quad_decimate': 1.5,
        'quad_sigma': 0.8,
        'refine_edges': 1,
        'decode_sharpening': 1.0,  # Enhanced sharpening
        'debug': 0
    }
}

# Initialize multiple detectors
detectors = {
    name: Detector(**config) 
    for name, config in detector_configs.items()
}

# ==================== GLOBAL TRACKING SYSTEMS ====================
tag_kalman_states = {}
tag_tracking_history = defaultdict(lambda: deque(maxlen=20))
tag_velocity_estimates = {}
detection_performance = {
    'frame_count': 0,
    'successful_detections': 0,
    'detection_times': deque(maxlen=100),
    'confidence_history': defaultdict(lambda: deque(maxlen=50)),
    'detector_performance': {name: deque(maxlen=50) for name in detector_configs.keys()}
}

adaptive_settings = {
    'current_detector': 'FAST',
    'last_lighting_assessment': 0,
    'average_brightness': 127,
    'detection_quality': 1.0,
    'frame_processing_time': deque(maxlen=30)
}

# ==================== ADVANCED IMAGE PREPROCESSING ====================
def adaptive_preprocess_image(img, lighting_conditions='auto'):
    """Multi-stage adaptive image preprocessing"""
    if lighting_conditions == 'auto':
        lighting_conditions = assess_lighting_conditions(img)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Lighting-adaptive processing
    if lighting_conditions == 'low_light':
        # Enhanced low-light processing
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        # Mild sharpening
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
    elif lighting_conditions == 'high_contrast':
        # Handle overexposure
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # Conservative contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(normalized)
        sharpened = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
        
    else:  # normal lighting
        # Balanced processing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.medianBlur(enhanced, 3)
        # Adaptive sharpening
        kernel = np.array([[-0.5, -1, -0.5], [-1, 7, -1], [-0.5, -1, -0.5]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def assess_lighting_conditions(img):
    """Assess current lighting conditions for adaptive processing"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    adaptive_settings['average_brightness'] = brightness
    adaptive_settings['last_lighting_assessment'] = time.time()
    
    if brightness < 60:
        return 'low_light'
    elif brightness > 200 and contrast > 60:
        return 'high_contrast'
    else:
        return 'normal'

# ==================== INTELLIGENT DETECTOR SELECTION ====================
def select_optimal_detector(img, previous_performance):
    """Dynamically select the best detector for current conditions"""
    brightness = adaptive_settings['average_brightness']
    avg_processing_time = np.mean(adaptive_settings['frame_processing_time']) if adaptive_settings['frame_processing_time'] else 33
    
    # Performance-based selection
    detector_scores = {}
    
    for detector_name in detectors.keys():
        perf_data = detection_performance['detector_performance'][detector_name]
        success_rate = np.mean([p[0] for p in perf_data]) if perf_data else 0.5
        avg_confidence = np.mean([p[1] for p in perf_data]) if perf_data else 0.5
        
        # Base score
        score = success_rate * 0.6 + avg_confidence * 0.4
        
        # Condition adjustments
        if detector_name == 'LOW_LIGHT' and brightness < 80:
            score *= 1.3
        elif detector_name == 'PRECISE' and brightness > 100 and avg_processing_time < 25:
            score *= 1.2
        elif detector_name == 'FAST' and avg_processing_time > 30:
            score *= 1.4
            
        detector_scores[detector_name] = score
    
    best_detector = max(detector_scores, key=detector_scores.get)
    adaptive_settings['current_detector'] = best_detector
    
    return best_detector

# ==================== ENHANCED TAG DETECTION ====================
def get_tags(img, enable_preprocessing=True, enable_tracking=True, enable_multi_detector=True):
    """ULTRA-ENHANCED AprilTag detection with multi-strategy approach"""
    start_time = time.time()
    detection_performance['frame_count'] += 1
    
    # Adaptive detector selection
    if enable_multi_detector:
        detector_name = select_optimal_detector(img, detection_performance)
        detector = detectors[detector_name]
    else:
        detector_name = 'PRECISE'
        detector = detectors['PRECISE']
    
    # Advanced preprocessing
    if enable_preprocessing:
        lighting_condition = assess_lighting_conditions(img)
        processed_img = adaptive_preprocess_image(img, lighting_condition)
    else:
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Multi-stage detection with fallback
    tags = []
    detection_attempts = []
    
    # Primary detection
    primary_tags = detector.detect(
        processed_img,
        estimate_tag_pose=True,
        camera_params=(april_focal[0], april_focal[1], CAMERA_CENTER[0], CAMERA_CENTER[1]),
        tag_size=TAG_SIZE
    )
    detection_attempts.append((detector_name, primary_tags))
    
    # Fallback detection if primary fails
    if len(primary_tags) == 0 and enable_multi_detector:
        for fallback_name, fallback_detector in detectors.items():
            if fallback_name != detector_name:
                fallback_tags = fallback_detector.detect(
                    processed_img,
                    estimate_tag_pose=True,
                    camera_params=(april_focal[0], april_focal[1], CAMERA_CENTER[0], CAMERA_CENTER[1]),
                    tag_size=TAG_SIZE
                )
                detection_attempts.append((fallback_name, fallback_tags))
                if len(fallback_tags) > 0:
                    break
    
    # Process all detected tags
    res = []
    current_time = time.time()
    
    for attempt_name, detected_tags in detection_attempts:
        for tag in detected_tags:
            # Enhanced pose estimation
            r = Rotation.from_matrix(tag.pose_R)
            angles = r.as_euler("zyx", degrees=True)
            T = tag.pose_t.reshape((3)).tolist()
            
            # Scale and convert to integers
            for i in range(3):
                T[i] = int(T[i] * april_cm)
            
            # Advanced confidence calculation
            confidence = calculate_enhanced_confidence(tag, processed_img, img)
            
            # Apply advanced tracking
            if enable_tracking:
                tracked_data = apply_advanced_tracking(tag.tag_id, T, angles, current_time, confidence)
                if tracked_data:
                    T, angles, confidence = tracked_data
            
            # Validate detection
            if is_valid_detection(tag, T, angles, confidence):
                tag_data = [
                    tag.tag_id, 
                    T[0], T[1], T[2],
                    int(angles[1]),  # Pitch angle for drone control
                    int(tag.center[0]),
                    int(tag.center[1]),
                    confidence,
                    attempt_name  # Detector used
                ]
                res.append(tag_data)
                
                # Update performance metrics
                detection_performance['successful_detections'] += 1
                detection_performance['confidence_history'][tag.tag_id].append(confidence)
                detection_performance['detector_performance'][attempt_name].append((1, confidence))
    
    # Update performance timing
    processing_time = (time.time() - start_time) * 1000
    detection_performance['detection_times'].append(processing_time)
    adaptive_settings['frame_processing_time'].append(processing_time)
    
    return res

# ==================== ENHANCED CONFIDENCE CALCULATION ====================
def calculate_enhanced_confidence(tag, processed_img, original_img):
    """Multi-factor confidence calculation"""
    confidence = 0.0
    
    # 1. Basic tag structure (20%)
    if hasattr(tag, 'corners') and len(tag.corners) == 4:
        confidence += 0.2
    
    # 2. Area-based confidence (20%)
    area = calculate_tag_area(tag.corners)
    area_confidence = min(area / 5000.0, 1.0)  # Normalize by reasonable area
    confidence += area_confidence * 0.2
    
    # 3. Edge quality assessment (25%)
    edge_quality = assess_enhanced_edge_quality(processed_img, tag.corners)
    confidence += edge_quality * 0.25
    
    # 4. Pose stability (15%)
    pose_stability = assess_pose_stability(tag)
    confidence += pose_stability * 0.15
    
    # 5. Historical consistency (20%)
    historical_consistency = assess_historical_consistency(tag.tag_id, tag.center)
    confidence += historical_consistency * 0.2
    
    return min(confidence, 1.0)

def calculate_tag_area(corners):
    """Calculate polygon area using shoelace formula"""
    if len(corners) != 4:
        return 0
    
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def assess_enhanced_edge_quality(img, corners):
    """Comprehensive edge quality assessment"""
    try:
        mask = np.zeros(img.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(corners), 255)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge consistency within tag region
        mean_gradient = np.mean(magnitude[mask > 0])
        std_gradient = np.std(magnitude[mask > 0])
        
        # High mean + low std = good edges
        edge_quality = (mean_gradient / 50.0) * (1.0 - min(std_gradient / mean_gradient, 1.0) if mean_gradient > 0 else 0)
        
        return min(edge_quality, 1.0)
    except:
        return 0.3

def assess_pose_stability(tag):
    """Assess pose estimation stability"""
    # Check if pose values are reasonable
    if hasattr(tag, 'pose_t'):
        t_norm = np.linalg.norm(tag.pose_t)
        # Reasonable distance range (10cm to 5m)
        if 0.1 < t_norm < 5.0:
            return 0.8
    return 0.3

def assess_historical_consistency(tag_id, current_center):
    """Check consistency with previous detections"""
    history = tag_tracking_history[tag_id]
    if len(history) < 2:
        return 0.5
    
    # Check if current position is consistent with recent history
    recent_centers = [pos for pos in list(history)[-3:]]
    distances = [np.linalg.norm(np.array(current_center) - np.array(pos)) for pos in recent_centers]
    avg_distance = np.mean(distances)
    
    # Smaller distance = more consistent
    consistency = 1.0 - min(avg_distance / 50.0, 1.0)
    return consistency

# ==================== ADVANCED TRACKING SYSTEM ====================
def apply_advanced_tracking(tag_id, position, angles, current_time, confidence):
    """Enhanced tracking with Kalman filtering and velocity prediction"""
    # Initialize Kalman filter if needed
    if tag_id not in tag_kalman_states:
        tag_kalman_states[tag_id] = {
            'position': position.copy(),
            'velocity': [0, 0, 0],
            'angle': angles[1],
            'angular_velocity': 0,
            'last_update': current_time,
            'confidence': confidence
        }
        return position, angles, confidence
    
    # Get previous state
    prev_state = tag_kalman_states[tag_id]
    dt = current_time - prev_state['last_update']
    
    if dt <= 0:
        dt = 0.033  # Assume 30fps
    
    # Simple Kalman-like prediction
    predicted_position = [
        prev_state['position'][i] + prev_state['velocity'][i] * dt 
        for i in range(3)
    ]
    
    predicted_angle = prev_state['angle'] + prev_state['angular_velocity'] * dt
    
    # Blend with new measurement based on confidence
    blend_alpha = min(confidence * 0.8, 0.7)  # Higher confidence = more weight to measurement
    
    smoothed_position = [
        int(predicted_position[i] * (1 - blend_alpha) + position[i] * blend_alpha)
        for i in range(3)
    ]
    
    smoothed_angle = int(predicted_angle * (1 - blend_alpha) + angles[1] * blend_alpha)
    smoothed_angles = [angles[0], smoothed_angle, angles[2]]
    
    # Update velocity estimates
    new_velocity = [
        (smoothed_position[i] - prev_state['position'][i]) / dt 
        for i in range(3)
    ]
    
    new_angular_velocity = (smoothed_angle - prev_state['angle']) / dt
    
    # Update state
    tag_kalman_states[tag_id] = {
        'position': smoothed_position,
        'velocity': new_velocity,
        'angle': smoothed_angle,
        'angular_velocity': new_angular_velocity,
        'last_update': current_time,
        'confidence': confidence
    }
    
    # Update tracking history
    tag_tracking_history[tag_id].append((smoothed_position[0], smoothed_position[1]))
    
    return smoothed_position, smoothed_angles, confidence

# ==================== VALIDATION AND FILTERING ====================
def is_valid_detection(tag, position, angles, confidence):
    """Comprehensive detection validation"""
    # Confidence threshold
    if confidence < 0.3:
        return False
    
    # Position sanity checks
    if any(abs(p) > 1000 for p in position):  # Unreasonable position
        return False
    
    # Angle sanity checks
    if any(abs(a) > 180 for a in angles):  # Invalid angles
        return False
    
    # Center point within image bounds
    if not (0 <= tag.center[0] < frame_size[0] and 0 <= tag.center[1] < frame_size[1]):
        return False
    
    return True

# ==================== ENHANCED VISUALIZATION ====================
def draw_enhanced_tags(img, tags, show_metrics=True):
    """Advanced visualization with performance metrics"""
    display_img = img.copy()
    
    for tag in tags:
        tag_id, x, y, z, angle, center_x, center_y, confidence, detector = tag
        
        # Color coding based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green - high confidence
        elif confidence > 0.5:
            color = (0, 255, 255)  # Yellow - medium confidence
        else:
            color = (0, 0, 255)  # Red - low confidence
        
        # Draw tag center and ID
        cv2.circle(display_img, (center_x, center_y), 8, color, -1)
        cv2.putText(display_img, f"ID:{tag_id}", (center_x-25, center_y-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw bounding box (if corners available)
        try:
            # This would require access to tag.corners in the main detection
            pass
        except:
            # Fallback: draw simple cross
            size = 20
            cv2.line(display_img, (center_x-size, center_y), (center_x+size, center_y), color, 2)
            cv2.line(display_img, (center_x, center_y-size), (center_x, center_y+size), color, 2)
        
        # Information display
        info_lines = [
            f"Pos:({x:4d},{y:4d},{z:4d})",
            f"Ang:{angle:3d}°",
            f"Conf:{confidence:.2f}",
            f"Det:{detector}"
        ]
        
        for i, line in enumerate(info_lines):
            y_offset = center_y + 35 + i * 20
            cv2.putText(display_img, line, (center_x-45, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Performance metrics overlay
    if show_metrics:
        draw_performance_metrics(display_img)
    
    return display_img

def draw_performance_metrics(img):
    """Draw real-time performance metrics"""
    avg_time = np.mean(detection_performance['detection_times']) if detection_performance['detection_times'] else 0
    success_rate = detection_performance['successful_detections'] / max(detection_performance['frame_count'], 1)
    
    metrics = [
        f"Frames: {detection_performance['frame_count']}",
        f"Success: {success_rate:.1%}",
        f"Avg Time: {avg_time:.1f}ms",
        f"Detector: {adaptive_settings['current_detector']}",
        f"Brightness: {adaptive_settings['average_brightness']:.0f}"
    ]
    
    for i, metric in enumerate(metrics):
        cv2.putText(img, metric, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ==================== SYSTEM MANAGEMENT ====================
def get_detection_stats():
    """Get comprehensive detection statistics"""
    return {
        'total_frames': detection_performance['frame_count'],
        'success_rate': detection_performance['successful_detections'] / max(detection_performance['frame_count'], 1),
        'avg_processing_time': np.mean(detection_performance['detection_times']) if detection_performance['detection_times'] else 0,
        'current_detector': adaptive_settings['current_detector'],
        'detector_performance': {
            name: (len(data), np.mean([p[1] for p in data]) if data else 0) 
            for name, data in detection_performance['detector_performance'].items()
        }
    }

def reset_detection_system():
    """Reset all tracking and performance data"""
    global tag_kalman_states, tag_tracking_history, detection_performance, adaptive_settings
    
    tag_kalman_states.clear()
    tag_tracking_history.clear()
    
    detection_performance = {
        'frame_count': 0,
        'successful_detections': 0,
        'detection_times': deque(maxlen=100),
        'confidence_history': defaultdict(lambda: deque(maxlen=50)),
        'detector_performance': {name: deque(maxlen=50) for name in detector_configs.keys()}
    }
    
    adaptive_settings = {
        'current_detector': 'FAST',
        'last_lighting_assessment': 0,
        'average_brightness': 127,
        'detection_quality': 1.0,
        'frame_processing_time': deque(maxlen=30)
    }
    
    print(" Detection system reset complete")

# ==================== MAIN TESTING ====================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    
    print(" ULTRA-ENHANCED AprilTag Detection System Started")
    print("Features: Multi-detector, Adaptive Processing, Advanced Tracking, Performance Monitoring")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        tags = get_tags(frame)
        processing_time = (time.time() - start_time) * 1000
        
        debug_frame = draw_enhanced_tags(frame, tags)
        
        # Display real-time info
        cv2.putText(debug_frame, f"Processing: {processing_time:.1f}ms", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("ULTRA AprilTag Detection", debug_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_detection_system()
        elif key == ord('s'):
            stats = get_detection_stats()
            print(" Current Stats:", json.dumps(stats, indent=2))
    
    cap.release()
    cv2.destroyAllWindows()
