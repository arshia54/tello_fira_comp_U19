# U19 Autonomous Air Competition - Gate Navigation System

## Table of Contents

1. Introduction
2. System Overview
3. Hardware Requirements
4. Software Architecture
5. Core Algorithms
6. State Machine
7. Installation Guide
8. Configuration
9. Usage Instructions
10. Performance Metrics
11. Troubleshooting Guide
12. Code Structure
13. Future Improvements
14. Appendices

---

## 1. Introduction

This document provides comprehensive documentation for the autonomous gate navigation system developed for the U19 Autonomous Air Competition. The system enables a DJI Tello drone to autonomously detect, approach, and navigate through multiple sequential gates using a combination of AprilTag localization and computer vision techniques.

### 1.1 Competition Requirements Addressed

- Fully autonomous operation (no human intervention)
- Takeoff and landing without remote control
- Gate detection and identification
- Multiple gate navigation (sequential)
- Obstacle avoidance
- Real-time status reporting
- Battery management

### 1.2 Key Innovations

- Dual-mode detection system (AprilTag + computer vision)
- Tanh-based smooth velocity control
- GOM fallback mechanism for temporary tag loss
- Low-pass filtered target stabilization

---

## 2. System Overview

### 2.1 High-Level Architecture





### 2.2 Data Flow

1. Drone captures 960x720 RGB frames at 30fps
2. Frames are processed for AprilTag detection (grayscale conversion)
3. Parallel processing for gate detection (LAB + HSV color spaces)
4. State machine determines appropriate action
5. Velocity commands sent to drone via UDP

---

## 3. Hardware Requirements

### 3.1 Minimum Specifications

| Component | Specification | Notes |
|-----------|---------------|-------|
| Drone | DJI Tello | Any firmware version |
| Processor | Intel Core i3 or equivalent | For image processing |
| RAM | 4GB minimum | 8GB recommended |
| OS | Windows 10/11, Ubuntu 18.04+, macOS | Any with Python support |
| Camera | Tello integrated (960x720) | Fixed focus |
| Battery | Tello OEM battery | 30% minimum for operation |

### 3.2 Physical Setup Requirements

- Clear flight area (minimum 5m x 5m x 3m)
- AprilTags printed on matte paper (tag36h11 family)
- Gates with detectable colors (frame + yellow elements)
- Uniform lighting (avoid direct sunlight)
- No obstacles between gates

### 3.3 AprilTag Specifications

| Parameter | Value |
|-----------|-------|
| Family | tag36h11 |
| Size | 5cm x 5cm (minimum) |
| Print quality | 600dpi recommended |
| Mounting | Center of gate, 1.5m height |
| Material | Matte paper (no glare) |

---

## 4. Software Architecture

### 4.1 Dependency Tree




### 4.2 Module Description

| Module | Function |
|--------|----------|
| `get_tags()` | AprilTag detection and pose estimation |
| `detect_gate_and_opening()` | Color-based gate detection |
| `low_pass_update()` | Target coordinate filtering |
| `get_tuning_speeds()` | Tanh-based velocity calculation |

---

## 5. Core Algorithms

### 5.1 AprilTag Detection

The system uses the pupil_apriltags library for robust tag detection.

```python
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)
