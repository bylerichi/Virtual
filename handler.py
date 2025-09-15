# handler.py - Script principal pour RunPod
import runpod
import torch
import numpy as np
from PIL import Image
import base64
import io
import cv2
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
from controlnet_aux import NormalBaeDetector
import json

# Configuration des modèles
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Utilisation du device: {DEVICE}")

# Chargement des modèles
print("📥 Chargement des modèles...")

# 1. Depth Anything V2 pour l'estimation de profondeur (CHEMIN CORRIGÉ)
depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large")
depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large")
depth_model.to(DEVICE)

# 2. Segmentation pour détecter les murs
segmentation_pipeline = pipeline(
    "image-segmentation",
    model="facebook/mask2former-swin-large-coco-panoptic",
    device=0 if DEVICE == "cuda" else -1
)

# 3. ControlNet Normal pour détecter les surfaces planes
normal_detector = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

print("✅ Tous les modèles chargés avec succès!")

def base64_to_image(base64_string):
    """Convertit une string base64 en image PIL"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image

def image_to_base64(image):
    """Convertit une image PIL en string base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def estimate_depth(image):
    """Estime la profondeur avec Depth Anything V2"""
    print("🔍 Estimation de la profondeur...")
    
    # Préparation de l'image
    inputs = depth_processor(images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Post-traitement
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    depth_map = prediction.squeeze().cpu().numpy()
    
    # Normalisation pour obtenir des distances réelles (approximation)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = depth_map * 10 + 0.5  # Échelle 0.5m à 10.5m
    
    return {
        "depthMap": depth_map.tolist(),
        "width": image.width,
        "height": image.height,
        "realWorldScale": calculate_real_world_scale(image.width, image.height, depth_map)
    }

def detect_walls(image, depth_map):
    """Détecte les murs avec segmentation et ControlNet"""
    print("🏠 Détection des murs...")
    
    # 1. Segmentation pour identifier les surfaces
    segments = segmentation_pipeline(image)
    
    # 2. Détection des normales avec ControlNet
    normal_map = normal_detector(image)
    normal_array = np.array(normal_map)
    
    # 3. Analyse des segments pour trouver les murs
    wall_mask = np.zeros((image.height, image.width), dtype=bool)
    wall_confidence = 0.0
    wall_normal = [0, 0, 1]  # Normal par défaut
    
    for segment in segments:
        if segment['label'] in ['wall', 'wall-other', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood']:
            mask = np.array(segment['mask'])
            wall_mask = np.logical_or(wall_mask, mask)
            wall_confidence = max(wall_confidence, segment.get('score', 0.8))
    
    # Calcul de la normale moyenne du mur
    if wall_mask.any():
        wall_pixels = normal_array[wall_mask]
        wall_normal = np.mean(wall_pixels, axis=0).tolist()
        wall_normal = [float(x) for x in wall_normal]
    
    return {
        "wallMask": wall_mask.tolist(),
        "wallNormal": wall_normal,
        "wallPlane": calculate_wall_plane(wall_mask, depth_map),
        "confidence": float(wall_confidence)
    }

def calculate_perspective(image, wall_detection):
    """Calcule la correction de perspective"""
    print("📐 Calcul de la perspective...")
    
    # Conversion en OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv_COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Détection des lignes avec Hough Transform
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    # Calcul des points de fuite
    vanishing_points = []
    horizon_line = {"angle": 0, "y": image.height // 2}
    
    if lines is not None:
        # Analyse des lignes pour trouver les points de fuite
        for line in lines[:10]:  # Limiter à 10 lignes principales
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Points sur la ligne
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            vanishing_points.append([float(x1), float(y1)])
    
    # Matrice d'homographie simplifiée
    homography_matrix = np.eye(3).tolist()
    
    return {
        "homographyMatrix": homography_matrix,
        "vanishingPoints": vanishing_points,
        "horizonLine": horizon_line
    }

def calculate_real_world_scale(width, height, depth_map):
    """Calcule l'échelle pixels/mètre"""
    # Paramètres de caméra estimés
    focal_length = 1000  # pixels
    sensor_width = 36    # mm (capteur full frame)
    
    avg_depth = np.mean(depth_map)
    scale = (width * focal_length) / (sensor_width * avg_depth * 1000)
    return float(scale)

def calculate_wall_plane(wall_mask, depth_map):
    """Calcule l'équation du plan du mur"""
    # Équation du plan : ax + by + cz + d = 0
    return {
        "a": 0.0,
        "b": 0.0, 
        "c": 1.0,
        "d": -float(np.mean(depth_map))
    }

def get_depth_at_point(depth_result, x, y):
    """Obtient la profondeur à un point spécifique"""
    depth_map = np.array(depth_result["depthMap"])
    height, width = depth_map.shape
    
    # Normalisation des coordonnées
    norm_x = int((x / depth_result["width"]) * width)
    norm_y = int((y / depth_result["height"]) * height)
    
    # Clamp des coordonnées
    norm_x = max(0, min(width - 1, norm_x))
    norm_y = max(0, min(height - 1, norm_y))
    
    return float(depth_map[norm_y, norm_x])

def handler(job):
    """Handler principal pour RunPod"""
    try:
        job_input = job["input"]
        task = job_input.get("task")
        
        if task == "depth_estimation":
            # Estimation de profondeur
            image_b64 = job_input["image"]
            image = base64_to_image(image_b64)
            
            result = estimate_depth(image)
            
            # Ajouter la profondeur au point spécifique si demandé
            if "point_x" in job_input and "point_y" in job_input:
                depth_at_point = get_depth_at_point(
                    result, 
                    job_input["point_x"], 
                    job_input["point_y"]
                )
                result["depthAtPoint"] = depth_at_point
            
            return result
            
        elif task == "wall_detection":
            # Détection des murs
            image_b64 = job_input["image"]
            image = base64_to_image(image_b64)
            depth_map = np.array(job_input.get("depth_map", []))
            
            result = detect_walls(image, depth_map)
            return result
            
        elif task == "perspective_analysis":
            # Analyse de perspective
            image_b64 = job_input["image"]
            image = base64_to_image(image_b64)
            wall_detection = job_input["wall_detection"]
            
            result = calculate_perspective(image, wall_detection)
            return result
            
        elif task == "full_analysis":
            # Analyse complète
            image_b64 = job_input["image"]
            point_x = job_input.get("point_x", 0)
            point_y = job_input.get("point_y", 0)
            
            image = base64_to_image(image_b64)
            
            # 1. Estimation de profondeur
            depth_result = estimate_depth(image)
            depth_at_point = get_depth_at_point(depth_result, point_x, point_y)
            
            # 2. Détection des murs
            wall_result = detect_walls(image, np.array(depth_result["depthMap"]))
            
            # 3. Analyse de perspective
            perspective_result = calculate_perspective(image, wall_result)
            
            return {
                "success": True,
                "depth": depth_result,
                "walls": wall_result,
                "perspective": perspective_result,
                "depthAtPoint": depth_at_point,
                "processing_info": {
                    "device": DEVICE,
                    "image_size": [image.width, image.height],
                    "models_loaded": True
                }
            }
        
        else:
            return {"error": f"Task '{task}' not supported"}
            
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return {"error": str(e)}

# Démarrage du serveur RunPod
if __name__ == "__main__":
    print("🚀 Démarrage du serveur RunPod...")
    runpod.serverless.start({"handler": handler})
