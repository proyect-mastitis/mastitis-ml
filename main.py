from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from datetime import datetime
import shutil, os, uuid
import cv2
import numpy as np

main = FastAPI(title="Servidor de Análisis de Mastitis")

# CORS
main.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "http://192.168.18.82:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
main.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

model = YOLO("models/best.pt")

# VALIDADORES
MIN_CONFIDENCE = 0.55    #confianza mínima
MIN_BOX_SIZE = 0.08      #tamaño mínimo
MAX_BOX_SIZE = 0.68      #tamaño máximo
MIN_ASPECT_RATIO = 0.6   #forma mínima
MAX_ASPECT_RATIO = 2.0   #forma máxima


def validate_udder_detection(box, img_width: float, img_height: float) -> dict:
    try:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

        box_width = x2 - x1
        box_height = y2 - y1
        box_area = (box_width * box_height) / (img_width * img_height)

        if box_area < MIN_BOX_SIZE:
            return {"valid": False, "error": "Detección muy pequeña. Acerca más la cámara."}
        if box_area > MAX_BOX_SIZE:
            return {"valid": False, "error": "Detección muy grande. Probablemente no sea una ubre."}

        aspect_ratio = box_height / box_width
        if aspect_ratio < MIN_ASPECT_RATIO:
            return {"valid": False, "error": "Ubre muy ancha. Verifica el ángulo."}
        if aspect_ratio > MAX_ASPECT_RATIO:
            return {"valid": False, "error": "Ubre muy alta. Probablemente sea otro objeto."}

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if cx / img_width < 0.1 or cx / img_width > 0.9:
            return {"valid": False, "error": "Ubre muy al borde. Centra la imagen."}
        if cy / img_height < 0.2 or cy / img_height > 0.9:
            return {"valid": False, "error": "Ubre en mala posición. Recaptura."}

        return {"valid": True}

    except Exception as e:
        return {"valid": False, "error": f"Error validando ubre: {str(e)}"}


@main.post("/analyze")
async def analyze(animal_id: str = Form(...), files: list[UploadFile] = File(...)):

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Debe subir al menos una imagen")
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="Máximo 2 imágenes permitidas")

    results_data = []
    valid_count = 0
    file_index = 0

    valid_paths = []

    # -----------------------------
    # 1. Guardar archivos
    # -----------------------------
    for file in files:
        file_index += 1

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            results_data.append({
                "image_position": file_index,
                "filename": file.filename,
                "valid": False,
                "error": "Formato no soportado",
                "image_path": None,
                "box": None,
                "confidence": 0
            })
            continue

        temp_path = f"uploads/temp_{uuid.uuid4()}{ext}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        valid_paths.append({
            "path": temp_path,
            "filename": file.filename,
            "position": file_index
        })

    # -----------------------------
    # 2. YOLO PRIMERO (DETECTA UBRE)
    # -----------------------------
    for vp in valid_paths:

        path = vp["path"]
        position = vp["position"]
        filename = vp["filename"]

        try:
            results = model.predict(path, save=False, conf=0.3, verbose=False)

            # ❌ NO HAY UBRE
            if not results[0].boxes or len(results[0].boxes) == 0:
                results_data.append({
                    "image_position": position,
                    "filename": filename,
                    "valid": False,
                    "error": "No se detecta ubre en la imagen",
                    "image_path": None,
                    "box": None,
                    "confidence": 0
                })
                continue

            confs = results[0].boxes.conf
            best = int(confs.argmax())
            box = results[0].boxes[best]

            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            # baja confianza
            if confidence < MIN_CONFIDENCE:
                results_data.append({
                    "image_position": position,
                    "filename": filename,
                    "valid": False,
                    "error": f"Baja confianza ({confidence*100:.1f}%)",
                    "image_path": None,
                    "box": None,
                    "confidence": round(confidence * 100, 2)
                })
                continue

            # -----------------------------
            # 3. VALIDACIÓN SOLO SI HAY UBRE
            # -----------------------------
            img = cv2.imread(path)
            h, w = img.shape[:2]

            box_check = validate_udder_detection(box, w, h)

            if not box_check["valid"]:
                results_data.append({
                    "image_position": position,
                    "filename": filename,
                    "valid": False,
                    "error": box_check["error"],
                    "image_path": None,
                    "box": None,
                    "confidence": round(confidence * 100, 2)
                })
                continue

                            # -----------------------------
            # 4. IMAGEN VÁLIDA
            # -----------------------------
            valid_count += 1
            status = "Con mastitis" if cls_id == 1 else "Sin mastitis"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_image_name = f"analysis_{animal_id}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            final_image_path = os.path.join("uploads", final_image_name)

            # guardar imagen ORIGINAL (sin resize)
            shutil.copy(path, final_image_path)

            # 🔥 USAR EL BEST BOX CORRECTO
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

            box_width = x2 - x1
            box_height = y2 - y1
            box_area_pct = (box_width * box_height) / (w * h) * 100

            results_data.append({
                "image_position": position,
                "filename": filename,
                "valid": True,
                "status": status,
                "mastitis_detected": cls_id == 1,
                "confidence": round(confidence * 100, 2),
                "image_path": f"/uploads/{final_image_name}",
                "image_id": str(uuid.uuid4()),
                "image_width": w,
                "image_height": h,
                "box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width": round(box_width, 2),
                    "height": round(box_height, 2),
                    "area_percentage": round(box_area_pct, 2),
                    "center_x": round((x1 + x2) / 2, 2),
                    "center_y": round((y1 + y2) / 2, 2),
                }
            })

        except Exception as e:
            results_data.append({
                "image_position": position,
                "filename": filename,
                "valid": False,
                "error": f"Error procesando: {str(e)}",
                "image_path": None,
                "box": None,
                "confidence": 0
            })
    
    # -----------------------------
    # BLOQUEAR SI HAY INVALIDAS
    # -----------------------------
    invalid_images = [r for r in results_data if not r.get("valid", False)]

    if len(invalid_images) > 0:
        for vp in valid_paths:
            if os.path.exists(vp["path"]):
                os.remove(vp["path"])

        raise HTTPException(
            status_code=400,
            detail={
                "message": "❌ Hay imágenes inválidas. No se procesó el análisis.",
                "details": invalid_images
            }
        )

    # -----------------------------
    # 5. RESULTADO FINAL (FIX KEYERROR)
    # -----------------------------
    has_mastitis = any(
        r.get("mastitis_detected", False)
        for r in results_data
        if r.get("valid", False)
    )

    final_conf = max(
        [r.get("confidence", 0) for r in results_data],
        default=0
    )

    # limpiar temporales
    for vp in valid_paths:
        if os.path.exists(vp["path"]):
            os.remove(vp["path"])

    return {
        "animal_id": animal_id,
        "status": "Con mastitis" if has_mastitis else "Sin mastitis",
        "mastitis_detected": has_mastitis,
        "confidence": final_conf,
        "analysis_date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "valid_count": valid_count,
        "total_uploaded": len(files),
        "is_valid": True,
        "details": results_data
    }


@main.get("/health")
def health():
    return {"status": "ok", "model": "YOLO", "uploads_folder": os.path.exists("uploads")}