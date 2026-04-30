import cv2
import os
from classifier import predict
from yolo_detector import detect

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run(image, save_path=None):
    boxes = detect(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label, conf = predict(crop)
        print(label, conf)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}",
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0), 2)

        if save_path:
            name, ext = os.path.splitext(os.path.basename(save_path))
            crop_path = os.path.join(OUTPUT_DIR, f"{name}_crop_{i}{ext}")
            cv2.imwrite(crop_path, crop)
            print(f"Saved crop to {crop_path}")

    if save_path:
        cv2.imwrite(save_path, image)
        print(f"Saved result to {save_path}")

    return image


if __name__ == "__main__":
    input_path = r"D:\Projects\DEPI Project\CV\testttt.jpeg"
    image = cv2.imread(input_path)

    filename = os.path.basename(input_path)
    save_path = os.path.join(OUTPUT_DIR, filename)

    output = run(image, save_path=save_path)

    cv2.imshow("result", output)
    cv2.waitKey(0)
