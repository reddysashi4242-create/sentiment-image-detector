import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# ---------------------------
# Load the pre-trained model
# ---------------------------
model = model = load_model("model.h5", compile=False)
  # Make sure this file is in the same directory

# ---------------------------
# Emotion labels
# ---------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------------------------
# Preprocess the image
# ---------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')     # Grayscale
    img = img.resize((64, 64))                    # Resize to 48x48
    img_array = np.array(img) / 255.0             # Normalize
    img_array = img_array.reshape(1, 64, 64, 1)   # Add batch and channel dims
    return img_array

# ---------------------------
# Predict emotion
# ---------------------------
def predict_emotion(image_path):
    processed = preprocess_image(image_path)
    prediction = model.predict(processed)
    print("Prediction Probabilities:", prediction)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

# ---------------------------
# Capture image from webcam
# ---------------------------
def capture_image_from_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Camera could not be opened!")
        return None

    print("📸 Press SPACE to capture image...")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow("Press SPACE to capture", frame)

        key = cv2.waitKey(1)
        if key % 256 == 32:  # SPACE pressed
            img_path = "captured_image.jpg"
            cv2.imwrite(img_path, frame)
            print(f"✅ Image saved as {img_path}")
            break

    cam.release()
    cv2.destroyAllWindows()
    return img_path

# ---------------------------
# Main Function
# ---------------------------
def main():
    print("===== Sentiment Detection =====")
    print("1. Capture from Camera")
    print("2. Provide Image Path")
    choice = input("Enter choice (1 or 2): ")

    if choice == '1':
        image_path = capture_image_from_camera()
        if image_path is None:
            print("❌ No image captured.")
            return
    elif choice == '2':
        image_path = input("Enter image path: ").strip('"')
    else:
        print("❌ Invalid choice")
        return

    try:
        emotion = predict_emotion(image_path)
        print(f"🧠 Predicted Emotion: {emotion}")
    except Exception as e:
        print("❌ Error during prediction:", str(e))


if __name__ == "__main__":
    main()
