from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import filedialog
import cv2

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

class FaceVerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Verification")
        self.id_img = None
        self.webcam_img = None
        self.id_img_path = ""
        
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.capture_btn = tk.Button(root, text="Capture Photo from Webcam", command=self.capture_webcam_image)
        self.capture_btn.pack()

        self.verify_btn = tk.Button(root, text="Verify", command=self.verify_face)
        self.verify_btn.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.id_img_path = file_path
            self.id_img = Image.open(file_path)
            self.display_image(self.id_img, "ID Image")

    def capture_webcam_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.webcam_img = Image.fromarray(frame)
            self.display_image(self.webcam_img, "Webcam Image")

    def display_image(self, img, title):
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(self.root, image=img_tk)
        img_label.image = img_tk
        img_label.pack()

    def verify_face(self):
        if self.id_img and self.webcam_img:
            aligned_id_img = self.align_face(self.id_img)
            aligned_webcam_img = self.align_face(self.webcam_img)

            if aligned_id_img is not None and aligned_webcam_img is not None:
                embeddings_id = resnet(aligned_id_img).detach()
                embeddings_webcam = resnet(aligned_webcam_img).detach()

                distance = (embeddings_id - embeddings_webcam).norm().item()
                if distance < 1.0:  # Adjust threshold as needed
                    self.result_label.config(text="Same person", fg="green")
                else:
                    self.result_label.config(text="Different person", fg="red")
            else:
                self.result_label.config(text="Face not detected in one or both images", fg="red")
        else:
            self.result_label.config(text="Please upload ID image and capture webcam photo", fg="red")

    def align_face(self, img):
        faces, _ = mtcnn.detect(img)
        if faces is not None:
            aligned_face = mtcnn(img)
            return aligned_face.unsqueeze(0)
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceVerificationApp(root)
    root.mainloop()