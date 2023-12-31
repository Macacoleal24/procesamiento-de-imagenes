import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, Label, Button, filedialog
from PIL import Image, ImageTk

class ImageRestorationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Restauración de Imágenes")

        # Propiedades de la imagen
        self.imagen_path = ""
        self.imagen_original = None
        self.resized_image = None
        self.mask = None
        self.parametros = {'kernel_size': 5, 'threshold': 180, 'alpha': 1.4, 'beta': 20, 'saturation': 1.8, 'denoising': 15, 'resize_factor': 1.0, 'display_scale': 1.0}

        # Interfaz gráfica
        self.create_widgets()

    def create_widgets(self):
        # Controles deslizantes para ajustar los parámetros
        # Columna 1
        Label(self.master, text="Tamaño del kernel").grid(row=0, column=0, pady=5)
        self.kernel_slider = Scale(self.master, from_=3, to=15, orient="horizontal", command=self.update_parameters)
        self.kernel_slider.set(self.parametros['kernel_size'])
        self.kernel_slider.grid(row=1, column=0, pady=5)

        Label(self.master, text="Umbral").grid(row=2, column=0, pady=5)
        self.threshold_slider = Scale(self.master, from_=0, to=255, orient="horizontal", command=self.update_parameters)
        self.threshold_slider.set(self.parametros['threshold'])
        self.threshold_slider.grid(row=3, column=0, pady=5)

        Label(self.master, text="Contraste").grid(row=4, column=0, pady=5)
        self.contrast_slider = Scale(self.master, from_=0, to=3, orient="horizontal", resolution=0.1, command=self.update_parameters)
        self.contrast_slider.set(self.parametros['alpha'])
        self.contrast_slider.grid(row=5, column=0, pady=5)

        # Columna 2
        Label(self.master, text="Brillo").grid(row=0, column=1, pady=5)
        self.brightness_slider = Scale(self.master, from_=-50, to=50, orient="horizontal", command=self.update_parameters)
        self.brightness_slider.set(self.parametros['beta'])
        self.brightness_slider.grid(row=1, column=1, pady=5)

        Label(self.master, text="Saturación").grid(row=2, column=1, pady=5)
        self.saturation_slider = Scale(self.master, from_=0.1, to=3, orient="horizontal", resolution=0.1, command=self.update_parameters)
        self.saturation_slider.set(self.parametros['saturation'])
        self.saturation_slider.grid(row=3, column=1, pady=5)

        Label(self.master, text="Reducción de ruido").grid(row=4, column=1, pady=5)
        self.denoising_slider = Scale(self.master, from_=0, to=50, orient="horizontal", command=self.update_parameters)
        self.denoising_slider.set(self.parametros['denoising'])
        self.denoising_slider.grid(row=5, column=1, pady=5)

        # Botones para acciones
        Button(self.master, text="Cargar Imagen", command=self.load_image).grid(row=6, column=0, columnspan=2, pady=10)

        # Mostrar la imagen original en Tkinter
        self.image_label = Label(self.master)
        self.image_label.grid(row=7, column=0, columnspan=2)

    def update_parameters(self, value):
        self.parametros['kernel_size'] = int(self.kernel_slider.get())
        self.parametros['threshold'] = int(self.threshold_slider.get())
        self.parametros['alpha'] = float(self.contrast_slider.get())
        self.parametros['beta'] = int(self.brightness_slider.get())
        self.parametros['saturation'] = float(self.saturation_slider.get())
        self.parametros['denoising'] = int(self.denoising_slider.get())

        # Actualizar la imagen en tiempo real
        self.restore_image()

    def load_image(self):
        self.imagen_path = filedialog.askopenfilename(title="Seleccionar Imagen", filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.gif")])
        if self.imagen_path:
            self.imagen_original = cv2.imread(self.imagen_path)
            self.display_image(self.imagen_original)

    def create_mask(self):
        # Convertir la imagen a escala de grises
        gris = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)

        # Aplicar umbralización para identificar áreas dañadas
        _, mask = cv2.threshold(gris, self.parametros['threshold'], 255, cv2.THRESH_BINARY)

        # Dilatar la máscara para cubrir un área más grande
        kernel = np.ones((self.parametros['kernel_size'], self.parametros['kernel_size']), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def restore_image(self):
        # Crear una nueva máscara con los parámetros actuales
        self.mask = self.create_mask()

        # Redimensionar la imagen original
        height, width = self.imagen_original.shape[:2]
        new_width = int(width * self.parametros['resize_factor'])
        new_height = int(height * self.parametros['resize_factor'])
        self.resized_image = cv2.resize(self.imagen_original, (new_width, new_height))

        # Aplicar ajustes de contraste y brillo
        imagen_editada = cv2.convertScaleAbs(self.resized_image, alpha=self.parametros['alpha'], beta=self.parametros['beta'])

        # Ajuste de saturación
        hsv = cv2.cvtColor(imagen_editada, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * self.parametros['saturation']
        imagen_editada = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Reducción de ruido
        imagen_filtrada = cv2.fastNlMeansDenoisingColored(imagen_editada, None, self.parametros['denoising'], self.parametros['denoising'], 7, 21)

        # Aplicar el algoritmo de inpainting
        restaurada = cv2.inpaint(imagen_filtrada, self.mask, 3, cv2.INPAINT_TELEA)

        # Redimensionar la imagen para la visualización
        display_height = int(new_height * self.parametros['display_scale'])
        display_width = int(new_width * self.parametros['display_scale'])
        display_image = cv2.resize(restaurada, (display_width, display_height))

        # Mostrar la imagen restaurada en Tkinter
        self.display_image(display_image)

        # Guardar la imagen restaurada
        cv2.imwrite('imagen_restaurada.jpg', restaurada)

    def display_image(self, image):
        # Convertir la imagen de OpenCV a formato compatible con Tkinter
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image=image)

        # Obtener las dimensiones de la imagen
        width, height = image.width(), image.height()

        # Redimensionar la etiqueta para que coincida con el tamaño de la imagen
        self.image_label.config(width=width, height=height)

        # Actualizar la etiqueta de la imagen en la interfaz
        self.image_label.configure(image=image)
        self.image_label.image = image

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRestorationApp(root)
    root.mainloop()