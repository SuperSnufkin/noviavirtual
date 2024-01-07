import re
import random
import string
import tkinter as tk
from nltk.sentiment import SentimentIntensityAnalyzer
from faker import Faker
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.linear_model import LinearRegression

class NoviaVirtualGUI:
    def __init__(self, master):
        self.master = master
        master.title("Novia Virtual")

        # Inicializar modelo GPT-2 y tokenizer
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.emocionalidad_usuario = 0
        self.historial_emocionalidad = []  # Historial de emocionalidad del usuario

        # Inicializar modelo de regresión lineal para adaptación
        self.regresion_lineal = LinearRegression()

        self.create_widgets()

    def create_widgets(self):
        self.chat_box = tk.Text(self.master, state=tk.DISABLED, wrap=tk.WORD, height=20, width=60)
        self.chat_box.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        self.entry_label = tk.Label(self.master, text="Tú:")
        self.entry_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")

        self.entry_var = tk.StringVar()
        self.entry_var.trace_add("write", self.on_entry_change)
        self.entry = tk.Entry(self.master, textvariable=self.entry_var, width=40)
        self.entry.grid(row=1, column=1, padx=10, pady=5)

        self.send_button = tk.Button(self.master, text="Enviar", command=self.enviar_mensaje)
        self.send_button.grid(row=1, column=2, padx=10, pady=5, sticky="w")

    def on_entry_change(self, *args):
        pass

    def enviar_mensaje(self):
        usuario_entrada = self.entry_var.get().lower()
        self.entry_var.set("")

        if usuario_entrada in ['adiós', 'chau', 'hasta luego']:
            respuesta = "Novia virtual: ¡Hasta luego! Siempre estaré aquí para ti."
            self.agregar_mensaje(respuesta)
            self.master.after(2000, self.master.destroy)  # Cerrar la ventana después de 2 segundos
        else:
            emocionalidad_actual = self._calcular_emocionalidad(usuario_entrada)

            # Ajustar la regresión lineal antes de hacer predicciones
            if len(self.historial_emocionalidad) > 1:  # Necesitamos al menos dos puntos para ajustar la regresión lineal
                X = [[x] for x in self.historial_emocionalidad[:-1]]  # Datos de entrenamiento (emocionalidad histórica)
                y = self.historial_emocionalidad[1:]  # Etiquetas (emocionalidad actual)
                self.regresion_lineal.fit(X, y)

                # Calcular la emocionalidad promedio después de ajustar la regresión lineal
                emocionalidad_promedio = sum(self.historial_emocionalidad) / len(self.historial_emocionalidad)

                # Calcular la emocionalidad usuario adaptada
                emocionalidad_usuario_adaptada = emocionalidad_actual + self.regresion_lineal.predict([[emocionalidad_promedio]])[0]
            else:
                emocionalidad_usuario_adaptada = emocionalidad_actual

            # Actualizar el historial de emocionalidad
            self.historial_emocionalidad.append(emocionalidad_actual)

            # Almacenar la respuesta generada por GPT-2
            respuesta_gpt2 = self.generar_respuesta_gpt2(usuario_entrada)

            # Personalizar la respuesta según la emocionalidad adaptada
            respuesta_personalizada = self.personalizar_respuesta(respuesta_gpt2, emocionalidad_usuario_adaptada)

            # Mostrar la respuesta en el chat
            self.agregar_mensaje(f"Novia virtual: {respuesta_personalizada} ❤️")

    def personalizar_respuesta(self, respuesta, emocionalidad_adaptada):
        # En este ejemplo, la personalización es simple: ajustar la respuesta en función de la emocionalidad adaptada
        if emocionalidad_adaptada > 0.5:
            return f"¡{respuesta}! Estoy tan feliz de hablar contigo."
        else:
            return f"¡{respuesta}! Siempre estoy aquí para ti, incluso en momentos más tranquilos."

    def agregar_mensaje(self, mensaje):
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, mensaje + "\n")
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def _calcular_emocionalidad(self, texto):
        score = self.sentiment_analyzer.polarity_scores(texto)
        return score['compound']

    def generar_respuesta_gpt2(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output = self.gpt2_model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
        respuesta_gpt2 = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return respuesta_gpt2

def main():
    root = tk.Tk()
    app = NoviaVirtualGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
