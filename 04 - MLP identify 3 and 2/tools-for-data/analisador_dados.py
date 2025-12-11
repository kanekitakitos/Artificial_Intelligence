import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# --- MOTOR DE ALINHAMENTO E LIMPEZA ---

class ImageProcessor:
    def __init__(self, img_size=20):
        self.img_size = img_size
        self.mean_2 = None
        self.mean_3 = None
        self.diff_map = None
        self.max_diff = 0
        self.is_fitted = False

    def center_of_mass_shift(self, X):
        """
        Alinha todas as imagens pelo Centro de Massa.
        Garante que todos os números ficam na posição (10,10).
        """
        X_aligned = np.zeros_like(X)
        rows, cols = np.indices((self.img_size, self.img_size))
        
        for i, img_flat in enumerate(X):
            img = img_flat.reshape(self.img_size, self.img_size)
            total_mass = img.sum()
            
            if total_mass <= 0:
                X_aligned[i] = img_flat
                continue

            # 1. Calcular Centro de Massa Atual
            cy = np.sum(rows * img) / total_mass
            cx = np.sum(cols * img) / total_mass
            
            # 2. Calcular deslocamento para o centro
            shift_y = int(round(self.img_size/2 - 0.5 - cy))
            shift_x = int(round(self.img_size/2 - 0.5 - cx))
            
            # 3. Mover a imagem
            img_shifted = self._shift_image(img, shift_y, shift_x)
            X_aligned[i] = img_shifted.flatten()
            
        return X_aligned

    def _shift_image(self, img, dy, dx):
        """Move a imagem preenchendo o vazio com 0."""
        res = np.zeros_like(img)
        src_y_start = max(0, -dy)
        src_y_end = min(self.img_size, self.img_size - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(self.img_size, self.img_size - dx)
        
        dst_y_start = max(0, dy)
        dst_y_end = min(self.img_size, self.img_size + dy)
        dst_x_start = max(0, dx)
        dst_x_end = min(self.img_size, self.img_size + dx)
        
        try:
            res[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                img[src_y_start:src_y_end, src_x_start:src_x_end]
        except:
            pass
        return res

    def fit(self, X_aligned, y):
        """Calcula o mapa de diferenças."""
        self.mean_2 = X_aligned[y == 2].mean(axis=0)
        self.mean_3 = X_aligned[y == 3].mean(axis=0)
        self.diff_map = np.abs(self.mean_2 - self.mean_3)
        self.max_diff = np.max(self.diff_map)
        self.is_fitted = True

    def get_mask(self, threshold_percentage):
        if not self.is_fitted: return None
        cutoff = (threshold_percentage / 100.0) * self.max_diff
        # Retorna 1 onde é importante, 0 onde é lixo
        return (self.diff_map >= cutoff).astype(float)

# --- INTERFACE GRÁFICA ---

class AdvancedOptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroLab: Otimizador 20x20 (Centrar + Limpar)")
        self.root.geometry("1400x900")
        
        self.X = None
        self.y = None
        self.X_23 = None
        self.y_23 = None
        self.X_aligned = None 
        
        self.processor = ImageProcessor(img_size=20)
        
        # --- CARREGAMENTO ---
        frame_top = ttk.LabelFrame(root, text="1. Carregar Dados")
        frame_top.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame_top, text="Carregar Dataset (.csv)", command=self.load_images).pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_x = ttk.Label(frame_top, text="...")
        self.lbl_x.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(frame_top, text="Carregar Labels (.csv)", command=self.load_labels).pack(side=tk.LEFT, padx=20, pady=5)
        self.lbl_y = ttk.Label(frame_top, text="...")
        self.lbl_y.pack(side=tk.LEFT, padx=5)

        # --- CONTROLO ---
        frame_controls = ttk.LabelFrame(root, text="2. Ajustes (Mantendo formato 20x20)")
        frame_controls.pack(fill=tk.X, padx=10, pady=10)
        
        self.var_align = tk.BooleanVar(value=True)
        self.chk_align = ttk.Checkbutton(frame_controls, text="1º: Auto-Centrar Imagens", variable=self.var_align, command=self.run_process)
        self.chk_align.pack(side=tk.LEFT, padx=20)
        
        self.lbl_slider = ttk.Label(frame_controls, text="2º: Corte de Ruído: 30%", font=("Arial", 10, "bold"))
        self.lbl_slider.pack(side=tk.LEFT, padx=20)
        
        self.slider = ttk.Scale(frame_controls, from_=0, to=95, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.slider.set(30)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
        ttk.Button(frame_controls, text="💾 EXPORTAR CSV (SÓ IMAGENS)", command=self.save_data).pack(side=tk.RIGHT, padx=20)

        # --- VISUALIZAÇÃO ---
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, self.axs = plt.subplots(2, 4, figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        for ax in self.axs.flatten(): ax.axis('off')

    def load_images(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self.X = pd.read_csv(path, header=None).values
            self.lbl_x.config(text="Carregado")
            self.check_ready()

    def load_labels(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self.y = pd.read_csv(path, header=None).values.flatten()
            self.lbl_y.config(text="Carregado")
            self.check_ready()

    def check_ready(self):
        if self.X is not None and self.y is not None:
            # Filtrar apenas 2 e 3
            idx = np.where((self.y == 2) | (self.y == 3))[0]
            self.X_23 = self.X[idx]
            self.y_23 = self.y[idx]
            self.run_process()

    def run_process(self):
        if self.X_23 is None: return
        
        # 1. Alinhamento
        if self.var_align.get():
            self.root.config(cursor="watch")
            self.root.update()
            self.X_aligned = self.processor.center_of_mass_shift(self.X_23)
            self.root.config(cursor="")
        else:
            self.X_aligned = self.X_23.copy()
            
        # 2. Calcular Diferenças
        self.processor.fit(self.X_aligned, self.y_23)
        self.update_plots(self.slider.get())

    def on_slider_change(self, val):
        self.lbl_slider.config(text=f"2º: Corte de Ruído: {int(float(val))}%")
        if self.X_aligned is not None:
            self.update_plots(float(val))

    def update_plots(self, threshold):
        if not self.processor.is_fitted: return
        
        mask = self.processor.get_mask(threshold)
        
        for ax in self.axs.flatten(): ax.clear(); ax.axis('off')
        
        # Linha 1: Análise
        self.axs[0, 0].imshow(self.processor.mean_2.reshape(20, 20), cmap='viridis')
        self.axs[0, 0].set_title("Média do '2'")
        
        self.axs[0, 1].imshow(self.processor.mean_3.reshape(20, 20), cmap='viridis')
        self.axs[0, 1].set_title("Média do '3'")
        
        self.axs[0, 2].imshow(self.processor.diff_map.reshape(20, 20), cmap='hot')
        self.axs[0, 2].set_title("MAPA DE DIFERENÇA\n(Áreas Fundamentais)")
        
        self.axs[0, 3].imshow(mask.reshape(20, 20), cmap='gray')
        self.axs[0, 3].set_title(f"Máscara (Mantém {np.sum(mask)} pixeis)")
        
        # Linha 2: Resultado
        idx_2 = np.where(self.y_23 == 2)[0][0]
        self.axs[1, 0].imshow(self.X_23[idx_2].reshape(20, 20), cmap='gray')
        self.axs[1, 0].set_title("2 Original")
        
        img_2_final = self.X_aligned[idx_2] * mask
        self.axs[1, 1].imshow(img_2_final.reshape(20, 20), cmap='gray')
        self.axs[1, 1].set_title("2 OTIMIZADO")
        
        idx_3 = np.where(self.y_23 == 3)[0][0]
        self.axs[1, 2].imshow(self.X_23[idx_3].reshape(20, 20), cmap='gray')
        self.axs[1, 2].set_title("3 Original")
        
        img_3_final = self.X_aligned[idx_3] * mask
        self.axs[1, 3].imshow(img_3_final.reshape(20, 20), cmap='gray')
        self.axs[1, 3].set_title("3 OTIMIZADO")
        
        self.canvas.draw()

    def save_data(self):
        if self.X_aligned is None: return
        
        mask = self.processor.get_mask(self.slider.get())
        
        # Aplica a máscara: Pixeis importantes ficam, os outros viram 0.0
        X_final = self.X_aligned * mask
        
        path = filedialog.asksaveasfilename(title="Salvar CSV Otimizado", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            # Salvar APENAS os dados da imagem (400 colunas)
            df = pd.DataFrame(X_final)
            
            # ATENÇÃO: Labels removidas aqui conforme pedido
            # df['label'] = self.y_23 
            
            # header=False para não escrever cabeçalho
            # index=False para não escrever número da linha
            df.to_csv(path, index=False, header=False)
            
            msg = f"Ficheiro guardado com sucesso!\n" \
                  f"Conteúdo: Apenas Pixeis (400 colunas)\n" \
                  f"Sem Labels."
            messagebox.showinfo("Sucesso", msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedOptimizationApp(root)
    root.mainloop()