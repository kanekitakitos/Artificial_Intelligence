import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Tenta importar scipy para rotação
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("AVISO: Scipy não instalado. A rotação não funcionará (pip install scipy).")

# --- Funções Auxiliares ---
def transladar_imagem(img, shift_x, shift_y):
    rows, cols = img.shape
    res = np.zeros_like(img)
    dx, dy = int(shift_x), int(shift_y)
    
    src_y_start = max(0, -dy); src_y_end = min(rows, rows - dy)
    src_x_start = max(0, -dx); src_x_end = min(cols, cols - dx)
    dst_y_start = max(0, dy); dst_y_end = min(rows, rows + dy)
    dst_x_start = max(0, dx); dst_x_end = min(cols, cols + dx)
    
    if (src_y_start < src_y_end) and (src_x_start < src_x_end):
        res[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            img[src_y_start:src_y_end, src_x_start:src_x_end]
    return res

def redimensionar_imagem_numpy(imagem, escala):
    if escala == 1.0: return imagem
    linhas_orig, cols_orig = imagem.shape
    novas_linhas = int(linhas_orig * escala)
    novas_cols = int(cols_orig * escala)
    linhas_indices = np.clip((np.arange(novas_linhas) * (linhas_orig / novas_linhas)).astype(int), 0, linhas_orig - 1)
    cols_indices = np.clip((np.arange(novas_cols) * (cols_orig / novas_cols)).astype(int), 0, cols_orig - 1)
    img_redimensionada = imagem[linhas_indices[:, None], cols_indices]
    
    resultado = np.zeros((20, 20))
    center_r_novo, center_c_novo = novas_linhas // 2, novas_cols // 2
    start_r_src = max(0, center_r_novo - 10); end_r_src = min(novas_linhas, center_r_novo + 10)
    start_c_src = max(0, center_c_novo - 10); end_c_src = min(novas_cols, center_c_novo + 10)
    start_r_dst = max(0, 10 - (center_r_novo - start_r_src))
    end_r_dst = start_r_dst + (end_r_src - start_r_src)
    start_c_dst = max(0, 10 - (center_c_novo - start_c_src))
    end_c_dst = start_c_dst + (end_c_src - start_c_src)
    
    if (end_r_dst - start_r_dst > 0) and (end_c_dst - start_c_dst > 0):
        resultado[start_r_dst:end_r_dst, start_c_dst:end_c_dst] = img_redimensionada[start_r_src:end_r_src, start_c_src:end_c_src]
    return resultado

class EditorImagensApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroLab - Editor Final (Com Labels)")
        self.root.geometry("1450x950")

        # --- Estado dos Dados ---
        self.dados_raw = None
        self.dados_processados = None 
        self.labels = None # Lista de labels
        self.indice_atual = 0

        # --- VARIÁVEIS DE CONTROLO ---
        
        # 1. Global
        self.var_zoom = tk.DoubleVar(value=1.0)
        self.var_intensidade_global = tk.DoubleVar(value=1.0) 

        # 2. Geometria
        self.chk_transform = tk.BooleanVar(value=False)
        self.var_trans_x = tk.IntVar(value=0)
        self.var_trans_y = tk.IntVar(value=0)
        self.var_rotacao = tk.IntVar(value=0)

        # 3. Manual
        self.chk_manual = tk.BooleanVar(value=False)
        self.mascara_manual = np.zeros((20, 20), dtype=bool)
        self.var_modo_manual = tk.StringVar(value="branco")

        # 4. Linhas
        self.chk_linha = tk.BooleanVar(value=False)
        self.var_linhas_qtd = tk.IntVar(value=5)
        self.var_modo_linha = tk.StringVar(value="preto")
        self.var_dens_linha = tk.DoubleVar(value=100.0)
        self.var_direcao_linha = tk.StringVar(value="topo") 

        # 5. Círculo
        self.chk_circulo = tk.BooleanVar(value=False)
        self.var_raio = tk.DoubleVar(value=8.0)
        self.var_modo_circulo = tk.StringVar(value="ruido")
        self.var_dens_circulo = tk.DoubleVar(value=100.0)

        # 6. Grid
        self.chk_grid = tk.BooleanVar(value=False)
        self.var_grid_size = tk.IntVar(value=2)
        self.var_modo_grid = tk.StringVar(value="branco")
        self.var_dens_grid = tk.DoubleVar(value=100.0)
        self.grid_vars = []
        self.frame_grid_checkboxes = None

        # 7. Label Atual (Editável)
        self.var_label_atual = tk.StringVar(value="?")

        # Batch
        self.var_inicio_range = tk.IntVar(value=0)
        self.var_fim_range = tk.IntVar(value=0)

        # ================= LAYOUT =================

        main_container = tk.Frame(root)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Coluna de Controlo (Esquerda) com Scroll
        canvas_control = tk.Canvas(main_container, width=440)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas_control.yview)
        col_esq = tk.Frame(canvas_control)
        
        col_esq.bind("<Configure>", lambda e: canvas_control.configure(scrollregion=canvas_control.bbox("all")))
        canvas_control.create_window((0, 0), window=col_esq, anchor="nw")
        canvas_control.configure(yscrollcommand=scrollbar.set)
        
        canvas_control.pack(side="left", fill="y", padx=5)
        scrollbar.pack(side="left", fill="y")
        
        # Coluna de Visualização (Direita)
        col_dir = tk.Frame(main_container)
        col_dir.pack(side="left", fill="both", expand=True, padx=5)

        # === PAINEL DE CONTROLO ===
        tk.Label(col_esq, text="CONFIGURAÇÕES", font=("Arial", 12, "bold")).pack(pady=5)

        # [A] GLOBAL
        frame_glob = tk.LabelFrame(col_esq, text="1. Global", padx=5, pady=5)
        frame_glob.pack(fill="x", pady=2)
        tk.Label(frame_glob, text="Zoom:").pack(side="left")
        tk.Scale(frame_glob, from_=0.5, to=2.5, resolution=0.1, orient="horizontal", variable=self.var_zoom, command=self.upd).pack(side="left", fill="x", expand=True)
        tk.Label(frame_glob, text="Ruído:").pack(side="left")
        tk.Scale(frame_glob, from_=0.0, to=1.0, resolution=0.1, orient="horizontal", variable=self.var_intensidade_global, command=self.upd).pack(side="left", fill="x", expand=True)

        # [B] GEOMETRIA
        frame_geo = tk.LabelFrame(col_esq, text="2. Geometria", padx=5, pady=5, fg="purple")
        frame_geo.pack(fill="x", pady=5)
        tk.Checkbutton(frame_geo, text="Ativar", variable=self.chk_transform, command=self.upd).pack(anchor="w")
        f_geo1 = tk.Frame(frame_geo); f_geo1.pack(fill="x")
        tk.Label(f_geo1, text="X:").pack(side="left"); tk.Scale(f_geo1, from_=-10, to=10, orient="horizontal", variable=self.var_trans_x, command=self.upd).pack(side="left", fill="x", expand=True)
        tk.Label(f_geo1, text="Y:").pack(side="left"); tk.Scale(f_geo1, from_=-10, to=10, orient="horizontal", variable=self.var_trans_y, command=self.upd).pack(side="left", fill="x", expand=True)
        f_geo2 = tk.Frame(frame_geo); f_geo2.pack(fill="x")
        tk.Label(f_geo2, text="Ângulo:").pack(side="left"); tk.Scale(f_geo2, from_=-45, to=45, orient="horizontal", variable=self.var_rotacao, command=self.upd).pack(side="left", fill="x", expand=True)

        # [C] MANUAL
        frame_man = tk.LabelFrame(col_esq, text="3. Desenho Manual", padx=5, pady=5, fg="darkorange")
        frame_man.pack(fill="x", pady=5)
        f_man1 = tk.Frame(frame_man); f_man1.pack(fill="x")
        tk.Checkbutton(f_man1, text="Ativar", variable=self.chk_manual, command=self.upd).pack(side="left")
        tk.Button(f_man1, text="✏️ Editar Máscara", command=self.abrir_editor_manual).pack(side="right")
        f_man2 = tk.Frame(frame_man); f_man2.pack(fill="x", pady=2)
        tk.Label(f_man2, text="Modo:").pack(side="left")
        ttk.Combobox(f_man2, textvariable=self.var_modo_manual, values=["branco", "preto", "ruido"], width=7, state="readonly").pack(side="left")

        # [D] LINHAS
        frame_lin = tk.LabelFrame(col_esq, text="4. Linhas", padx=5, pady=5, fg="blue")
        frame_lin.pack(fill="x", pady=5)
        f_l1 = tk.Frame(frame_lin); f_l1.pack(fill="x")
        tk.Checkbutton(f_l1, text="Ativar", variable=self.chk_linha, command=self.upd).pack(side="left")
        ttk.Combobox(f_l1, textvariable=self.var_direcao_linha, values=["topo", "baixo"], width=6, state="readonly").pack(side="left", padx=5)
        tk.Label(f_l1, text="Qtd:").pack(side="left"); tk.Scale(f_l1, from_=0, to=20, orient="horizontal", variable=self.var_linhas_qtd, command=self.upd).pack(side="left", fill="x", expand=True)
        f_l2 = tk.Frame(frame_lin); f_l2.pack(fill="x", pady=2)
        tk.Label(f_l2, text="Modo:").pack(side="left"); ttk.Combobox(f_l2, textvariable=self.var_modo_linha, values=["ruido", "preto", "branco"], width=7, state="readonly").pack(side="left")
        tk.Label(f_l2, text="Dens%:").pack(side="left"); tk.Scale(f_l2, from_=0, to=100, orient="horizontal", variable=self.var_dens_linha, command=self.upd).pack(side="left", fill="x", expand=True)

        # [E] CÍRCULO
        frame_circ = tk.LabelFrame(col_esq, text="5. Círculo", padx=5, pady=5, fg="green")
        frame_circ.pack(fill="x", pady=5)
        f_c1 = tk.Frame(frame_circ); f_c1.pack(fill="x")
        tk.Checkbutton(f_c1, text="Ativar", variable=self.chk_circulo, command=self.upd).pack(side="left")
        tk.Label(f_c1, text="Raio:").pack(side="left"); tk.Scale(f_c1, from_=0, to=15, orient="horizontal", variable=self.var_raio, command=self.upd).pack(side="left", fill="x", expand=True)
        f_c2 = tk.Frame(frame_circ); f_c2.pack(fill="x", pady=2)
        tk.Label(f_c2, text="Modo:").pack(side="left"); ttk.Combobox(f_c2, textvariable=self.var_modo_circulo, values=["ruido", "preto", "branco"], width=7, state="readonly").pack(side="left")
        tk.Label(f_c2, text="Dens%:").pack(side="left"); tk.Scale(f_c2, from_=0, to=100, orient="horizontal", variable=self.var_dens_circulo, command=self.upd).pack(side="left", fill="x", expand=True)

        # [F] GRID
        frame_grid = tk.LabelFrame(col_esq, text="6. Grid/Matriz", padx=5, pady=5, fg="red")
        frame_grid.pack(fill="x", pady=5)
        f_g1 = tk.Frame(frame_grid); f_g1.pack(fill="x")
        tk.Checkbutton(f_g1, text="Ativar", variable=self.chk_grid, command=self.toggle_grid).pack(side="left")
        tk.Label(f_g1, text="Tam:").pack(side="left"); tk.Scale(f_g1, from_=2, to=5, orient="horizontal", variable=self.var_grid_size, command=self.gerar_grid).pack(side="left", fill="x", expand=True)
        f_g2 = tk.Frame(frame_grid); f_g2.pack(fill="x", pady=2)
        tk.Label(f_g2, text="Modo:").pack(side="left"); ttk.Combobox(f_g2, textvariable=self.var_modo_grid, values=["ruido", "preto", "branco"], width=7, state="readonly").pack(side="left")
        tk.Label(f_g2, text="Dens%:").pack(side="left"); tk.Scale(f_g2, from_=0, to=100, orient="horizontal", variable=self.var_dens_grid, command=self.upd).pack(side="left", fill="x", expand=True)
        self.frame_grid_container = tk.Frame(frame_grid, bg="#eee", bd=1, relief="sunken"); self.frame_grid_container.pack(fill="x", pady=2)
        self.gerar_grid()

        # [G] BATCH
        frame_batch = tk.LabelFrame(col_esq, text="7. Aplicar em Massa", padx=5, pady=10, bg="#fff9c4")
        frame_batch.pack(fill="x", pady=10)
        f_b = tk.Frame(frame_batch, bg="#fff9c4"); f_b.pack(fill="x")
        tk.Label(f_b, text="Idx Inicial:", bg="#fff9c4").pack(side="left")
        tk.Entry(f_b, textvariable=self.var_inicio_range, width=5).pack(side="left")
        tk.Label(f_b, text="Final:", bg="#fff9c4").pack(side="left", padx=5)
        tk.Entry(f_b, textvariable=self.var_fim_range, width=5).pack(side="left")
        tk.Button(frame_batch, text="APLICAR", command=self.aplicar_batch, bg="#8bc34a").pack(side="right")

        # === ÁREA DIREITA ===
        # Gestão de Ficheiros
        frame_files = tk.Frame(col_dir, bd=1, relief="raised")
        frame_files.pack(fill="x", pady=5)
        tk.Button(frame_files, text="1. Carregar Dados", command=self.load_data).pack(side="left", padx=5)
        tk.Button(frame_files, text="2. Carregar Labels", command=self.load_labels).pack(side="left", padx=5)
        tk.Button(frame_files, text="🔀 Misturar", command=self.shuffle_data, bg="yellow").pack(side="left", padx=15)
        
        # BOTÃO LIMPAR EM DESTAQUE
        tk.Button(frame_files, text="🗑️ Limpar Tudo", command=self.resetar_dados, bg="#ffcdd2", fg="red", font=("Arial", 9, "bold")).pack(side="right", padx=10)
        
        self.lbl_info = tk.Label(frame_files, text="Dados: 0 | Labels: 0", font=("Arial", 9, "bold"))
        self.lbl_info.pack(side="left", padx=10)
        
        # Gráficos
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=col_dir)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Navegação + Label + Busca
        frame_nav = tk.Frame(col_dir); frame_nav.pack(fill="x", pady=10)
        tk.Button(frame_nav, text="< Anterior", command=self.prev_img).pack(side="left")
        self.lbl_idx = tk.Label(frame_nav, text="0/0", font=("Arial", 12)); self.lbl_idx.pack(side="left", padx=15)
        tk.Button(frame_nav, text="Próxima >", command=self.next_img).pack(side="left")
        
        # LABEL ATUAL (EDITÁVEL)
        tk.Label(frame_nav, text="LABEL:", font=("Arial", 10, "bold"), fg="blue").pack(side="left", padx=(30, 5))
        self.entry_label = tk.Entry(frame_nav, textvariable=self.var_label_atual, width=5, font=("Arial", 11, "bold"), justify="center", fg="blue")
        self.entry_label.pack(side="left")
        tk.Button(frame_nav, text="Definir", command=self.salvar_label_manual, font=("Arial", 8)).pack(side="left", padx=2)

        # BUSCA
        tk.Label(frame_nav, text="Ir para idx:").pack(side="left", padx=(30, 5))
        self.entry_ir = tk.Entry(frame_nav, width=6)
        self.entry_ir.pack(side="left")
        self.entry_ir.bind('<Return>', self.ir_para_indice_manual)
        tk.Button(frame_nav, text="Ir", command=self.ir_para_indice_manual).pack(side="left", padx=2)

        tk.Button(frame_nav, text="💾 SALVAR TUDO", command=self.save_all, bg="#2196f3", fg="white", height=2).pack(side="right", padx=10)

    # --- Lógica de UI ---
    def upd(self, e=None): self.atualizar_visualizacao()
    
    def gerar_grid(self, e=None):
        for w in self.frame_grid_container.winfo_children(): w.destroy()
        if not self.chk_grid.get():
            tk.Label(self.frame_grid_container, text="(Grid Desativada)", bg="#eee").pack()
            return
        val = self.var_grid_size.get()
        if val == 3: val = 4; self.var_grid_size.set(4)
        self.grid_vars = []
        for r in range(val):
            row = tk.Frame(self.frame_grid_container, bg="#eee"); row.pack()
            r_vars = []
            for c in range(val):
                v = tk.BooleanVar(value=False)
                tk.Checkbutton(row, variable=v, command=self.upd, bg="#eee").pack(side="left")
                r_vars.append(v)
            self.grid_vars.append(r_vars)
        self.upd()

    def toggle_grid(self): self.gerar_grid(); self.upd()

    def abrir_editor_manual(self):
        top = tk.Toplevel(self.root)
        top.title("Editor Manual"); top.geometry("500x550")
        f = tk.Frame(top); f.pack(expand=True)
        self.btn_refs = []
        for r in range(20):
            row = []
            for c in range(20):
                bg = "black" if self.mascara_manual[r,c] else "white"
                btn = tk.Button(f, width=2, height=1, bg=bg, command=lambda rr=r, cc=c: self.toggle_man(rr, cc))
                btn.grid(row=r, column=c)
                row.append(btn)
            self.btn_refs.append(row)
        tk.Button(top, text="Fechar e Atualizar", command=lambda:[top.destroy(), self.upd()], bg="green").pack(pady=5)

    def toggle_man(self, r, c):
        self.mascara_manual[r,c] = not self.mascara_manual[r,c]
        cor = "black" if self.mascara_manual[r,c] else "white"
        self.btn_refs[r][c].config(bg=cor)

    # --- Processamento ---
    def processar_imagem(self, img_flat):
        img = img_flat.reshape(20, 20).copy()
        
        if self.chk_transform.get():
            rot = self.var_rotacao.get()
            if rot != 0 and SCIPY_AVAILABLE:
                img = ndimage.rotate(img, rot, reshape=False, order=1)
                img = np.clip(img, 0.0, 1.0)
            tx, ty = self.var_trans_x.get(), self.var_trans_y.get()
            if tx != 0 or ty != 0: img = transladar_imagem(img, tx, ty)

        zoom = self.var_zoom.get()
        if zoom != 1.0: img = redimensionar_imagem_numpy(img, zoom)

        if self.chk_manual.get():
            img = self.aplicar_efeito(img, self.mascara_manual, self.var_modo_manual.get(), 100.0)

        if self.chk_linha.get():
            mask = np.zeros((20, 20), dtype=bool)
            qtd = self.var_linhas_qtd.get()
            if self.var_direcao_linha.get() == "topo": mask[:qtd, :] = True
            else: mask[max(0, 20-qtd):, :] = True
            img = self.aplicar_efeito(img, mask, self.var_modo_linha.get(), self.var_dens_linha.get())

        if self.chk_circulo.get():
            y, x = np.ogrid[:20, :20]
            mask = np.sqrt((x-9.5)**2 + (y-9.5)**2) > self.var_raio.get()
            img = self.aplicar_efeito(img, mask, self.var_modo_circulo.get(), self.var_dens_circulo.get())

        if self.chk_grid.get() and self.grid_vars:
            mask = np.zeros((20, 20), dtype=bool)
            n = len(self.grid_vars)
            divs = np.linspace(0, 20, n+1, dtype=int)
            for r in range(n):
                for c in range(n):
                    if r < len(self.grid_vars) and c < len(self.grid_vars[r]):
                        if self.grid_vars[r][c].get():
                            mask[divs[r]:divs[r+1], divs[c]:divs[c+1]] = True
            img = self.aplicar_efeito(img, mask, self.var_modo_grid.get(), self.var_dens_grid.get())

        return img

    def aplicar_efeito(self, img, mask, modo, densidade):
        if not np.any(mask): return img
        mask_final = np.logical_and(mask, np.random.rand(20,20) < (densidade/100.0))
        res = img.copy()
        if modo == "preto": res[mask_final] = 0.0
        elif modo == "branco": res[mask_final] = 1.0
        else: # ruido
            ruido = np.random.rand(20,20) * self.var_intensidade_global.get()
            res[mask_final] = ruido[mask_final]
        return res

    # --- Gestão de Dados ---
    def atualizar_visualizacao(self):
        if self.dados_raw is None: return
        idx = self.indice_atual
        
        # Atualizar Label
        lbl_txt = "?"
        if self.labels and idx < len(self.labels):
            try: lbl_txt = str(self.labels[idx][0])
            except: pass
        self.var_label_atual.set(lbl_txt)

        self.ax1.clear(); self.ax1.imshow(self.dados_raw[idx].reshape(20,20), cmap='gray', vmin=0, vmax=1); self.ax1.axis('off'); self.ax1.set_title("Original")
        self.ax2.clear(); self.ax2.imshow(self.dados_processados[idx].reshape(20,20), cmap='gray', vmin=0, vmax=1); self.ax2.axis('off'); self.ax2.set_title("Salvo")
        prev = self.processar_imagem(self.dados_raw[idx])
        self.ax3.clear(); self.ax3.imshow(prev, cmap='gray', vmin=0, vmax=1); self.ax3.axis('off'); self.ax3.set_title("Preview")
        self.canvas.draw()
        self.lbl_idx.config(text=f"{idx} / {len(self.dados_raw)-1}")

    def salvar_label_manual(self):
        if self.labels and self.indice_atual < len(self.labels):
            self.labels[self.indice_atual] = [self.var_label_atual.get()]
            self.atualizar_visualizacao() # Para confirmar visualmente

    def aplicar_batch(self):
        if self.dados_processados is None: return
        try: ini, fim = self.var_inicio_range.get(), self.var_fim_range.get()
        except: return
        if ini < 0 or fim >= len(self.dados_raw): return
        for i in range(ini, fim+1):
            res = self.processar_imagem(self.dados_raw[i])
            self.dados_processados[i] = res.flatten()
        self.upd()
        messagebox.showinfo("OK", "Aplicado!")

    def load_data(self):
        fns = filedialog.askopenfilenames(filetypes=[("CSV", "*.csv")])
        if not fns: return
        data = []
        for fn in fns:
            try:
                with open(fn, 'r') as f:
                    for row in csv.reader(f):
                        if not row: continue
                        try: data.append([float(x) for x in row])
                        except: pass
            except: pass
        if data:
            arr = np.array(data)
            self.dados_raw = arr if self.dados_raw is None else np.vstack((self.dados_raw, arr))
            self.dados_processados = arr.copy() if self.dados_processados is None else np.vstack((self.dados_processados, arr.copy()))
            self.var_fim_range.set(len(self.dados_raw)-1)
            self.atualizar_info_labels()
            self.upd()

    def load_labels(self):
        fns = filedialog.askopenfilenames(filetypes=[("CSV", "*.csv")])
        if not fns: return
        lbls = []
        for fn in fns:
            try:
                with open(fn, 'r') as f:
                    for row in csv.reader(f):
                        if row and not row[0].lower().startswith("label"): lbls.append(row)
            except: pass
        if lbls:
            self.labels = lbls if self.labels is None else self.labels + lbls
            self.atualizar_info_labels()
            self.upd()

    def atualizar_info_labels(self):
        d = len(self.dados_raw) if self.dados_raw is not None else 0
        l = len(self.labels) if self.labels is not None else 0
        self.lbl_info.config(text=f"Dados: {d} | Labels: {l}", fg="green" if d==l and d>0 else "red")

    def resetar_dados(self):
        if not self.dados_raw is None:
            if not messagebox.askyesno("Confirmar Limpeza", "Tem a certeza? Isto apaga todos os dados da memória."):
                return
        self.dados_raw = None; self.dados_processados = None; self.labels = None
        self.indice_atual = 0
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear(); self.canvas.draw()
        self.lbl_info.config(text="Dados: 0 | Labels: 0", fg="black")
        self.var_label_atual.set("?")

    def shuffle_data(self):
        if self.dados_raw is None: return
        if self.labels and len(self.labels) != len(self.dados_raw):
            messagebox.showwarning("Aviso", "Número de labels diferente do número de imagens. Pode haver dessincronização.")
        
        p = np.random.permutation(len(self.dados_raw))
        self.dados_raw = self.dados_raw[p]
        self.dados_processados = self.dados_processados[p]
        if self.labels and len(self.labels) == len(p): 
            # Só faz shuffle das labels se tiverem o mesmo tamanho, senão crasha
            self.labels = [self.labels[i] for i in p]
        
        self.indice_atual = 0; self.upd()

    def save_all(self):
        if self.dados_processados is None: return
        fn = filedialog.asksaveasfilename(title="Salvar Dados", defaultextension=".csv")
        if fn:
            with open(fn, 'w', newline='') as f:
                csv.writer(f).writerows(self.dados_processados)
            
            if self.labels and messagebox.askyesno("Labels", "Salvar Labels também?"):
                fn_l = filedialog.asksaveasfilename(title="Salvar Labels", defaultextension=".csv")
                if fn_l:
                    with open(fn_l, 'w', newline='') as f: csv.writer(f).writerows(self.labels)
            messagebox.showinfo("Sucesso", "Salvo.")

    def next_img(self):
        if self.dados_raw is not None and self.indice_atual < len(self.dados_raw)-1:
            self.indice_atual+=1; self.upd()
    def prev_img(self):
        if self.dados_raw is not None and self.indice_atual > 0:
            self.indice_atual-=1; self.upd()
    def ir_para_indice_manual(self, e=None):
        try:
            val = int(self.entry_ir.get())
            if self.dados_raw is not None and 0 <= val < len(self.dados_raw):
                self.indice_atual = val; self.upd()
        except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = EditorImagensApp(root)
    root.mainloop()