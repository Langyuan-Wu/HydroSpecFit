import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import time

# App Appearance Settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Disable standard Matplotlib Toolbar
plt.rcParams['toolbar'] = 'None' 

# ==========================================
# --- 1. Physics Engine ---
# ==========================================

def q0_cal(dn):
    return (1+1j) / dn

def q1_cal(q0, qsi):
    qsi = np.maximum(qsi, 1e-12) 
    return np.sqrt(q0**2 + (1/(qsi**2)))

def dalta_model(dn, thata, qsi_val, h_val, Liquid_Density, Quartz_Density, Quartz_Viscosity):
    try:
        q0 = q0_cal(dn)
        q1 = q1_cal(q0, qsi_val)
 
        arg = q1 * h_val
        if np.max(np.real(arg)) > 100: return None, None

        ch = np.cosh(arg)
        sh = np.sinh(arg)   
        A_val = q1 * ch + q0 * sh
        
        if np.any(np.abs(A_val) < 1e-15): return None, None

        part1 = 1/q0
        part2 = h_val / ((qsi_val * q1)**2)
        part3a = 1 / (A_val * ((qsi_val * q1)**2))
        part3b = ((2 * q0) / q1) * (ch - 1) + sh
        part3 = part3a * part3b
        val_total = part1 + part2 - part3
        
        coeff = 1 / np.sqrt(Quartz_Viscosity * Quartz_Density)
        
        part1_eq = -(thata) * coeff
        part3_eq = ((1 - thata)) * coeff

        term_A = 2 * part1_eq * np.real(val_total) - 2 * part3_eq * np.real(1/q0)
        term_B = 4 * part1_eq * np.imag(val_total) - 4 * part3_eq * np.imag(1/q0)
        
        norm_factor = 1e12 
        
        DF = term_A * norm_factor
        DW = term_B * norm_factor
        return DW, DF

    except Exception:
        return None, None

def model_Kanazawa_line(dn, Quartz_Density, Quartz_Viscosity):
    try:
        q0 = q0_cal(dn)  
        part1_eq = -1/ np.sqrt(Quartz_Viscosity * Quartz_Density)      
        term_A = 2 * part1_eq * np.real(1/q0) 
        term_B = 4 * part1_eq * np.imag(1/q0) 
        
        return term_B * 1e12, term_A * 1e12 # DW, DF
    except: return None, None

def find_cutoff(dn_raw_co, dw_test_co, thata, Liquid_Density, Quartz_Density, Quartz_Viscosity):
    try:
        max_scan = dn_raw_co[-1] 
        calculated_noise_floor = (np.mean(dw_test_co)/1000)

        h_scan_um = np.linspace(0.001, max_scan, 2000)
        h_scan_nm = h_scan_um * 1e3
        h_scan_meters = h_scan_nm * 1e-9 
        
        dn_input = max_scan * 1e-6 
        
        w_soft, _ = dalta_model(dn_input, thata, 5e-9, h_scan_meters, Liquid_Density, Quartz_Density, Quartz_Viscosity)
        w_hard, _ = dalta_model(dn_input, thata, 20e-9, h_scan_meters, Liquid_Density, Quartz_Density, Quartz_Viscosity)

        if w_soft is None or w_hard is None: return None

        diff_array = np.abs(w_soft - w_hard)
        valid_indices = np.where(diff_array > calculated_noise_floor)[0]
        
        if len(valid_indices) > 0:
            return h_scan_nm[valid_indices[0]]
        else:
            return None
    except:
        return None

# ==========================================
# --- 2. Custom Toolbar Class ---
# ==========================================
class CustomVerticalToolbar(NavigationToolbar2Tk):
    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
    )

    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        
        bg_color = "#404040" 
        self.config(background=bg_color)
        
        self._message_label.pack_forget()

        for widget in self.winfo_children():
            if widget != self._message_label:
                widget.config(background=bg_color)
                widget.pack_forget()
                widget.pack(side="top", pady=5, padx=2, fill="x")

    def set_message(self, s):
        pass

# ==========================================
# --- 3. Window Classes (Graphs & Tools) ---
# ==========================================

class DynamicParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent_app, run_callback):
        super().__init__()
        self.title("Dynamic Parameters Configuration")
        self.geometry("500x650") 
        self.parent_app = parent_app
        self.run_callback = run_callback
        
        self.visc_ranges = [] 
        self.theta_ranges = []

        ctk.CTkLabel(self, text="Define Independent Parameter Ranges", font=("Arial", 16, "bold")).pack(pady=(15, 5))
        
        ctk.CTkLabel(self, text="Add time ranges here to override global parameters.\nValues are auto-clamped to experiment duration.", 
                     text_color="gray", font=("Arial", 12)).pack(pady=(0, 10))

        self.tabview = ctk.CTkTabview(self, width=460, height=450)
        self.tabview.pack(pady=5, padx=20, fill="both", expand=True)

        self.tabview.add("Viscosity")
        self.tabview.add("Theta")

        self.setup_visc_tab()
        self.setup_theta_tab()

        self.btn_run_multi = ctk.CTkButton(self, text="Run Optimization with Dynamic Params", 
                                           command=self.trigger_run, 
                                           fg_color="green", height=45, font=("Arial", 14, "bold"))
        self.btn_run_multi.pack(pady=20, padx=20, fill="x", side="bottom")
        
        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def _get_time_bounds(self):
        if self.parent_app.df is None:
            return 0, float('inf')
        
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        for col in self.parent_app.df.columns:
            if any(t.lower() in col.lower() for t in possible_time_cols):
                try:
                    vals = pd.to_numeric(self.parent_app.df[col], errors='coerce').dropna().values
                    if len(vals) > 0:
                        return np.min(vals), np.max(vals)
                except:
                    continue
        return 0, float('inf')

    def _on_focus_out(self, entry_widget):
        try:
            val_str = entry_widget.get()
            if not val_str: return 
            
            val = float(val_str)
            min_t, max_t = self._get_time_bounds()
            
            new_val = val
            if val < min_t: new_val = min_t
            if val > max_t: new_val = max_t
            
            if new_val != val:
                entry_widget.delete(0, 'end')
                entry_widget.insert(0, f"{new_val:.2f}")
                
        except ValueError:
            pass 

    def setup_visc_tab(self):
        tab = self.tabview.tab("Viscosity")
        
        input_frame = ctk.CTkFrame(tab)
        input_frame.pack(pady=10) 

        ctk.CTkLabel(input_frame, text="Start (s)").grid(row=0, column=0, padx=5)
        ctk.CTkLabel(input_frame, text="End (s)").grid(row=0, column=1, padx=5)
        ctk.CTkLabel(input_frame, text="Viscosity (Pa·s)").grid(row=0, column=2, padx=5)

        self.v_start = ctk.CTkEntry(input_frame, width=80)
        self.v_start.grid(row=1, column=0, padx=5, pady=5)
        self.v_start.bind("<FocusOut>", lambda e: self._on_focus_out(self.v_start))

        self.v_end = ctk.CTkEntry(input_frame, width=80)
        self.v_end.grid(row=1, column=1, padx=5, pady=5)
        self.v_end.bind("<FocusOut>", lambda e: self._on_focus_out(self.v_end))

        self.v_val = ctk.CTkEntry(input_frame, width=80)
        self.v_val.grid(row=1, column=2, padx=5, pady=5)

        btn_add = ctk.CTkButton(input_frame, text="Add Visc Range", command=self.add_visc, width=120)
        btn_add.grid(row=2, column=0, columnspan=3, pady=10)

        ctk.CTkFrame(tab, height=2, fg_color="gray40").pack(fill="x", padx=30, pady=5)

        self.visc_scroll = ctk.CTkScrollableFrame(tab)
        self.visc_scroll.pack(pady=5, fill="both", expand=True)
        self.refresh_visc_list()

    def setup_theta_tab(self):
        tab = self.tabview.tab("Theta")
        
        input_frame = ctk.CTkFrame(tab)
        input_frame.pack(pady=10) 

        ctk.CTkLabel(input_frame, text="Start (s)").grid(row=0, column=0, padx=5)
        ctk.CTkLabel(input_frame, text="End (s)").grid(row=0, column=1, padx=5)
        ctk.CTkLabel(input_frame, text="Theta (0-1)").grid(row=0, column=2, padx=5)

        self.t_start = ctk.CTkEntry(input_frame, width=80)
        self.t_start.grid(row=1, column=0, padx=5, pady=5)
        self.t_start.bind("<FocusOut>", lambda e: self._on_focus_out(self.t_start))

        self.t_end = ctk.CTkEntry(input_frame, width=80)
        self.t_end.grid(row=1, column=1, padx=5, pady=5)
        self.t_end.bind("<FocusOut>", lambda e: self._on_focus_out(self.t_end))

        self.t_val = ctk.CTkEntry(input_frame, width=80)
        self.t_val.grid(row=1, column=2, padx=5, pady=5)

        btn_add = ctk.CTkButton(input_frame, text="Add Theta Range", command=self.add_theta, width=120)
        btn_add.grid(row=2, column=0, columnspan=3, pady=10)

        ctk.CTkFrame(tab, height=2, fg_color="gray40").pack(fill="x", padx=30, pady=5)

        self.theta_scroll = ctk.CTkScrollableFrame(tab)
        self.theta_scroll.pack(pady=5, fill="both", expand=True)
        self.refresh_theta_list()

    def add_visc(self):
        try:
            s = float(self.v_start.get())
            e = float(self.v_end.get())
            v = float(self.v_val.get())
            
            min_t, max_t = self._get_time_bounds()
            if s < min_t: s = min_t
            if e > max_t: e = max_t

            if s >= e or v <= 0: 
                messagebox.showerror("Error", "Invalid Time Range or Viscosity.")
                return
            
            for r in self.visc_ranges:
                if s < r['end'] and e > r['start']:
                    messagebox.showerror("Error", "Viscosity range overlap detected!")
                    return
            
            self.visc_ranges.append({'start': s, 'end': e, 'val': v})
            self.visc_ranges.sort(key=lambda x: x['start'])
            self.refresh_visc_list()
            self.v_start.delete(0, 'end'); self.v_end.delete(0, 'end'); self.v_val.delete(0, 'end')
        except ValueError: 
            messagebox.showerror("Error", "Invalid numeric inputs.")

    def refresh_visc_list(self):
        for w in self.visc_scroll.winfo_children(): w.destroy()
        if not self.visc_ranges: ctk.CTkLabel(self.visc_scroll, text="No Viscosity ranges.").pack()
        else:
             for i, r in enumerate(self.visc_ranges):
                f = ctk.CTkFrame(self.visc_scroll)
                f.pack(fill="x", pady=2)
                ctk.CTkLabel(f, text=f"{r['start']:.1f}-{r['end']:.1f}s: {r['val']} Pa.s").pack(side="left", padx=10)
                ctk.CTkButton(f, text="X", width=30, fg_color="red", command=lambda idx=i: self.del_visc(idx)).pack(side="right", padx=5)

    def del_visc(self, idx):
        del self.visc_ranges[idx]
        self.refresh_visc_list()

    def add_theta(self):
        try:
            s = float(self.t_start.get())
            e = float(self.t_end.get())
            v = float(self.t_val.get())
            
            min_t, max_t = self._get_time_bounds()
            if s < min_t: s = min_t
            if e > max_t: e = max_t

            if s >= e: 
                messagebox.showerror("Error", "Start time must be less than End time.")
                return
            
            for r in self.theta_ranges:
                if s < r['end'] and e > r['start']:
                    messagebox.showerror("Error", "Theta range overlap detected!")
                    return
            
            self.theta_ranges.append({'start': s, 'end': e, 'val': v})
            self.theta_ranges.sort(key=lambda x: x['start'])
            self.refresh_theta_list()
            self.t_start.delete(0, 'end'); self.t_end.delete(0, 'end'); self.t_val.delete(0, 'end')
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric inputs.")

    def refresh_theta_list(self):
        for w in self.theta_scroll.winfo_children(): w.destroy()
        if not self.theta_ranges: ctk.CTkLabel(self.theta_scroll, text="No Theta ranges.").pack()
        else:
             for i, r in enumerate(self.theta_ranges):
                f = ctk.CTkFrame(self.theta_scroll)
                f.pack(fill="x", pady=2)
                ctk.CTkLabel(f, text=f"{r['start']:.1f}-{r['end']:.1f}s: Theta={r['val']}").pack(side="left", padx=10)
                ctk.CTkButton(f, text="X", width=30, fg_color="red", command=lambda idx=i: self.del_theta(idx)).pack(side="right", padx=5)

    def del_theta(self, idx):
        del self.theta_ranges[idx]
        self.refresh_theta_list()

    def trigger_run(self):
        config = {
            'visc': self.visc_ranges,
            'theta': self.theta_ranges
        }
        self.destroy()
        self.parent_app.after(200, lambda: self.run_callback(config))


# === COMBINED ANALYSIS WINDOW ===
class CombinedGraphWindow(ctk.CTkToplevel):
    def __init__(self, df, harmonics, filename_base, on_click_callback, cycle_times=None, cycle_indices=None, mode_label="H"):
        super().__init__()
        self.title("Combined Analysis: F, D, h, Xi")
        self.geometry("1300x950") 
        
        self.df = df
        self.filename_base = filename_base
        self.on_click_callback = on_click_callback
        self.cycle_times = cycle_times
        self.cycle_indices = cycle_indices
        self.mode_label = mode_label
        
        self.x_data = None
        self.x_label = "Index"
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        for col in df.columns:
            if any(t.lower() in col.lower() for t in possible_time_cols):
                self.x_data = pd.to_numeric(df[col], errors='coerce').fillna(0).to_numpy()
                self.x_label = col
                break
        if self.x_data is None: self.x_data = df.index.to_numpy()

        # BASE ABSOLUTE H
        self.y_height_abs = pd.to_numeric(df["Optimized_Height_nm"], errors='coerce').to_numpy()
        self.y_qsi = pd.to_numeric(df["Optimized_Qsi_nm"], errors='coerce').to_numpy()
        
        self.y_theo = None
        if "Theo_Calibrated_Active" in df.columns:
            self.y_theo = pd.to_numeric(df["Theo_Calibrated_Active"], errors='coerce').to_numpy()
            
        self.y_voltage = None
        if "E_we_V" in df.columns:
            self.y_voltage = pd.to_numeric(df["E_we_V"], errors='coerce').to_numpy()

        # --- LAYOUT ---
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, pady=(0, 10))

        toolbar_frame = ctk.CTkFrame(main_container, width=45, corner_radius=0, fg_color="#404040")
        toolbar_frame.pack(side="left", fill="y")

        graph_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        graph_frame.pack(side="right", fill="both", expand=True)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(side="bottom", pady=10, fill="x")
        
        # --- BUTTONS ---
        btn_save_img = ctk.CTkButton(btn_frame, text="Save Graph", command=self.save_img, fg_color="#1f538d", width=120)
        btn_save_img.pack(side="left", padx=40)
        
        btn_save_xls = ctk.CTkButton(btn_frame, text="Save Data", command=self.save_xls, fg_color="green", width=120)
        btn_save_xls.pack(side="right", padx=40)

        # --- H MODE SELECTOR (CENTER) ---
        self.h_view_var = ctk.StringVar(value="Absolute H")
        self.seg_h_view = ctk.CTkSegmentedButton(btn_frame, values=["Absolute H", "Relative H", "H Accumulated"],
                                                 command=self.update_h_plot, variable=self.h_view_var)
        self.seg_h_view.place(relx=0.5, rely=0.5, anchor="center")

        # Plotting Setup
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 9), sharex=False)
        self.fig.subplots_adjust(right=0.85, hspace=0.3, top=0.95, bottom=0.05, left=0.1) 
        
        self.legend_pos = (1.05, 1)
        self.harmonics = harmonics

        # --- CREATE CANVAS FIRST (FIXED) ---
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # --- CREATE TOOLBAR ---
        self.toolbar = CustomVerticalToolbar(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="y") 

        # --- NOW CALL PLOTTING FUNCTIONS ---
        self.plot_static_graphs()
        self.update_h_plot("Absolute H")

        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def plot_static_graphs(self):
        # Color Logic
        f_colors_fixed = {3: "#0819CE", 5: "#1143E5", 7: "#2E71E9", 9: "#68A9E8", 11: "#6DBCE4", 13: "#8CD3E6"}
        d_colors_fixed = {3: "#FF2C14", 5: "#FD5328", 7: "#FD753A", 9: "#FAA56D", 11: "#FBB261", 13: "#FBC873"}

        # Plot 1: F
        self.ax1.set_ylabel(r"$\Delta f_n / n$ [Hz]", fontsize=9)
        if self.y_voltage is not None:
            self.ax1_twin = self.ax1.twinx()
            self.ax1_twin.set_ylabel("E [V]", color='#333333', fontsize=9)
            self.ax1_twin.plot(self.x_data, self.y_voltage, color='#333333', alpha=0.9, linewidth=1.2, linestyle='-', label='E [V]')
            self.ax1_twin.tick_params(axis='y', labelcolor='#333333')

        for n in self.harmonics:
            col_name = f"f{n}"
            if col_name in self.df.columns:
                raw_f = pd.to_numeric(self.df[col_name], errors='coerce').fillna(0).to_numpy()
                norm_f = (raw_f - raw_f[0]) / n
                c = f_colors_fixed.get(n, plt.cm.Blues(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax1.plot(self.x_data, norm_f, color=c, linewidth=1.0, alpha=0.7, label=f"f{n}")
        if self.y_theo is not None:
            self.ax1.plot(self.x_data, self.y_theo, color='black', linestyle='-.', linewidth=1.5, label=r'$\Delta f_{theo} (Calib)$')
            
        lines, labels = self.ax1.get_legend_handles_labels()
        if self.y_voltage is not None:
            lines2, labels2 = self.ax1_twin.get_legend_handles_labels()
            lines += lines2; labels += labels2
        self.ax1.legend(lines, labels, bbox_to_anchor=self.legend_pos, loc='upper left', borderaxespad=0, fontsize='small')
        self.ax1.grid(True, linestyle='--', alpha=0.5)

        # Plot 2: D
        self.ax2.set_ylabel(r"$\Delta D_n$ [ppm]", fontsize=9)
        for n in self.harmonics:
            col_name = f"D{n}"
            if col_name not in self.df.columns: col_name = f"d{n}"
            if col_name in self.df.columns:
                raw_d = pd.to_numeric(self.df[col_name], errors='coerce').fillna(0).to_numpy()
                delta_d = (raw_d - raw_d[0]) 
                c = d_colors_fixed.get(n, plt.cm.YlOrRd(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax2.plot(self.x_data, delta_d, color=c, linewidth=1.2, label=rf"$\Delta D_{{{n}}}$")
        self.ax2.legend(bbox_to_anchor=self.legend_pos, loc='upper left', borderaxespad=0, fontsize='small')
        self.ax2.grid(True, linestyle='--', alpha=0.5)

        # Plot 4: Xi
        self.ax4.set_ylabel(r"$\xi$ [nm]", fontsize=9)
        self.ax4.plot(self.x_data, self.y_qsi, color='#ff7f0e', linewidth=1.5, label=r'Correlation ($\xi$)')
        self.ax4.set_xlabel(self.x_label)
        self.ax4.legend(bbox_to_anchor=self.legend_pos, loc='upper left', borderaxespad=0, fontsize='small')
        self.ax4.grid(True, linestyle='--', alpha=0.5)

        # Cycle Lines
        if self.cycle_times is not None:
            for ct in self.cycle_times:
                self.ax1.axvline(x=ct, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    def calculate_relative_h(self):
        """Logic: For each cycle, find min H, shift cycle down."""
        if self.cycle_indices is None or len(self.cycle_indices) < 3:
            return self.y_height_abs

        h_rel = np.copy(self.y_height_abs)
        for i in range(0, len(self.cycle_indices) - 2, 2):
            idx_start = self.cycle_indices[i]
            idx_end = self.cycle_indices[i+2]
            
            cycle_slice = h_rel[idx_start : idx_end + 1]
            if len(cycle_slice) > 0:
                # Filter out None/NaN for min calculation
                valid_vals = [v for v in cycle_slice if v is not None and not np.isnan(v)]
                if valid_vals:
                    min_val = np.min(valid_vals)
                    new_slice = []
                    for v in cycle_slice:
                        if v is not None and not np.isnan(v):
                            new_slice.append(v - min_val)
                        else:
                            new_slice.append(v)
                    h_rel[idx_start : idx_end + 1] = new_slice
        return h_rel

    def update_h_plot(self, mode):
        self.ax3.clear()
        self.ax3.set_ylabel(r"$H$ [nm]", fontsize=9)
        self.ax3.grid(True, linestyle='--', alpha=0.5)
        
        if mode == "Absolute H":
            y_data = self.y_height_abs
            label_txt = r'Total Height ($H_{total}$)'
            color = '#1f77b4'
        elif mode == "Relative H":
            y_data = self.calculate_relative_h()
            label_txt = r'Relative Height ($H_{rel}$)'
            color = '#2ca02c' 
        else: # H Accumulated
            # Calculation: Absolute - Relative
            abs_data = self.y_height_abs
            rel_data = self.calculate_relative_h()
            
            # Robust subtraction (handling Nones)
            y_data = []
            for a, r in zip(abs_data, rel_data):
                if a is not None and r is not None and not np.isnan(a) and not np.isnan(r):
                    y_data.append(a - r)
                else:
                    y_data.append(None)
            
            label_txt = r'$H_{Accumulated}$'
            color = '#d62728' # Red color for the new graph

        self.ax3.plot(self.x_data, y_data, color=color, linewidth=1.5, label=label_txt)
        
        if self.cycle_times is not None:
            for ct in self.cycle_times:
                self.ax3.axvline(x=ct, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
                
        self.ax3.legend(bbox_to_anchor=self.legend_pos, loc='upper left', borderaxespad=0, fontsize='small')
        self.canvas.draw()

    def on_plot_click(self, event):
        if self.toolbar.mode != "": return
        is_ax1 = (event.inaxes == self.ax1)
        if hasattr(self, 'ax1_twin'):
            is_ax1 = is_ax1 or (event.inaxes == self.ax1_twin)
            
        if (is_ax1 or event.inaxes in [self.ax2, self.ax3, self.ax4]) and event.xdata is not None:
            distances = np.abs(self.x_data - event.xdata)
            nearest_idx = distances.argmin()
            self.on_click_callback(nearest_idx)

    def save_img(self):
        # CHANGE 1: APPEND MODE TO FILENAME
        current_mode = self.h_view_var.get()
        mode_suffix = current_mode.replace(" ", "_")
        
        default_name = f"{self.filename_base}_Combined_Analysis_{mode_suffix}"
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            initialfile=default_name,
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.fig.savefig(path)
            messagebox.showinfo("Saved", "Graph saved successfully.")

    def save_xls(self):
        # 1. Determine Mode
        current_mode = self.h_view_var.get()
        mode_suffix = "Absolute"
        if "Relative" in current_mode: mode_suffix = "Relative"
        elif "Accumulated" in current_mode: mode_suffix = "H_Accumulated"
        
        # 2. Prepare Data
        export_df = self.df.copy()
        
        if mode_suffix == "Relative":
            # Calculate and overwrite with relative data
            rel_data = self.calculate_relative_h()
            export_df["Optimized_Height_nm"] = rel_data
        elif mode_suffix == "H_Accumulated":
            # Calculate Baseline data
            abs_data = self.y_height_abs
            rel_data = self.calculate_relative_h()
            base_data = []
            for a, r in zip(abs_data, rel_data):
                if a is not None and r is not None and not np.isnan(a) and not np.isnan(r):
                    base_data.append(a - r)
                else:
                    base_data.append(None)
            export_df["Optimized_Height_nm"] = base_data
            
        # 3. Rename Column for Clarity
        col_name = f"Optimized_Height_nm_{mode_suffix}"
        export_df.rename(columns={"Optimized_Height_nm": col_name}, inplace=True)
        
        # 4. Save
        default_name = f"{self.filename_base}_Final_Results_{mode_suffix}_H"
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", 
                                            initialfile=default_name,
                                            filetypes=[("Excel", "*.xlsx")])
        if path:
            export_df.to_excel(path, index=False)
            messagebox.showinfo("Saved", f"Data saved as {mode_suffix} H successfully.")

# === ROW WINDOW CLASS ===
class RowGraphWindow(ctk.CTkToplevel):
    def __init__(self, idx, time_val, best_h, best_xi, 
                 dn_model, f_model, w_model, 
                 dn_exp, f_exp, w_exp,
                 kana_f, kana_w,
                 filename_base, export_data_dict):
        super().__init__()
        self.title(f"Row {idx} Analysis")
        self.geometry("900x800")
        
        self.filename_base = filename_base
        self.idx = idx
        self.export_data_dict = export_data_dict

        # Layout
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, pady=(0, 10))

        toolbar_frame = ctk.CTkFrame(main_container, width=45, corner_radius=0, fg_color="#404040")
        toolbar_frame.pack(side="left", fill="y")

        graph_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        graph_frame.pack(side="right", fill="both", expand=True)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(side="bottom", pady=10, fill="x")

        # Buttons
        btn_save_img = ctk.CTkButton(btn_frame, text="Save Graph (Image)", command=self.save_img, fg_color="#1f538d")
        btn_save_img.pack(side="left", padx=20, expand=True)
        
        btn_save_xls = ctk.CTkButton(btn_frame, text="Save Data (Excel)", command=self.save_xls, fg_color="green")
        btn_save_xls.pack(side="left", padx=20, expand=True)
        
        # Plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8))
        plt.subplots_adjust(bottom=0.15, left=0.15, right=0.95)
        
        self.fig.suptitle(fr"Row {idx} | Time: {time_val}sec" + "\n" + r"$H$=" + f"{best_h:.1f}, " + r"$\xi$=" + f"{best_xi:.1f}")
        
        # Freq
        # CHANGE 2: Mathematical Labels
        self.ax1.plot(dn_model*1e6, f_model, 'b-', label='Model')
        if kana_f is not None: self.ax1.plot(dn_model*1e6, kana_f, 'k--', label='Kanazawa', alpha=0.7)
        self.ax1.plot(dn_exp, f_exp, 'bo', label=r'Exp ($\Delta f_{calib}$)')
        self.ax1.set_ylabel(r'$\Delta f / n$ [Hz]'); self.ax1.grid(True); self.ax1.legend()
        
        # Dissipation
        self.ax2.plot(dn_model*1e6, w_model, 'r-', label='Model')
        if kana_w is not None: self.ax2.plot(dn_model*1e6, kana_w, 'k--', label='Kanazawa', alpha=0.7)
        self.ax2.plot(dn_exp, w_exp, 'ro', label='Exp')
        self.ax2.set_ylabel(r'$\Delta W$ [ppm]'); self.ax2.set_xlabel(r'Penetration Depth $\delta$ [$\mu m$]')
        self.ax2.grid(True); self.ax2.legend()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        self.toolbar = CustomVerticalToolbar(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="y") 

        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def save_img(self):
        default_name = f"{self.filename_base}_AUTOFIT_Row_{self.idx}_Graph"
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            initialfile=default_name,
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.fig.savefig(path)
            messagebox.showinfo("Saved", "Row Graph saved successfully.")

    def save_xls(self):
        df_export = pd.DataFrame(self.export_data_dict)
        default_name = f"{self.filename_base}_AUTOFIT_Row_{self.idx}_Data"
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", 
                                            initialfile=default_name,
                                            filetypes=[("Excel", "*.xlsx")])
        if path:
            df_export.to_excel(path, index=False)
            messagebox.showinfo("Saved", "Row Data saved successfully.")

# ==========================================
# --- 3. GUI Application Class ---
# ==========================================

class PhysicsOptimizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("QCM Analysis & Optimization Tool")
        self.geometry("1100x950")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # --- Data Containers ---
        self.df = None          
        self.df_echem = None    
        self.cycle_indices = None # Store cycle indices
        
        self.filename_qcmd = "Results" 
        self.filename_echem = "EChem_Data"
        
        self.current_harmonics = [3, 5, 7, 9, 11] 
        self.harmonics_widgets = [] 
        self.entries = {}
        self.air_f_entries = {}
        self.air_d_entries = {}
        
        self.is_running = False
        self.stop_flag = False
        
        self.harmonics_frame = None

        self.combined_window = None 
        self.row_window = None 
        self.dynamic_window = None 

        self.setup_ui()
        self.create_inputs_frame()
        self.update_harmonic_fields()

    def setup_ui(self):
        # --- File Loading Area ---
        self.step1_frame = ctk.CTkFrame(self)
        self.step1_frame.pack(pady=10, padx=20, fill="x")
        self.step1_frame.columnconfigure(0, weight=1)
        self.step1_frame.columnconfigure(1, weight=1)
        
        load_btn_color = "#7C3AED" 
        load_btn_hover = "#6D28D9" 

        self.btn_load_qcmd = ctk.CTkButton(self.step1_frame, text="1. Load QCM-D Data (Excel)", 
                                           command=self.load_qcmd_file,
                                           fg_color=load_btn_color, hover_color=load_btn_hover)
        self.btn_load_qcmd.grid(row=0, column=0, padx=20, pady=(10, 5))
        self.lbl_file_qcmd = ctk.CTkLabel(self.step1_frame, text="No file selected", text_color="gray")
        self.lbl_file_qcmd.grid(row=1, column=0, padx=20, pady=(0, 10))

        self.btn_load_echem = ctk.CTkButton(self.step1_frame, text="2. Load Electro-chemical Data (Excel)", 
                                            command=self.load_echem_file, 
                                            fg_color=load_btn_color, hover_color=load_btn_hover)
        self.btn_load_echem.grid(row=0, column=1, padx=20, pady=(10, 5))
        self.lbl_file_echem = ctk.CTkLabel(self.step1_frame, text="No file selected", text_color="gray")
        self.lbl_file_echem.grid(row=1, column=1, padx=20, pady=(0, 10))

        # Parameters Scroll Area
        self.scroll_frame = ctk.CTkScrollableFrame(self, height=350)
        self.scroll_frame.pack(pady=10, padx=20, fill="x")

        # Action Area
        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.pack(pady=15)

        # GREEN BUTTON 
        self.btn_run = ctk.CTkButton(self.action_frame, text="Run Optimization", 
                                     command=lambda: self.run_full_process(dynamic_config=None), 
                                     fg_color="green", width=160, height=35, font=("Arial", 13, "bold"))
        self.btn_run.pack(side="left", padx=5)

        # RED BUTTON
        self.btn_visc_correct = ctk.CTkButton(self.action_frame, text="Define Dynamic Params", 
                                              command=self.open_dynamic_window,
                                              fg_color="#D32F2F", hover_color="#B71C1C",
                                              width=180, height=35, font=("Arial", 13, "bold"))
        self.btn_visc_correct.pack(side="left", padx=5)

        # Resolution
        self.res_sub_frame = ctk.CTkFrame(self.action_frame, fg_color="transparent")
        self.res_sub_frame.pack(side="left", padx=10)
        ctk.CTkLabel(self.res_sub_frame, text="Resolution:").pack(side="left", padx=5)
        self.res_var = ctk.StringVar(value="Normal")
        self.combo_res = ctk.CTkOptionMenu(self.res_sub_frame, 
                                           values=["Low (Fast)", "Normal", "High (Slow)"],
                                           variable=self.res_var, width=100)
        self.combo_res.pack(side="left")

        # Status
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.pack(pady=(0, 10), fill="x", padx=40)
        self.lbl_progress_text = ctk.CTkLabel(self.progress_frame, text="Waiting to start... (0/0)", font=("Arial", 12))
        self.lbl_progress_text.pack(pady=5)
        
        # Info
        self.info_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.info_frame.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(self.info_frame, text="Note: optimization uses calculated & calibrated EChem data.", 
                     font=("Arial", 12, "bold"), text_color="yellow").pack(pady=5)

        # Row Plotting
        self.vis_frame = ctk.CTkFrame(self)
        self.vis_frame.pack(pady=5, padx=20, fill="x")
        self.row_input_frame = ctk.CTkFrame(self.vis_frame, fg_color="transparent")
        self.row_input_frame.pack(pady=5)
        ctk.CTkLabel(self.row_input_frame, text="Row Index to Plot:").pack(side="left", padx=5)
        self.entry_row_idx = ctk.CTkEntry(self.row_input_frame, width=60)
        self.entry_row_idx.pack(side="left", padx=5)
        self.btn_plot_row = ctk.CTkButton(self.row_input_frame, text="Show Plot", 
                                          command=lambda: self.plot_specific_row(None), fg_color="#1f538d")
        self.btn_plot_row.pack(side="left", padx=10)

        # Logs
        self.txt_log = ctk.CTkTextbox(self, height=100)
        self.txt_log.pack(pady=10, padx=20, fill="both", expand=True)

    def create_inputs_frame(self):
        self.input_container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.input_container.pack(pady=10, padx=10, expand=True)

        ctk.CTkLabel(self.input_container, text="QCM Physical Parameters:", font=("Arial", 14, "bold"))\
            .grid(row=0, column=0, columnspan=2, pady=(5, 10))
        
        defaults = {
            "Liquid Density [g/cm^3]": "1.3228",
            "Quartz Density [g/cm^3]": "2.648",
            "Liquid Viscosity [Pa·s]": "0.0032",
            "Quartz Viscosity [Pa]": "2.947e10",
            "Theta [0-1]": "1.0"
        }
        
        idx = 1
        for text, val in defaults.items():
            lbl = ctk.CTkLabel(self.input_container, text=text + ":")
            lbl.grid(row=idx, column=0, padx=10, pady=5, sticky="e")
            entry = ctk.CTkEntry(self.input_container)
            entry.insert(0, val)
            entry.grid(row=idx, column=1, padx=10, pady=5, sticky="w")
            self.entries[text] = entry
            idx += 1

        ctk.CTkLabel(self.input_container, text="Electrochemical Parameters:", font=("Arial", 14, "bold"))\
            .grid(row=0, column=2, columnspan=2, pady=(5, 10), padx=(40, 0))
        
        theo_defaults = {
            "Molecular Weight [g/mol]": "65",
            "Number of Electrons": "2",
            "Active Area [cm^2]": "0.785",
            "Sensitivity [Hz·cm^2/ug]": "56.5" 
        }
        
        theo_idx = 1
        for text, val in theo_defaults.items():
            lbl = ctk.CTkLabel(self.input_container, text=text + ":")
            lbl.grid(row=theo_idx, column=2, padx=(40, 10), pady=5, sticky="e") 
            entry = ctk.CTkEntry(self.input_container)
            entry.insert(0, val)
            entry.grid(row=theo_idx, column=3, padx=10, pady=5, sticky="w")
            self.entries[text] = entry 
            theo_idx += 1

        last_row = max(idx, theo_idx)

        last_row += 1
        self.harmonics_frame = ctk.CTkFrame(self.input_container, fg_color="transparent")
        self.harmonics_frame.grid(row=last_row, column=0, columnspan=4, pady=20)
        
        h_row = 0
        ctk.CTkLabel(self.harmonics_frame, text="Harmonics Configuration:", font=("Arial", 14, "bold"), text_color="#ff9f1c").grid(row=h_row, column=0, columnspan=3, pady=(0,5))
        h_row += 1

        self.control_frame = ctk.CTkFrame(self.harmonics_frame, fg_color="transparent")
        self.control_frame.grid(row=h_row, column=0, columnspan=3, pady=5)
        
        ctk.CTkLabel(self.control_frame, text="Max Harmonic (N):").pack(side="left", padx=5)
        self.combo_max_n = ctk.CTkOptionMenu(self.control_frame, values=[str(n) for n in range(3, 18, 2)], width=80, command=self.update_harmonic_fields)
        self.combo_max_n.set("11") 
        self.combo_max_n.pack(side="left", padx=5)
        h_row += 1

        ctk.CTkLabel(self.harmonics_frame, text="Air Reference Data (Hz / ppm):", font=("Arial", 14, "bold"), text_color="#4ea8de").grid(row=h_row, column=0, columnspan=3, pady=(15,5))
        h_row += 1

        ctk.CTkLabel(self.harmonics_frame, text="F_Air (Hz)").grid(row=h_row, column=1, padx=5)
        ctk.CTkLabel(self.harmonics_frame, text="D_Air (ppm)").grid(row=h_row, column=2, padx=5)
        h_row += 1
        
        self.harmonics_container_idx = h_row

    def update_harmonic_fields(self, selection=None):
        for widget in self.harmonics_widgets: widget.destroy()
        self.harmonics_widgets = []
        self.air_f_entries = {}; self.air_d_entries = {}
        
        try: max_n = int(self.combo_max_n.get())
        except: max_n = 11
        
        self.current_harmonics = list(range(3, max_n + 2, 2)) 
        
        def_f = {3: 14831489.89, 5: 24713249.91, 7: 34593697.21, 9: 44475233.72, 11: 54356482.32}
        def_d = {3: 15.5665, 5: 11.4873, 7: 8.8589, 9: 7.1840, 11: 7.0608}
        
        current_row = self.harmonics_container_idx
        
        for n in self.current_harmonics:
            lbl = ctk.CTkLabel(self.harmonics_frame, text=f"Harmonic n={n}:")
            lbl.grid(row=current_row, column=0, padx=5, sticky="e")
            self.harmonics_widgets.append(lbl)
            
            ef = ctk.CTkEntry(self.harmonics_frame, width=130)
            if n in def_f: ef.insert(0, str(def_f[n]))
            ef.grid(row=current_row, column=1, padx=5, pady=2)
            self.harmonics_widgets.append(ef)
            self.air_f_entries[n] = ef
            
            ed = ctk.CTkEntry(self.harmonics_frame, width=130)
            if n in def_d: ed.insert(0, str(def_d[n]))
            ed.grid(row=current_row, column=2, padx=5, pady=2)
            self.harmonics_widgets.append(ed)
            self.air_d_entries[n] = ed
            
            current_row += 1

    def open_dynamic_window(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a QCM-D file first to establish time bounds.")
            return

        if self.dynamic_window is None or not self.dynamic_window.winfo_exists():
            self.dynamic_window = DynamicParamsWindow(self, run_callback=lambda config: self.run_full_process(dynamic_config=config))
            self.dynamic_window.lift()
        else:
            self.dynamic_window.focus()

    # ==========================================
    # --- 3. Core Optimization Engine ---
    # ==========================================
    
    def calculate_optimization_loop(self, p, calc_tol, calc_pop, dynamic_config=None):
        harmonics = self.current_harmonics
        working_data = self.working_df.copy() 
        total_rows = len(working_data)
        
        time_col = None
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        for col in working_data.columns:
            if any(t.lower() in col.lower() for t in possible_time_cols):
                time_col = col
                break
        
        if dynamic_config and time_col is None:
             self.log("Warning: Could not find Time column. Using default parameters for all rows.")
             dynamic_config = None

        if "Theo_Calibrated_Active" in working_data.columns:
            theo_col = "Theo_Calibrated_Active"
        else:
            messagebox.showerror("Error", "Calibrated Theoretical Data is missing. Check EChem file.")
            return None, None, None, None, None, None, None, None
        
        res_h, res_xi, res_cutoff, res_status = [], [], [], []
        res_curves = [] 
        res_exp_3rd = [] 
        
        res_visc = [] 
        res_theta = []

        visc_ranges = dynamic_config.get('visc', []) if dynamic_config else []
        theta_ranges = dynamic_config.get('theta', []) if dynamic_config else []

        for index, row in working_data.iterrows():
            if self.stop_flag:
                self.log(">>> Process Stopped by user.")
                return None, None, None, None, None, None, None, None
            
            if index % 5 == 0:
                self.lbl_progress_text.configure(text=f"Processing: {index+1} / {total_rows}")
                self.update()

            current_visc_liq = p["visc_liq"] 
            current_theta = p["theta"]

            if time_col:
                t_val = row[time_col]
                
                for r in visc_ranges:
                    if r['start'] <= t_val <= r['end']:
                        current_visc_liq = r['val']
                        break
                
                for r in theta_ranges:
                      if r['start'] <= t_val <= r['end']:
                        current_theta = r['val']
                        break
            
            res_visc.append(current_visc_liq)
            res_theta.append(current_theta)

            row_dn_vals = []
            for n in harmonics:
                f_air = p["air_f"][n]
                val_dn = 1e6 * np.sqrt(current_visc_liq / (np.pi * p["rho_liq"] * f_air))
                row_dn_vals.append(val_dn)
            
            row_dn_um = np.array(row_dn_vals)
            row_dn_meters = row_dn_um * 1e-6

            try:
                current_df_vals = []
                current_dw_vals = []
                valid_row = True
                
                for n in harmonics:
                    col_f = f"f{n}"
                    col_d = f"D{n}" if f"D{n}" in working_data.columns else f"d{n}"
                    if col_f not in working_data.columns or col_d not in working_data.columns:
                        valid_row = False; break

                    val_f = row[col_f]; val_d = row[col_d]
                    f_air = p["air_f"][n]; d_air = p["air_d"][n]

                    delta_f = val_f - f_air
                    denominator = (p["rho_liq"] / 1000.0) * ((f_air / n) ** 2)
                    num_f = (delta_f / n) - row[theo_col] 
                    calc_df = num_f / denominator 
                    
                    term_w_liq = (val_d * 1e-6) * (val_f / n)
                    term_w_air = (d_air * 1e-6) * (f_air / n)
                    num_w = term_w_liq - term_w_air
                    calc_dw = num_w / denominator 

                    current_df_vals.append(calc_df)
                    current_dw_vals.append(calc_dw)

                if not valid_row:
                    res_h.append(None); res_xi.append(None); res_cutoff.append(None); res_status.append("Missing Cols")
                    res_curves.append(None); res_exp_3rd.append(None)
                    continue

                target_df = np.array(current_df_vals) * 1e9
                target_dw = np.array(current_dw_vals) * 1e9

                if 3 in harmonics:
                    idx_3 = harmonics.index(3)
                    val_dn_3 = row_dn_um[idx_3] 
                    val_w_3 = target_dw[idx_3]
                    res_exp_3rd.append((val_dn_3, val_w_3))
                else:
                    res_exp_3rd.append(None)

                def objective_function(params_opt):
                    qsi_nm, h_nm = params_opt
                    qsi = qsi_nm * 1e-9; h = h_nm * 1e-9  
                    w_model, f_model = dalta_model(row_dn_meters, current_theta, qsi, h, p["rho_liq"], p["rho_quartz"], p["visc_quartz"])
                    if w_model is None: return 1e9 
                    
                    weights_list = [10 if k < 3 else 1.0 for k in range(len(harmonics))]
                    f_weights_arr = np.array(weights_list)
                    
                    error_w = np.sum((w_model - target_dw)**2)
                    error_f = np.sum(((f_model - target_df)**2) * f_weights_arr)
                    return (100 * error_w) + error_f

                bounds = [(0.1, 200.0), (0.1, 1500.0)]
                result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=10000, popsize=calc_pop, tol=calc_tol)
                
                cutoff_val = find_cutoff(row_dn_um, target_dw, current_theta, p["rho_liq"], p["rho_quartz"], p["visc_quartz"])

                if result.success:
                    best_qsi_nm, best_h_nm = result.x
                    status = "Optimized"
                    if cutoff_val is not None and best_h_nm < cutoff_val:
                        best_qsi_nm = 0.1 
                        status = "Cutoff Correction"
                    
                    res_h.append(best_h_nm); res_xi.append(best_qsi_nm); res_cutoff.append(cutoff_val); res_status.append(status)

                    dn_smooth_m = np.linspace(0.003, 0.3, 500) * 1e-6
                    
                    model_w_curve, model_f_curve = dalta_model(dn_smooth_m, current_theta, best_qsi_nm * 1e-9, best_h_nm * 1e-9, p["rho_liq"], p["rho_quartz"], p["visc_quartz"])

                    if model_w_curve is not None:
                        combined_data = np.column_stack((dn_smooth_m, model_w_curve))
                        res_curves.append(combined_data)
                    else:
                        res_curves.append(None)

                else:
                    res_h.append(None); res_xi.append(None); res_cutoff.append(cutoff_val); res_status.append("Failed")
                    res_curves.append(None)

            except Exception as e:
                res_h.append(None); res_xi.append(None); res_cutoff.append(None); res_status.append("Error")
                res_curves.append(None); res_exp_3rd.append(None)
        
        return res_h, res_xi, res_cutoff, res_status, res_curves, res_exp_3rd, res_visc, res_theta

    # ==========================================
    # --- 4. Business Logic (Buttons) ---
    # ==========================================

    def calibrate_theoretical_data(self, t_arr, f_theo_arr, f_qcm_avg_arr):
        try:
            if len(f_theo_arr) < 3: return f_theo_arr 
            
            dy = np.diff(f_theo_arr)
            slope_sign = np.sign(dy)
            change_indices = np.where(np.diff(slope_sign) != 0)[0] + 1
            pois = np.concatenate(([0], change_indices, [len(f_theo_arr)-1])).astype(int)
            pois = np.unique(pois)
            
            # --- NEW: SAVE CYCLE INDICES ---
            self.cycle_indices = pois
            # -------------------------------

            f_calibrated = np.copy(f_theo_arr)

            for i in range(0, len(pois) - 2, 2):
                idx_start = pois[i] 
                idx_mid = pois[i+1]
                idx_end = pois[i+2]

                qcm_val_end = f_qcm_avg_arr[idx_end]
                theo_val_end = f_theo_arr[idx_end]
                offset = qcm_val_end - theo_val_end
                
                f_calibrated[idx_mid : idx_end + 1] = f_theo_arr[idx_mid : idx_end + 1] + offset
                
                new_mid_val = f_calibrated[idx_mid]

                qcm_val_start = f_qcm_avg_arr[idx_start] 
                
                n_points = idx_mid - idx_start
                if n_points > 0:
                    linear_segment = np.linspace(qcm_val_start, new_mid_val, n_points + 1)
                    f_calibrated[idx_start : idx_mid + 1] = linear_segment

            self.log(">>> Calibration Logic Applied: Linear Fall + Shifted Rise.")
            return f_calibrated

        except Exception as e:
            self.log(f"Error in Calibration: {e}")
            return f_theo_arr

    def run_full_process(self, dynamic_config=None):
        if self.is_running:
            self.log(">>> Stopping current process to restart...")
            self.stop_flag = True
            self.after(200, lambda: self.run_full_process(dynamic_config))
            return

        if self.df is None:
            messagebox.showwarning("Warning", "Please load a QCM-D file first.")
            return

        if self.df_echem is None:
            messagebox.showerror("Error", "EChem Data is required for calculation. Please load it.")
            return

        p = self.get_params()
        if not p: return
        
        res_choice = self.res_var.get()
        if "High" in res_choice: calc_tol, calc_pop = 0.001, 40
        elif "Low" in res_choice: calc_tol, calc_pop = 0.1, 10
        else: calc_tol, calc_pop = 0.01, 20

        mode_str = "Dynamic Params" if dynamic_config else "Standard"
        self.log(f"Starting Process... ({res_choice} | {mode_str})")
        if dynamic_config:
            v_len = len(dynamic_config.get('visc', []))
            t_len = len(dynamic_config.get('theta', []))
            self.log(f" > Config: {v_len} Visc Ranges, {t_len} Theta Ranges")
        
        self.is_running = True
        self.stop_flag = False
        start_time = time.time()

        # ============================================
        # === STEP A: PREPARE CALIBRATED DATA ===
        # ============================================
        
        self.working_df = self.df.copy()
        
        self.log("Calculating Theoretical Curve from EChem Data (Forced)...")
        try:
            Mw = float(self.entries["Molecular Weight [g/mol]"].get())
            n_elec = float(self.entries["Number of Electrons"].get())
            Area = float(self.entries["Active Area [cm^2]"].get())
            Cm = float(self.entries["Sensitivity [Hz·cm^2/ug]"].get())
            F_const = 96485.332
            
            time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'time/s']
            q_cols = ['Q', 'Charge', 'C', '(Q-Qo)', '(Q-Qo)/mC', 'Q/mC']
            e_cols = ['Ewe', 'Potential', 'Voltage', '<Ewe/V>', 'Ewe/V']
            
            t_col_ec = next((c for c in self.df_echem.columns if any(x in c for x in time_cols)), None)
            q_col_ec = next((c for c in self.df_echem.columns if any(x in c for x in q_cols)), None)
            e_col_ec = next((c for c in self.df_echem.columns if any(x in c for x in e_cols)), None)
            
            qcm_time_col = next((c for c in self.working_df.columns if any(x in c for x in time_cols)), None)

            if t_col_ec and q_col_ec and qcm_time_col:
                t_ec = pd.to_numeric(self.df_echem[t_col_ec], errors='coerce').fillna(0).to_numpy()
                q_raw = pd.to_numeric(self.df_echem[q_col_ec], errors='coerce').fillna(0).to_numpy()
                Q_coulombs = q_raw * 1e-3 if ("mC" in q_col_ec or "mc" in q_col_ec) else q_raw
                delta_f_theo_raw_all = ((Cm * 1e6) * Mw * Q_coulombs) / (n_elec * F_const * Area)
                
                e_raw_all = None
                if e_col_ec:
                    e_raw_all = pd.to_numeric(self.df_echem[e_col_ec], errors='coerce').fillna(0).to_numpy()
                
                t_qcm = pd.to_numeric(self.working_df[qcm_time_col], errors='coerce').fillna(0).to_numpy()
                indices_to_keep = []
                for t_val in t_qcm:
                    idx = (np.abs(t_ec - t_val)).argmin()
                    indices_to_keep.append(idx)
                
                f_theo_synced = delta_f_theo_raw_all[indices_to_keep]
                
                if e_raw_all is not None:
                      e_synced = e_raw_all[indices_to_keep]
                      self.working_df["E_we_V"] = e_synced
                      self.log(">>> Synced Voltage (E) data found and added.")

                qcm_avg_curve = np.zeros_like(f_theo_synced)
                count = 0
                for n in self.current_harmonics:
                    col_f = f"f{n}"
                    if col_f in self.working_df.columns:
                        raw_f = self.working_df[col_f].to_numpy()
                        norm_f = (raw_f - raw_f[0]) / n
                        qcm_avg_curve += norm_f
                        count += 1
                if count > 0: qcm_avg_curve /= count

                f_theo_calibrated = self.calibrate_theoretical_data(t_qcm, f_theo_synced, qcm_avg_curve)

                self.working_df["Theo_Calibrated_Active"] = f_theo_calibrated
                self.working_df["F_Calibrated_View"] = f_theo_calibrated 
                
                self.log(">>> UPDATE: Now using CALIBRATED Delta F Theo for optimization!")
                
                print("\n" + "="*80)
                print(">>> DATA FOR EXCEL COMPARISON (Time vs Calibrated) <<<")
                print("Index\tTime[s]\tF_Theo_Original[Hz]\tF_Theo_Calibrated[Hz]")
                for i in range(len(t_qcm)):
                    t_val = t_qcm[i]
                    f_orig = f_theo_synced[i]
                    f_cal = f_theo_calibrated[i]
                    print(f"{i}\t{t_val:.4f}\t{f_orig:.6f}\t{f_cal:.6f}")
                print("="*80 + "\n")
                
            else:
                messagebox.showerror("Error", "Could not find Time/Charge columns in files.")
                self.is_running = False
                return

        except Exception as e:
            self.log(f"Error in Forced Calculation: {e}")
            messagebox.showerror("Error", f"Calculation failed: {e}")
            self.is_running = False
            return

        # ============================================

        res_h, res_xi, res_cutoff, res_status, res_curves, res_exp_3rd, res_visc, res_theta = self.calculate_optimization_loop(p, calc_tol, calc_pop, dynamic_config)
        
        self.is_running = False 

        if res_h is None: 
            self.log("Process Stopped or Failed.")
            return 

        self.working_df["Optimized_Height_nm"] = res_h
        self.working_df["Optimized_Qsi_nm"] = res_xi
        self.working_df["Cutoff_nm"] = res_cutoff
        self.working_df["Fit_Status"] = res_status
        self.working_df["Used_Viscosity_Pa_s"] = res_visc
        self.working_df["Used_Theta"] = res_theta
        
        duration = time.time() - start_time
        self.lbl_progress_text.configure(text="Finished!")
        self.log(f"Done! {len(self.working_df)} rows in {duration:.2f}s.")
        
        # --- PREPARE CYCLE TIMES FOR PLOTTING ---
        current_cycle_times = None
        if self.cycle_indices is not None and len(self.cycle_indices) > 0:
            try:
                # Reuse t_qcm from above or re-fetch
                possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'time/s']
                qcm_time_col = next((c for c in self.working_df.columns if any(x in c for x in possible_time_cols)), None)
                if qcm_time_col:
                    times = pd.to_numeric(self.working_df[qcm_time_col], errors='coerce').fillna(0).to_numpy()
                    # Filter for only "odd" intervals (every cycle of rise and fall) -> [::2]
                    filtered_indices = self.cycle_indices[::2]
                    current_cycle_times = times[filtered_indices]
            except Exception as e:
                print("Error extracting cycle times:", e)
        # ----------------------------------------
        
        # Pass indices to window for calculations
        self.open_combined_graph_window(cycle_times=current_cycle_times, cycle_indices=self.cycle_indices)

    def open_combined_graph_window(self, cycle_times=None, cycle_indices=None):
        if self.working_df is None: return
        
        if self.combined_window is None or not self.combined_window.winfo_exists():
            self.combined_window = CombinedGraphWindow(self.working_df, self.current_harmonics, self.filename_qcmd, self.plot_specific_row, cycle_times=cycle_times, cycle_indices=cycle_indices)
            self.combined_window.lift()
        else:
            self.combined_window.destroy()
            self.combined_window = CombinedGraphWindow(self.working_df, self.current_harmonics, self.filename_qcmd, self.plot_specific_row, cycle_times=cycle_times, cycle_indices=cycle_indices)

    # ==========================================
    # --- 5. Helper Functions ---
    # ==========================================

    def load_qcmd_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            try:
                self.df = pd.read_excel(file_path)
                self.filename_qcmd = os.path.splitext(os.path.basename(file_path))[0]
                self.lbl_file_qcmd.configure(text=os.path.basename(file_path), text_color="white")
                self.log(f"QCM-D File loaded: {os.path.basename(file_path)} with {len(self.df)} rows.")
            except Exception as e: messagebox.showerror("Error", str(e))

    def load_echem_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            try:
                self.df_echem = pd.read_excel(file_path)
                self.filename_echem = os.path.splitext(os.path.basename(file_path))[0]
                self.lbl_file_echem.configure(text=os.path.basename(file_path), text_color="white")
                self.log(f"Electro-chemical File loaded: {os.path.basename(file_path)} with {len(self.df_echem)} rows.")
            except Exception as e: messagebox.showerror("Error", str(e))

    def log(self, message):
        self.txt_log.insert("end", message + "\n")
        self.txt_log.see("end")
        self.update_idletasks()

    def get_params(self):
        try:
            air_f = {}; air_d = {}
            for n in self.current_harmonics:
                vf = self.air_f_entries[n].get(); vd = self.air_d_entries[n].get()
                if not vf or not vd: return None
                air_f[n] = float(vf); air_d[n] = float(vd)
            
            return {
                "rho_liq": float(self.entries["Liquid Density [g/cm^3]"].get()) * 1e3,
                "rho_quartz": float(self.entries["Quartz Density [g/cm^3]"].get()) * 1e3,
                "visc_liq": float(self.entries["Liquid Viscosity [Pa·s]"].get()),
                "visc_quartz": float(self.entries["Quartz Viscosity [Pa]"].get()),
                "theta": float(self.entries["Theta [0-1]"].get()),
                "air_f": air_f, "air_d": air_d
            }
        except ValueError: messagebox.showerror("Error", "Invalid numerical parameters"); return None

    def plot_specific_row(self, target_idx=None):
        if self.working_df is None: return
        idx = -1
        if target_idx is not None: idx = target_idx
        else:
            try: idx = int(self.entry_row_idx.get())
            except: return
        if idx < 0 or idx >= len(self.working_df): return
        p = self.get_params()
        if not p: return

        row_data = self.working_df.iloc[idx]
        best_h_nm = row_data.get("Optimized_Height_nm", 0)
        best_qsi_nm = row_data.get("Optimized_Qsi_nm", 0)
        
        row_visc = row_data.get("Used_Viscosity_Pa_s")
        if pd.notna(row_visc):
            p["visc_liq"] = float(row_visc)
        
        row_theta = row_data.get("Used_Theta")
        if pd.notna(row_theta):
            p["theta"] = float(row_theta)
        
        if pd.isna(best_h_nm) or pd.isna(best_qsi_nm) or best_h_nm is None:
            messagebox.showerror("Error", f"Row {idx}: Invalid optimization results (failed fit).")
            return
        
        time_val = "N/A"
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        for col in self.working_df.columns:
            if any(t.lower() in col.lower() for t in possible_time_cols):
                val = row_data[col]
                time_val = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                break

        harmonics = self.current_harmonics
        
        if "Theo_Calibrated_Active" in self.working_df.columns:
            theo_col = "Theo_Calibrated_Active"
        else:
            theo_col = [c for c in self.working_df.columns if "theo" in c.lower()][0]

        exp_df_vals = []; exp_dw_vals = []
        dn_calculated_vals = []
        for n in harmonics:
            f_air = p["air_f"][n]
            val_dn = 1e6 * np.sqrt(p["visc_liq"] / (np.pi * p["rho_liq"] * f_air))
            dn_calculated_vals.append(val_dn)
        dn_calculated_um = np.array(dn_calculated_vals)

        for i, n in enumerate(harmonics):
            val_f = row_data[f"f{n}"]; val_d = row_data.get(f"D{n}", row_data.get(f"d{n}"))
            f_air = p["air_f"][n]; d_air = p["air_d"][n]
            delta_f = val_f - f_air
            denominator = (p["rho_liq"] / 1000.0) * ((f_air / n) ** 2)
            num_f = (delta_f / n) - row_data[theo_col] 
            calc_df = num_f / denominator 
            term_w_liq = (val_d * 1e-6) * (val_f / n)
            term_w_air = (d_air * 1e-6) * (f_air / n)
            num_w = term_w_liq - term_w_air
            calc_dw = num_w / denominator
            exp_df_vals.append(calc_df * 1e9); exp_dw_vals.append(calc_dw * 1e9)

        dn_smooth_m = np.linspace(0.003, 0.3, 500) * 1e-6
        model_w_curve, model_f_curve = dalta_model(dn_smooth_m, p["theta"], best_qsi_nm * 1e-9, best_h_nm * 1e-9, p["rho_liq"], p["rho_quartz"], p["visc_quartz"])
        
        if model_f_curve is None or model_w_curve is None:
            messagebox.showerror("Error", f"Row {idx}: Physics model failed to calculate curve (values diverged).")
            return

        kana_w, kana_f = model_Kanazawa_line(dn_smooth_m, p["rho_quartz"], p["visc_quartz"])

        col_n = harmonics
        col_dn = list(dn_calculated_um)
        col_df = exp_df_vals
        col_dw = exp_dw_vals
        col_model_dn = list(dn_smooth_m * 1e6) 
        col_model_f = list(model_f_curve)
        col_model_w = list(model_w_curve)

        param_names = ['ρ_L [g/cm^3]', 'η_L [Pa·s]', 'ρ_q [g/cm^3]', 'μ_q [Pa]', 'h [nm]', 'ξ [nm]', 'θ [0–1]']
        param_values = [p["rho_liq"] / 1000.0, p["visc_liq"], p["rho_quartz"] / 1000.0, p["visc_quartz"], best_h_nm, best_qsi_nm, p["theta"]]
        
        max_len = max(len(col_n), len(param_names), len(col_model_dn))
        def pad_list(l, length): return l + [None] * (length - len(l))
        
        export_data_dict = {
            'n (Exp)': pad_list(col_n, max_len), 'δ_µm (Exp)': pad_list(col_dn, max_len),
            'Y_Δf_1e9 (Exp)': pad_list(col_df, max_len), 'Y_ΔW_1e9 (Exp)': pad_list(col_dw, max_len),
            ' | ': [None] * max_len,
            'Model_δ_µm': pad_list(col_model_dn, max_len), 'Model_Δf': pad_list(col_model_f, max_len), 'Model_ΔW': pad_list(col_model_w, max_len),
            ' || ': [None] * max_len,
            'Parameter': pad_list(param_names, max_len), 'Value': pad_list(param_values, max_len)
        }

        if self.row_window is None or not self.row_window.winfo_exists():
            self.row_window = RowGraphWindow(idx, time_val, best_h_nm, best_qsi_nm,
                                             dn_smooth_m, model_f_curve, model_w_curve,
                                             dn_calculated_um, exp_df_vals, exp_dw_vals,
                                             kana_f, kana_w,
                                             self.filename_qcmd, export_data_dict)
            self.row_window.lift()
        else:
            self.row_window.destroy()
            self.row_window = RowGraphWindow(idx, time_val, best_h_nm, best_qsi_nm,
                                             dn_smooth_m, model_f_curve, model_w_curve,
                                             dn_calculated_um, exp_df_vals, exp_dw_vals,
                                             kana_f, kana_w,
                                             self.filename_qcmd, export_data_dict)

    def on_closing(self):
        self.stop_flag = True
        self.destroy()
        self.quit()

if __name__ == "__main__":
    app = PhysicsOptimizerApp()
    app.mainloop()