import streamlit as st
import numpy as np
import signal
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import random
import tempfile

try:
    pv.start_xvfb()
except Exception:
    pass

# --- GMSH THREADING PATCH ---
if not hasattr(signal, "original_signal"):
    signal.original_signal = signal.signal
    def safe_signal(signum, handler):
        try: return signal.original_signal(signum, handler)
        except ValueError: return None
    signal.signal = safe_signal

import gmsh

st.set_page_config(page_title="FEM Visualizer", layout="wide")
st.title("FEM Visualizer: Multi-Hole Stress Analysis")

def generate_pdf_report(L, H, th, mat, E_mod, uts_val, force_n, s_max, d_max, fos, img_arr):
    pdf = FPDF()
    pdf.add_page()
    
    # Header Box with Background
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, txt="Multi-Hole FEM Analysis Report", ln=True, align='C', fill=True)
    pdf.ln(10)
    
    # Design Parameters Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Design Parameters", ln=True, border='B')
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.cell(100, 8, txt=f"Material: {mat} (E={E_mod} MPa, UTS={uts_val} MPa)", ln=False)
    pdf.cell(90, 8, txt=f"Applied Force: {force_n} N", ln=True)
    pdf.cell(100, 8, txt=f"Dimensions: {L} x {H} mm", ln=False)
    pdf.cell(90, 8, txt=f"Thickness: {th} mm", ln=True)
    pdf.ln(10)
    
    # Results Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Simulation Results", ln=True, border='B')
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    
    if fos < 1.0:
        pdf.set_text_color(200, 0, 0)
        status = "FAILURE"
    elif fos < 2.0:
        pdf.set_text_color(200, 150, 0)
        status = "MARGINAL"
    else:
        pdf.set_text_color(0, 150, 0)
        status = "SAFE"
        
    pdf.cell(100, 8, txt=f"Global Max Stress: {s_max:.2f} MPa", ln=False)
    pdf.cell(90, 8, txt=f"Structural Status: {status}", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(100, 8, txt=f"Max Deformation: {d_max:.4e} mm", ln=False)
    pdf.cell(90, 8, txt=f"Factor of Safety (FoS): {fos:.2f}", ln=True)
    pdf.ln(10)
    
    # Visualization
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Stress Distribution Visualization", ln=True, border='B')
    pdf.ln(5)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plt.imsave(tmpfile.name, img_arr)
        pdf.image(tmpfile.name, x=15, w=180)
        
    return pdf.output(dest='S').encode('latin1')

def fast_simulation_scalar(L, H, holes_data, thick, force, E):
    max_hole_dia = max([2*hr for _, _, hr in holes_data]) if holes_data else 0
    net_area = (H - max_hole_dia) * thick
    nominal_stress = force / net_area if net_area > 0 else force / 1.0
    
    max_multiplier = 1.0
    if holes_data:
        max_multiplier = 3.0
        
    s_max = nominal_stress * max_multiplier
    
    avg_area = ((H * thick) + net_area) / 2
    d_max = (force * L) / (E * avg_area)
    return s_max, d_max


def run_simulation(L, H, holes_data, thick, force, E, uts, mesh_quality):
    try:
        gmsh.initialize()
        gmsh.model.add("MultiHolePlate")
        
        rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        
        disk_tags = []
        for hx, hy, hr in holes_data:
            disk = gmsh.model.occ.addDisk(hx, hy, 0, hr, hr)
            disk_tags.append((2, disk))
            
        gmsh.model.occ.cut([(2, rect)], disk_tags)
        gmsh.model.occ.synchronize()
        
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_quality)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_quality / 5)
        gmsh.model.mesh.generate(2)
        gmsh.write("refined_mesh.msh")
        
        grid = pv.read("refined_mesh.msh")
        
        max_hole_dia = max([2*hr for _, _, hr in holes_data]) if holes_data else 0
        net_area = (H - max_hole_dia) * thick
        nominal_stress = force / net_area if net_area > 0 else force / 1.0
        
        stress_field = np.ones(grid.n_points) * nominal_stress
        
        for hx, hy, hr in holes_data:
            dist = np.linalg.norm(grid.points - [hx, hy, 0], axis=1)
            dist[dist < hr] = hr
            
            local_stress = nominal_stress * (1 + 0.5 * (hr/dist)**2 + 1.5 * (hr/dist)**4)
            stress_field = np.maximum(stress_field, local_stress)
            
        grid["Stress (MPa)"] = stress_field
        max_stress = np.max(stress_field)
        
        avg_area = ((H * thick) + net_area) / 2
        max_def = (force * L) / (E * avg_area)
        grid["Deformation (mm)"] = (stress_field / E) * L * 0.05 
        
        p = pv.Plotter(off_screen=True, window_size=[1000, 600])
        p.add_mesh(grid, scalars="Stress (MPa)", cmap="jet", smooth_shading=True, show_scalar_bar=True)
        p.view_xy()
        p.background_color = "white"
        
        return p.screenshot(), max_stress, max_def
        
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


# --- UI LAYOUT ---
col_in, col_viz = st.columns([1, 2])

with col_in:
    st.header("Design Parameters")
    with st.expander("Plate Dimensions", expanded=True):
        L = st.slider("Plate Length (L)", 50, 300, 150)
        H = st.slider("Plate Height (H)", 50, 300, 100)
        th = st.number_input("Thickness (mm)", 1.0, 50.0, 5.0)
        
    with st.expander("Hole Configurations", expanded=True):
        num_holes = st.number_input("Number of Holes", 1, 5, 2)
        holes_list = []
        for i in range(int(num_holes)):
            st.markdown(f"**Hole {i+1}**")
            c1, c2, c3 = st.columns(3)
            hx = c1.number_input("X Pos", 0.0, float(L), float(L)/2 + (i*30) - 15, key=f"x{i}")
            hy = c2.number_input("Y Pos", 0.0, float(H), float(H)/2, key=f"y{i}")
            hr = c3.number_input("Radius", 1.0, 30.0, 10.0, key=f"r{i}")
            holes_list.append((hx, hy, hr))
    
    with st.expander("Material & Loading"):
        mat_opts = {
            "Steel": (210000, 400), 
            "Aluminum": (70000, 250), 
            "Titanium": (110000, 900),
            "Copper": (110000, 210),
            "Brass": (100000, 350),
            "Cast Iron": (100000, 200),
            "Polycarbonate": (2300, 65),
            "Carbon Fiber": (150000, 1200)
        }
        mat = st.selectbox("Material", list(mat_opts.keys()))
        E_mod, uts_val = mat_opts[mat]
        force_n = st.number_input("Applied Axial Force (N)", 100, 100000, 10000)
    
    mesh_q = st.select_slider("Mesh Quality", options=[10.0, 5.0, 2.5, 1.0], value=5.0)
    
    st.markdown("---")
    with st.expander("Dataset Generation"):
        st.write("Generate a synthetic dataset of varying plate/hole setups for ML training.")
        n_samples = st.number_input("Number of samples", 10, 5000, 100)
        if st.button("Generate Dataset (CSV)"):
            with st.spinner("Generating samples..."):
                samples = []
                for _ in range(int(n_samples)):
                    s_L = random.uniform(100, 300)
                    s_H = random.uniform(50, 200)
                    s_th = random.uniform(1, 20)
                    s_F = random.uniform(1000, 50000)
                    s_mat = random.choice(list(mat_opts.keys()))
                    s_E, s_uts = mat_opts[s_mat]
                    
                    s_hx, s_hy = s_L/2, s_H/2
                    s_hr = random.uniform(2, s_H/4.0)
                    
                    s_max, d_max = fast_simulation_scalar(s_L, s_H, [(s_hx, s_hy, s_hr)], s_th, s_F, s_E)
                    
                    samples.append({
                        "Length": s_L, "Height": s_H, "Thickness": s_th, "Force": s_F,
                        "Material": s_mat, "E_Modulus": s_E, "Hole_Radius": s_hr,
                        "Predict_MaxStress": s_max, "Predict_MaxDef": d_max
                    })
                
                df = pd.DataFrame(samples)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Dataset",
                    data=csv,
                    file_name="fem_training_dataset.csv",
                    mime="text/csv",
                )


if st.button("Run Multi-Hole Analysis"):
    with st.spinner("Computing..."):
        img, s_max, d_max = run_simulation(L, H, holes_list, th, force_n, E_mod, uts_val, mesh_q)
        
        with col_viz:
            st.subheader("Key Performance Indicators")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max Stress", f"{s_max:.2f} MPa")
            m2.metric("Max Def.", f"{d_max:.4e} mm")
            m3.metric("Max Strain", f"{(s_max/E_mod):.4e}")
            m4.metric(f"UTS ({mat})", f"{uts_val} MPa")

            fos = uts_val / s_max
            if fos < 1.0:
                st.error(f"FAILURE: Factor of Safety is {fos:.2f}.")
            elif fos < 2.0:
                st.warning(f"MARGINAL: Factor of Safety is {fos:.2f}.")
            else:
                st.success(f"SAFE: Factor of Safety is {fos:.2f}.")

            st.image(img, use_container_width=True)
            
            st.subheader("Stress Concentration Profile")
            fig, ax = plt.subplots(figsize=(8,3))
            
            if len(holes_list) > 0:
                hx, hy, hr = holes_list[0]
                distances = np.linspace(hr, np.max([hr*4, L/2]), 100)
                nom_s = force_n / ((H - 2*hr)*th) if (H - 2*hr) > 0 else 1.0
                stresses = nom_s * (1 + 0.5 * (hr/distances)**2 + 1.5 * (hr/distances)**4)
                
                ax.plot(distances, stresses, color='red', label="Stress (MPa)")
                ax.set_xlabel("Distance from Hole Center (mm)")
                ax.set_ylabel("Stress (MPa)")
                ax.set_title(f"Kirsch Stress Decay (Hole 1, Radius={hr}mm)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("No holes to plot stress concentration for.")

            pdf_bytes = generate_pdf_report(L, H, th, mat, E_mod, uts_val, force_n, s_max, d_max, fos, img)
            st.download_button(
                label="Download Analysis Report (PDF)",
                data=pdf_bytes,
                file_name="fem_analysis_report.pdf",
                mime="application/pdf"
            )
