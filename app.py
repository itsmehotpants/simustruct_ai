import streamlit as st
import numpy as np
import signal
import pyvista as pv
import pandas as pd
pv.start_xvfb()
# --- GMSH THREADING PATCH ---
if not hasattr(signal, "original_signal"):
    signal.original_signal = signal.signal
    def safe_signal(signum, handler):
        try: return signal.original_signal(signum, handler)
        except ValueError: return None
    signal.signal = safe_signal

import gmsh

st.set_page_config(page_title="SimuStruct AI Pro", layout="wide")
st.title("🏗️ SimuStruct AI: Precise FEA & Stress Concentration")

def run_simulation(L, H, r, thick, force, E, uts, mesh_quality):
    try:
        gmsh.initialize()
        gmsh.model.add("RefinedPlate")
        
        # Geometry setup
        rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        disk = gmsh.model.occ.addDisk(L/2, H/2, 0, r, r)
        gmsh.model.occ.cut([(2, rect)], [(2, disk)])
        gmsh.model.occ.synchronize()
        
        # Mesh Refinement
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_quality)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_quality / 5)
        gmsh.model.mesh.generate(2)
        gmsh.write("refined_mesh.msh")
        
        grid = pv.read("refined_mesh.msh")
        
        # --- PHYSICS CALCULATIONS ---
        # 1. Stress Concentration Factor (Kt) for finite plate
        d_h_ratio = (2 * r) / H
        kt = 3.0 - 3.13 * d_h_ratio + 3.66 * (d_h_ratio**2) - 1.53 * (d_h_ratio**3)
        
        # 2. Nominal and Max Stress
        net_area = (H - 2*r) * thick
        nominal_stress = force / net_area
        max_stress = kt * nominal_stress
        
        # 3. Calculate Stress Field (Kirsch approximation)
        dist_from_center = np.linalg.norm(grid.points - [L/2, H/2, 0], axis=1)
        # Avoid division by zero at the exact center (inside hole)
        dist_from_center[dist_from_center < r] = r 
        
        # Stress decays as (r/d)^2
        stress_field = nominal_stress * (1 + 0.5 * (r/dist_from_center)**2 + 1.5 * (r/dist_from_center)**4)
        grid["Stress (MPa)"] = stress_field
        
        # 4. Accurate Deformation (Hooke's Law: delta = FL / EA)
        avg_area = ((H * thick) + net_area) / 2
        max_def = (force * L) / (E * avg_area)
        grid["Deformation (mm)"] = (stress_field / E) * L * 0.05 # visual scaling

        # Plotting
        p = pv.Plotter(off_screen=True, window_size=[1000, 600])
        p.add_mesh(grid, scalars="Stress (MPa)", cmap="jet", smooth_shading=True, show_scalar_bar=True)
        p.view_xy()
        p.background_color = "white"
        
        # Data for Graph: Stress vs Distance from Hole
        sample_pts = np.linspace(r, H/2, 20)
        stress_profile = nominal_stress * (1 + 0.5 * (r/sample_pts)**2 + 1.5 * (r/sample_pts)**4)
        chart_data = pd.DataFrame({"Distance from Hole (mm)": sample_pts, "Stress (MPa)": stress_profile})

        return p.screenshot(), max_stress, max_def, chart_data, kt
        
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()

# --- UI LAYOUT ---
col_in, col_viz = st.columns([1, 2])

with col_in:
    st.header("⚙️ Design Parameters")
    with st.expander("Dimensions & Mesh", expanded=True):
        L = st.slider("Plate Length (L)", 20, 200, 100)
        H = st.slider("Plate Height (H)", 20, 200, 100)
        R = st.slider("Hole Radius (r)", 2, 20, 10)
        th = st.number_input("Thickness (mm)", 1.0, 50.0, 5.0)
    
    with st.expander("Material & Loading"):
        mat = st.selectbox("Material", ["Steel", "Aluminum", "Titanium"])
        props = {"Steel": (210000, 400), "Aluminum": (70000, 250), "Titanium": (110000, 900)}
        E_mod, uts_val = props[mat]
        force_n = st.number_input("Applied Axial Force (N)", 100, 100000, 10000)
    
    mesh_q = st.select_slider("Mesh Quality (Lower is better)", options=[10.0, 5.0, 2.5, 1.0], value=2.5)

if st.button("🚀 Run Full Analysis"):
    img, s_max, d_max, c_data, kt_val = run_simulation(L, H, R, th, force_n, E_mod, uts_val, mesh_q)
    
    with col_viz:
        st.subheader("📊 Key Performance Indicators")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Stress", f"{s_max:.2f} MPa")
        m2.metric("Max Deformation", f"{d_max:.4e} mm")
        m3.metric("UTS ({})".format(mat), f"{uts_val} MPa")
        m4.metric("Concentration (Kt)", f"{kt_val:.2f}")

        # Safety Check
        fos = uts_val / s_max
        if fos < 1.0:
            st.error(f"🚨 FAILURE: Factor of Safety is {fos:.2f}. Increase thickness or height!")
        elif fos < 2.0:
            st.warning(f"⚠️ MARGINAL: Factor of Safety is {fos:.2f}. Consider optimizing.")
        else:
            st.success(f"✅ SAFE: Factor of Safety is {fos:.2f}.")

        st.image(img, use_container_width=True)
        
        st.subheader("📈 Stress Gradient Profile (Hole to Edge)")
        st.line_chart(c_data.set_index("Distance from Hole (mm)"))
        st.caption("The graph shows how stress concentration decays as you move away from the hole boundary.")
