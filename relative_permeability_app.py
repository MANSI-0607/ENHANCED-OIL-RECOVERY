import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table as RLTable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# ========================================================================
# PAGE CONFIG
# ========================================================================
st.set_page_config(
    page_title="Relative Permeability & Fractional Flow Calculator",
    layout="wide",
    page_icon="🌊"
)

st.title("🌊 Relative Permeability and Fractional Flow Calculator")
st.markdown("""
Compute **Krw, Kro, Fw**, and generate plots with a full **PDF report**.
""")

# ========================================================================
# SESSION STATE INIT
# ========================================================================
# Initialize session state keys if they don't exist
if "data" not in st.session_state:
    st.session_state.data = None
if "Swf" not in st.session_state:
    st.session_state.Swf = None
if "Fwf" not in st.session_state:
    st.session_state.Fwf = None
if "dFw" not in st.session_state:
    st.session_state.dFw = None

for figkey in ["fig1","fig2","fig3","fig4","fig5","fig6"]:
    if figkey not in st.session_state:
        st.session_state[figkey] = None

# ========================================================================
# SIDEBAR INPUTS
# ========================================================================
st.sidebar.header("Input Parameters")

st.sidebar.info("💡 Units: Use consistent units (typically cP or mPa·s)")
water_vis = st.sidebar.number_input("Water viscosity (μw) [cP]", min_value=1e-6, value=1.0, step=0.1, format="%.2f")
oil_vis = st.sidebar.number_input("Oil viscosity (μo) [cP]", min_value=1e-6, value=2.0, step=0.1, format="%.2f")
swi = st.sidebar.number_input("Irreducible water saturation (Swi)", min_value=0.0, max_value=1.0, value=0.2)
sor = st.sidebar.number_input("Residual oil saturation (Sor)", min_value=0.0, max_value=1.0, value=0.2)
end_krw = st.sidebar.number_input("End-point krw", min_value=0.0, max_value=10.0, value=1.0)
end_kro = st.sidebar.number_input("End-point kro", min_value=0.0, max_value=10.0, value=1.0)
model = st.sidebar.selectbox("Relative Permeability Model", ["Corey", "Wyllie & Gardner", "Pirson"])
N = st.sidebar.slider("Resolution (points)", 10, 400, 100)

EPS = 1e-12

# ========================================================================
# FUNCTION: GET MODEL FORMULAS
# ========================================================================
def get_model_formulas(model):
    """Return LaTeX formulas for the selected model"""
    
    base_formulas = r"""
    **Normalized Saturations:**
    
    $$S_{w,eff} = \frac{S_w - S_{wi}}{1 - S_{wi} - S_{or}}$$
    
    $$S_{o,eff} = \frac{S_o}{1 - S_{wi}} = \frac{1 - S_w}{1 - S_{wi}}$$
    """
    
    if model == "Corey":
        rel_perm = r"""
    **Corey Model:**
    
    $$k_{rw} = k_{rw}^{end} \cdot (S_{w,eff})^4$$
    
    $$k_{ro} = k_{ro}^{end} \cdot (S_{o,eff})^4$$
    """
    elif model == "Wyllie & Gardner":
        rel_perm = r"""
    **Wyllie & Gardner Model:**
    
    $$k_{rw} = k_{rw}^{end} \cdot (S_{w,eff})^3$$
    
    $$k_{ro} = k_{ro}^{end} \cdot (1 - S_{w,eff})^3$$
    """
    elif model == "Pirson":
        rel_perm = r"""
    **Pirson Model:**
    
    $$k_{rw} = k_{rw}^{end} \cdot (S_{w,eff})^{3.5}$$
    
    $$k_{ro} = k_{ro}^{end} \cdot (1 - S_{w,eff})^2$$
    """
    else:
        rel_perm = ""
    
    fractional_flow = r"""
    **Fractional Flow (Fw):**
    
    $$F_w = \frac{1}{1 + \frac{k_{ro} \cdot \mu_w}{k_{rw} \cdot \mu_o}}$$
    """
    
    # NOTE: Mobility-related formula block intentionally removed/commented out as requested.
    breakthrough = r"""
    **Breakthrough Point (Welge Tangent Method):**
    
    The breakthrough saturation $S_{wf}$ is found by drawing a tangent from the initial point $(S_{wi}, 0)$ to the $F_w$ curve.
    
    $$\text{Slope} = \frac{F_w - 0}{S_w - S_{wi}} = \frac{F_w}{S_w - S_{wi}}$$
    
    $$S_{wf} = \text{argmax}\left(\frac{F_w}{S_w - S_{wi}}\right)$$
    
    $$F_{wf} = F_w(S_{wf})$$
    """
    
    return base_formulas + rel_perm + fractional_flow + breakthrough

# ========================================================================
# FUNCTION: RELATIVE PERMEABILITY CALCULATION
# ========================================================================
def compute_relperm(model, water_vis, oil_vis, swi, sor, end_krw, end_kro, N):
    Sw = np.linspace(swi, 1.0 - sor, N)
    So = 1 - Sw
    denom = (1 - swi - sor)
    denom = denom if denom > EPS else EPS

    Sw_eff = (Sw - swi) / denom
    So_eff = So / (1 - swi)

    Krw = np.zeros_like(Sw)
    Kro = np.zeros_like(Sw)

    if model == "Corey":
        Krw = end_krw * np.clip(Sw_eff, 0, 1)**4
        Kro = end_kro * np.clip(So_eff, 0, 1)**4

    elif model == "Wyllie & Gardner":
        Krw = end_krw * np.clip(Sw_eff, 0, 1)**3
        Kro = end_kro * np.clip(1 - Sw_eff, 0, 1)**3

    elif model == "Pirson":
        Krw = end_krw * np.clip(Sw_eff, 0, 1)**3.5
        Kro = end_kro * np.clip(1 - ((Sw - swi)/denom), 0, 1)**2

    Krw_safe = np.where(Krw <= 0, EPS, Krw)
    Kro_safe = np.where(Kro <= 0, EPS, Kro)

    Fw = 1 / (1 + (Kro_safe * water_vis) / (Krw_safe * oil_vis))
    
    # -------------------------
    # Mobility calculation removed/commented out as requested
    # mobility = (Krw / water_vis) / (Kro_safe / oil_vis)
    # -------------------------

    # Welge tangent method: find tangent from (Swi, 0) to Fw curve
    # Slope from initial point: (Fw - 0) / (Sw - Swi)
    # Exclude points too close to Swi to avoid numerical instability
    min_distance = 0.01  # Minimum distance from Swi to consider
    
    denominators = Sw - swi
    # Create mask for valid points (far enough from Swi)
    valid_mask = denominators > min_distance
    
    # Initialize slopes array with -inf
    slopes = np.full_like(Sw, -np.inf)
    
    # Calculate slopes only for valid points
    slopes[valid_mask] = Fw[valid_mask] / denominators[valid_mask]
    
    # Find maximum slope (breakthrough point)
    # Only consider finite values
    finite_mask = np.isfinite(slopes)
    if np.any(finite_mask):
        idx = np.argmax(slopes)
        Swf = float(Sw[idx])
        Fwf = float(Fw[idx])
    else:
        # Fallback if no valid breakthrough point found
        idx = len(Sw) // 2
        Swf = float(Sw[idx])
        Fwf = float(Fw[idx])
    
    # Also compute derivative for plotting
    dFw = np.gradient(Fw, Sw)

    df = pd.DataFrame({
        "Sw": Sw,
        "Krw": Krw,
        "Kro": Kro,
        "Fw": Fw,
        # "Mobility": mobility  # mobility intentionally not included
    })

    return df, Swf, Fwf, dFw

# ========================================================================
# COMPUTE BUTTON
# ========================================================================
compute_pressed = st.sidebar.button("Compute")

if compute_pressed:
    # Clear all previous figures
    for figkey in ["fig1","fig2","fig3","fig4","fig5","fig6"]:
        if figkey in st.session_state:
            if st.session_state[figkey] is not None:
                plt.close(st.session_state[figkey])
            st.session_state[figkey] = None
    
    # Recalculate everything
    data, Swf, Fwf, dFw = compute_relperm(model, water_vis, oil_vis, swi, sor, end_krw, end_kro, N)

    # SAVE IN SESSION STATE
    st.session_state.data = data
    st.session_state.Swf = Swf
    st.session_state.Fwf = Fwf
    st.session_state.dFw = dFw
    st.session_state.model = model

    # --------------------------------------------------------------------
    # Generate PLOTS and Save in Session
    # --------------------------------------------------------------------

    # FIG 1 – Krw / Kro vs Sw (MEDIUM size)
    fig1, ax1 = plt.subplots()
    ax1.plot(data["Sw"], data["Krw"], label="Krw", linewidth=2)
    ax1.plot(data["Sw"], data["Kro"], label="Kro", linewidth=2)
    ax1.scatter([swi], [end_kro], color="red")
    ax1.scatter([1 - sor], [end_krw], color="green")
    ax1.set_xlabel("Sw")
    ax1.set_ylabel("Relative Permeability")
    ax1.legend()
    ax1.grid(True)
    st.session_state.fig1 = fig1

    # FIG 2 – Fw vs Sw with tangent line
    fig2, ax2 = plt.subplots()
    ax2.plot(data["Sw"], data["Fw"], linewidth=2, label='Fw curve')
    ax2.scatter([Swf], [Fwf], color="red", s=100, zorder=5, label=f'Breakthrough (Swf={Swf:.4f})')
    
    # Draw tangent line from (Swi, 0) through breakthrough point
    # Handle case where Swf might equal Swi
    if abs(Swf - swi) > EPS:
        tangent_slope = Fwf / (Swf - swi)
        tangent_x = np.array([swi, Swf])
        tangent_y = tangent_slope * (tangent_x - swi)
        ax2.plot(tangent_x, tangent_y, 'r--', linewidth=1.5, label='Tangent from Swi', alpha=0.7)
    
    ax2.scatter([swi], [0], color='orange', s=80, marker='s', zorder=5, label=f'Initial point (Swi={swi})')
    ax2.set_xlabel("Sw")
    ax2.set_ylabel("Fw")
    ax2.set_ylim(0, 1.0)  # Set y-axis limit to 1.0
    ax2.legend()
    ax2.grid(True)
    st.session_state.fig2 = fig2

    # FIG 3 – Crossplot: Krw vs Kro
    fig3, ax3 = plt.subplots()
    ax3.plot(data["Kro"], data["Krw"], linewidth=2)
    ax3.set_xlabel("Kro")
    ax3.set_ylabel("Krw")
    ax3.grid(True)
    st.session_state.fig3 = fig3

    # FIG 4 – Mobility ratio (COMMENTED OUT as requested)
    # fig4, ax4 = plt.subplots()
    # ax4.plot(data["Sw"], data["Mobility"], linewidth=2)
    # ax4.axhline(1.0, linestyle=":")
    # ax4.set_xlabel("Sw")
    # ax4.set_ylabel("Mobility")
    # ax4.grid(True)
    # st.session_state.fig4 = fig4

    # FIG 5 – dFw/dSw
    fig5, ax5 = plt.subplots()
    ax5.plot(data["Sw"], dFw, linewidth=2)
    ax5.set_xlabel("Sw")
    ax5.set_ylabel("dFw/dSw")
    ax5.grid(True)
    st.session_state.fig5 = fig5



    # FIG 6 – Combined plot: Fw and Kr curves with breakthrough 

    fig6, ax6 = plt.subplots(figsize=(4, 3)) 

    # Twin axis for Fw
    ax6_twin = ax6.twinx()

    # Relative permeabilities
    ax6.plot(data["Sw"], data["Krw"], label="Krw", linewidth=1.5, color='blue')
    ax6.plot(data["Sw"], data["Kro"], label="Kro", linewidth=1.5, color='orange')
    # Force same Y-limits for both axes so the zero lines align
    ax6.set_ylim(0, 1)
    ax6_twin.set_ylim(0, 1)

    ax6.set_xlabel("Sw", fontsize=7)
    ax6.set_ylabel("Relative Permeability", fontsize=6, color='black')
    ax6.tick_params(axis='both', labelsize=5, labelcolor='black')

    # Fw on right axis
    ax6_twin.plot(data["Sw"], data["Fw"], linewidth=2, color='green', label='Fw', alpha=0.8)

    ax6_twin.scatter([Swf], [Fwf], color="red", s=90, zorder=5, marker='*',
                     edgecolors='darkred', linewidths=1.5,
                     label=f'Breakthrough (Swf={Swf:.4f})')

    ax6_twin.set_ylabel("Fractional Flow (Fw)", fontsize=5, color='green')
    ax6_twin.tick_params(axis='y', labelsize=5, labelcolor='green')
    ax6_twin.set_ylim(0, 1.0)

    # Legends
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2,
               loc='center right', fontsize=5)

    ax6.grid(True, alpha=0.3)
    ax6.set_title("Relative Permeability & Fractional Flow\nwith Breakthrough Point",
                  fontsize=6, fontweight='bold')

    st.session_state.fig6 = fig6

    st.success("✔ Calculation completed!")


# ========================================================================
# SHOW RESULTS IF DATA EXISTS
# ========================================================================
if st.session_state.data is not None:

    data = st.session_state.data
    Swf = st.session_state.Swf
    Fwf = st.session_state.Fwf
    dFw = st.session_state.dFw
    model = st.session_state.get("model", "Corey")

    # ====================================================================
    # DISPLAY FORMULAS
    # ====================================================================
    st.subheader("📐Formulas Used")
    with st.expander("Show Formulas Used", expanded=True):
        formulas = get_model_formulas(model)
        st.markdown(formulas)

    st.subheader("📊 Computed Data")
    st.dataframe(data.round(6))

    st.markdown(f"""
### 🔵 Breakthrough Point  
**Swf = `{Swf:.6f}`**  
**Fwf = `{Fwf:.6f}`**  
    """)

    # --------------------------------------------------------------------
    # DISPLAY PLOTS
    # --------------------------------------------------------------------
    st.subheader("📈 Plots")
    
    # Show combined plot first
    st.pyplot(st.session_state.fig6,use_container_width=False)
    with st.expander("About this plot"):
        st.markdown("""
        This plot combines:
        - **Left Y-axis**: Relative permeabilities (Krw and Kro)
        - **Right Y-axis**: Fractional flow (Fw) 
        - **Red star**: Breakthrough point where the tangent from (Swi, 0) touches the Fw curve
        
        This visualization helps understand how relative permeabilities influence the fractional flow behavior.
        """)
    
    st.markdown("---")  
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(st.session_state.fig1)
        with st.expander("Formula for Relative Permeability"):
            if model == "Corey":
                st.latex(r"k_{rw} = k_{rw}^{end} \cdot (S_{w,eff})^4")
                st.latex(r"k_{ro} = k_{ro}^{end} \cdot (S_{o,eff})^4")
            elif model == "Wyllie & Gardner":
                st.latex(r"k_{rw} = k_{rw}^{end} \cdot (S_{w,eff})^3")
                st.latex(r"k_{ro} = k_{ro}^{end} \cdot (1 - S_{w,eff})^3")
            elif model == "Pirson":
                st.latex(r"k_{rw} = k_{rw}^{end} \cdot (S_{w,eff})^{3.5}")
                st.latex(r"k_{ro} = k_{ro}^{end} \cdot (1 - S_{w,eff})^2")
    
    with col2:
        st.pyplot(st.session_state.fig2)
        with st.expander("Formula for Fractional Flow"):
            st.latex(r"F_w = \frac{1}{1 + \frac{k_{ro} \cdot \mu_w}{k_{rw} \cdot \mu_o}}")

    st.subheader("Additional Plots")
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(st.session_state.fig3)
        st.caption("Crossplot: Relative permeability relationship")
    
    # Mobility-related UI and plots intentionally omitted/commented out as requested.

    with st.expander("Show dFw/dSw and Tangent Slope"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.pyplot(st.session_state.fig5)
            st.latex(r"\frac{dF_w}{dS_w}")
            st.markdown("Standard derivative of fractional flow")
        
        with col_b:
            # Plot tangent slopes
            fig_temp, ax_temp = plt.subplots()
            
            # Calculate slopes with minimum distance filter
            min_distance = 0.01
            denominators = data["Sw"] - swi
            valid_mask = denominators > min_distance
            
            slopes = np.full(len(data), np.nan)
            slopes[valid_mask] = data["Fw"][valid_mask] / denominators[valid_mask]
            
            ax_temp.plot(data["Sw"], slopes, linewidth=2, color='purple')
            
            # Only plot breakthrough point if valid
            if abs(Swf - swi) > min_distance:
                ax_temp.scatter([Swf], [Fwf / (Swf - swi)], color="red", s=100, zorder=5)
            
            ax_temp.set_xlabel("Sw")
            ax_temp.set_ylabel("Fw/(Sw-Swi)")
            ax_temp.set_title("Tangent Slope (Welge Method)")
            ax_temp.grid(True)
            st.pyplot(fig_temp)
            st.markdown("**Breakthrough occurs at:** " + r"$S_{wf} = \text{argmax}\left(\frac{F_w}{S_w - S_{wi}}\right)$")
        
        plt.close(fig_temp)

    # ====================================================================
    # CSV DOWNLOAD 
    # ====================================================================
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results (CSV)",
        data=csv,
        file_name="relative_permeability_results.csv",
        mime="text/csv",
        key="csv_download"
    )

    # ====================================================================
    # PDF EXPORT
    # ====================================================================
    def create_pdf_buffer(data_df, Swf, Fwf, fig_list, inputs):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        flow = []

        flow.append(Paragraph("<b>Relative Permeability Report</b>", styles["Title"]))
        flow.append(Spacer(1, 12))

        flow.append(Paragraph("<b>Inputs:</b>", styles["Heading2"]))
        for k, v in inputs.items():
            flow.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        flow.append(Spacer(1, 12))

        flow.append(Paragraph("<b>Breakthrough:</b>", styles["Heading2"]))
        flow.append(Paragraph(f"Swf = {Swf:.6f}", styles["Normal"]))
        flow.append(Paragraph(f"Fwf = {Fwf:.6f}", styles["Normal"]))
        flow.append(Spacer(1, 12))

        # Table (first 150 rows) - note: Mobility column removed/commented out
        flow.append(Paragraph("<b>Computed Table Sample:</b>", styles["Heading2"]))
        max_rows = min(len(data_df), 150)
        tbl = [["Sw", "Krw", "Kro", "Fw"]]
        for i in range(max_rows):
            r = data_df.iloc[i]
            tbl.append([
                f"{r.Sw:.6f}",
                f"{r.Krw:.4e}",
                f"{r.Kro:.4e}",
                f"{r.Fw:.4e}",
            ])
        flow.append(RLTable(tbl))
        flow.append(Spacer(1, 12))

        # Add figures
        for title, fig in fig_list:
            flow.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
            b = BytesIO()
            fig.savefig(b, format="png", dpi=300, bbox_inches="tight")
            b.seek(0)
            flow.append(Image(b, width=6*inch, height=4*inch))
            flow.append(Spacer(1, 16))

        doc.build(flow)
        buffer.seek(0)
        return buffer

    inputs = {
        "Water viscosity": water_vis,
        "Oil viscosity": oil_vis,
        "Swi": swi,
        "Sor": sor,
        "End Krw": end_krw,
        "End Kro": end_kro,
        "Model": model,
        "N": N
    }

    figs = [
        ("Krw & Kro vs Sw", st.session_state.fig1),
        ("Fw vs Sw", st.session_state.fig2),
        ("Krw vs Kro", st.session_state.fig3),
        # Mobility plot intentionally omitted
        ("Combined Kr & Fw with Breakthrough", st.session_state.fig6)
    ]

    pdf = create_pdf_buffer(data, Swf, Fwf, figs, inputs)

    st.download_button(
        "📄 Download PDF Report",
        data=pdf,
        file_name="Relative_Permeability_Report.pdf",
        mime="application/pdf",
        key="pdf_download"
    )

else:
    st.info("👉 Enter parameters and click **Compute**.")
