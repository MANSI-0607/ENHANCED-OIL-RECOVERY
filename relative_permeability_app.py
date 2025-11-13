# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table as RLTable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Relative Permeability & Fractional Flow Calculator",
    layout="wide",
    page_icon="🌊"
)

st.title("🌊 Relative Permeability and Fractional Flow Calculator")
st.markdown("""
Compute relative permeabilities (Krw, Kro), fractional flow (Fw), draw Krw–Kro crossplot and Mobility ratio curve,
and export a printable PDF report containing inputs, breakthrough values, table and plots.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Parameters")

water_vis = st.sidebar.number_input("Water viscosity (μw)", min_value=1e-6, value=1.0, step=0.1, format="%.4f")
oil_vis = st.sidebar.number_input("Oil viscosity (μo)", min_value=1e-6, value=2.0, step=0.1, format="%.4f")
swi = st.sidebar.number_input("Irreducible water saturation (Swi)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.4f")
sor = st.sidebar.number_input("Residual oil saturation (Sor)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.4f")
end_krw = st.sidebar.number_input("End-point krw", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.4f")
end_kro = st.sidebar.number_input("End-point kro", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.4f")
model = st.sidebar.selectbox("Select Relative Permeability Model", ["Corey", "Wyllie & Gardner", "Pirson"])
N = st.sidebar.slider("Number of points (resolution)", 10, 500, 100)

# small epsilon to avoid divide-by-zero
EPS = 1e-12

# --- COMPUTATION FUNCTION ---
def compute_relperm(model, water_vis, oil_vis, swi, sor, end_krw, end_kro, N):
    # Sw range between Swi and 1 - Sor inclusive
    Sw = np.linspace(swi, 1.0 - sor, N)
    So = 1.0 - Sw
    denom = (1.0 - swi - sor)
    denom = denom if denom > EPS else EPS
    Sw_eff = (Sw - swi) / denom
    So_eff = So / (1.0 - swi)  # used sometimes

    Krw = np.zeros_like(Sw)
    Kro = np.zeros_like(Sw)

    if model == "Corey":
        Krw = end_krw * np.clip(Sw_eff, 0, 1)**4
        Kro = end_kro * np.clip(So_eff, 0, 1)**4
    elif model == "Wyllie & Gardner":
        Krw = end_krw * np.clip(Sw_eff, 0, 1)**3
        Kro = end_kro * np.clip(1 - Sw_eff, 0, 1)**3
    elif model == "Pirson":
        # as used previously: Krw ~ Sw_eff^3.5, Kro ~ (1 - scaled)^2
        Krw = end_krw * np.clip(Sw_eff, 0, 1)**3.5
        Kro = end_kro * np.clip(1 - ((Sw - swi) / denom), 0, 1)**2

    # Avoid zeros in Krw (for Fw calculation) — use small floor
    Krw_safe = np.where(Krw <= 0, EPS, Krw)
    Fw = 1.0 / (1.0 + ((Kro * water_vis) / (Krw_safe * oil_vis)))

    # Mobility ratio as function of Sw: M(Sw) = (Krw/μw) / (Kro/μo)
    Kro_safe = np.where(Kro <= 0, EPS, Kro)
    mobility = (Krw / water_vis) / (Kro_safe / oil_vis)

    # derivative dFw/dSw, find max
    dFw = np.gradient(Fw, Sw)
    max_idx = int(np.nanargmax(dFw))
    Swf = float(Sw[max_idx])
    Fwf = float(Fw[max_idx])

    data = pd.DataFrame({
        "Sw": Sw,
        "Krw": Krw,
        "Kro": Kro,
        "Fw": Fw,
        "Mobility": mobility
    })

    return data, Swf, Fwf, dFw

# --- RUN COMPUTATIONS WHEN BUTTON CLICKED ---
if st.sidebar.button("Compute"):
    data, Swf, Fwf, dFw = compute_relperm(model, water_vis, oil_vis, swi, sor, end_krw, end_kro, N)

    st.subheader("📊 Computed Results (first & last rows)")
    st.dataframe(data.round(6))

    # Display breakthrough
    st.markdown(f"**Breakthrough point (maximum dFw/dSw):**  Swf = `{Swf:.6f}`,  Fwf = `{Fwf:.6f}`")

    # --- PLOTS ---
    st.subheader("📈 Plots")
    col1, col2 = st.columns(2)

    # Krw & Kro vs Sw
    with col1:
        st.markdown("**Krw and Kro vs Sw**")
        fig1, ax1 = plt.subplots()
        ax1.plot(data["Sw"], data["Krw"], label="Krw", linewidth=2)
        ax1.plot(data["Sw"], data["Kro"], label="Kro", linewidth=2)
        ax1.scatter([swi], [end_kro], color="red", label="Swi marker (end Kro)", zorder=5)
        ax1.scatter([1.0 - sor], [end_krw], color="green", label="1-Sor marker (end Krw)", zorder=5)
        ax1.set_xlabel("Water Saturation (Sw)")
        ax1.set_ylabel("Relative Permeability")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()
        st.pyplot(fig1)

    # Fw vs Sw + marker
    with col2:
        st.markdown("**Fractional Flow (Fw) vs Sw**")
        fig2, ax2 = plt.subplots()
        ax2.plot(data["Sw"], data["Fw"], label="Fw", linewidth=2)
        ax2.scatter([Swf], [Fwf], color="red", label=f"(Swf, Fwf) = ({Swf:.4f}, {Fwf:.4f})", zorder=5)
        ax2.set_xlabel("Water Saturation (Sw)")
        ax2.set_ylabel("Fractional Flow (Fw)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()
        st.pyplot(fig2)

    # Crossplot Krw vs Kro and Mobility plot in new row
    st.markdown("**Additional Plots**")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("Krw vs Kro (Crossplot)")
        fig3, ax3 = plt.subplots()
        ax3.plot(data["Kro"], data["Krw"], linewidth=2)
        ax3.set_xlabel("Kro")
        ax3.set_ylabel("Krw")
        ax3.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig3)

    with c2:
        st.markdown("Mobility Ratio M(Sw) = (Krw/μw) / (Kro/μo)")
        fig4, ax4 = plt.subplots()
        ax4.plot(data["Sw"], data["Mobility"], linewidth=2)
        ax4.set_xlabel("Sw")
        ax4.set_ylabel("Mobility Ratio M(Sw)")
        ax4.grid(True, linestyle="--", alpha=0.4)
        # annotate typical stable threshold M=1
        ax4.axhline(1.0, linestyle=":", linewidth=1)
        st.pyplot(fig4)

    # Show derivative plot optionally
    with st.expander("Show dFw/dSw (derivative)"):
        fig5, ax5 = plt.subplots()
        ax5.plot(data["Sw"], dFw, linewidth=2)
        ax5.set_xlabel("Sw")
        ax5.set_ylabel("dFw/dSw")
        ax5.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig5)

    # --- DOWNLOAD SECTION (CSV) ---
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results (CSV)",
        data=csv,
        file_name="relative_permeability_results.csv",
        mime="text/csv"
    )

    # --- PDF EXPORT ---
    def create_pdf_buffer(data_df, Swf, Fwf, fig_list, inputs):
        """
        data_df: pd.DataFrame
        Swf, Fwf: floats
        fig_list: list of matplotlib figures [fig1, fig2, fig3, fig4]
        inputs: dict
        returns: BytesIO buffer containing PDF
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        flow = []

        # Title
        flow.append(Paragraph("<b>Relative Permeability & Fractional Flow Report</b>", styles['Title']))
        flow.append(Spacer(1, 8))

        # Inputs
        flow.append(Paragraph("<b>Input Parameters</b>", styles['Heading2']))
        for k, v in inputs.items():
            flow.append(Paragraph(f"{k}: {v}", styles['Normal']))
        flow.append(Spacer(1, 6))

        # Breakthrough
        flow.append(Paragraph("<b>Breakthrough Point (max dFw/dSw)</b>", styles['Heading2']))
        flow.append(Paragraph(f"Swf = {Swf:.6f}", styles['Normal']))
        flow.append(Paragraph(f"Fwf = {Fwf:.6f}", styles['Normal']))
        flow.append(Spacer(1, 10))

        # Add table (first up to 200 rows to keep PDF reasonable)
        flow.append(Paragraph("<b>Sample of Computed Table (Sw, Krw, Kro, Fw, Mobility)</b>", styles['Heading2']))
        # Limit rows for PDF readability
        max_rows = min(len(data_df), 200)
        tbl_data = [["Sw", "Krw", "Kro", "Fw", "Mobility"]]
        for i in range(max_rows):
            row = data_df.iloc[i]
            tbl_data.append([f"{row.Sw:.6f}", f"{row.Krw:.6e}", f"{row.Kro:.6e}", f"{row.Fw:.6e}", f"{row.Mobility:.6e}"])
        rl_table = RLTable(tbl_data, colWidths=[1*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.4*inch])
        flow.append(rl_table)
        flow.append(Spacer(1, 12))

        # Save figures to buffers and include
        img_width = 6 * inch
        img_height = 4 * inch
        for title, fig in fig_list:
            flow.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))
            img_buf = BytesIO()
            fig.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
            img_buf.seek(0)
            flow.append(Image(img_buf, width=img_width, height=img_height))
            flow.append(Spacer(1, 12))

        doc.build(flow)
        buffer.seek(0)
        return buffer

    inputs = {
        "Water viscosity (μw)": water_vis,
        "Oil viscosity (μo)": oil_vis,
        "Swi": swi,
        "Sor": sor,
        "End-point krw": end_krw,
        "End-point kro": end_kro,
        "Model": model,
        "Resolution (N)": N
    }

    # List of figures with titles
    figs = [
        ("Krw & Kro vs Sw", fig1),
        ("Fractional Flow (Fw) vs Sw", fig2),
        ("Krw vs Kro (Crossplot)", fig3),
        ("Mobility Ratio M(Sw) vs Sw", fig4)
    ]

    pdf_buffer = create_pdf_buffer(data, Swf, Fwf, figs, inputs)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name="Relative_Permeability_Report.pdf",
        mime="application/pdf"
    )

    st.success("✅ Computation, visualization and report generation completed successfully!")

else:
    st.info("👉 Enter parameters in the sidebar and click *Compute* to start calculations.")
