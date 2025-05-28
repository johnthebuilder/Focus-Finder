import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize_scalar, curve_fit
from scipy.spatial.distance import cdist
import io
import re

# Configure page
st.set_page_config(
    page_title="Dual Reflector Focus Finder",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def parse_dxf_points(dxf_content):
    """Extract points from DXF content - simplified parser for LWPOLYLINE and LINE entities"""
    points = []
    lines = dxf_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for LWPOLYLINE or LINE entities
        if line == 'LWPOLYLINE' or line == 'LINE':
            entity_start = i
            i += 1
            
            # Parse coordinates within this entity
            while i < len(lines) and not lines[i].strip().startswith('ENDSEC'):
                if lines[i].strip() == '10':  # X coordinate code
                    if i + 1 < len(lines):
                        try:
                            x = float(lines[i + 1].strip())
                            i += 2
                            # Look for corresponding Y coordinate (code 20)
                            if i < len(lines) and lines[i].strip() == '20':
                                if i + 1 < len(lines):
                                    y = float(lines[i + 1].strip())
                                    points.append([x, y])
                                    i += 2
                                else:
                                    i += 1
                            else:
                                continue
                        except ValueError:
                            i += 1
                else:
                    i += 1
                
                # Break if we hit another entity
                if i < len(lines) and lines[i].strip() in ['LWPOLYLINE', 'LINE', 'CIRCLE', 'ARC']:
                    break
        else:
            i += 1
    
    return np.array(points) if points else np.array([[0, 0]])

def fit_parabola(points):
    """Fit parabola y = ax² + bx + c to points"""
    if len(points) < 3:
        return None, None
    
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # Fit parabola
        def parabola(x, a, b, c):
            return a * x**2 + b * x + c
        
        popt, _ = curve_fit(parabola, x, y)
        a, b, c = popt
        
        # Calculate focus for parabola y = ax² + bx + c
        # Convert to vertex form and find focus
        h = -b / (2 * a)  # vertex x
        k = c - b**2 / (4 * a)  # vertex y
        
        # Focus is at (h, k + 1/(4a))
        focus_x = h
        focus_y = k + 1 / (4 * a)
        
        return (focus_x, focus_y), popt
    except:
        return None, None

def fit_hyperbola(points):
    """Fit hyperbola to points - simplified approach"""
    if len(points) < 4:
        return None, None
    
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # Try fitting (x-h)²/a² - (y-k)²/b² = 1
        # This is a simplified approach - for complex hyperbolas, more sophisticated fitting is needed
        
        # Estimate center
        h = np.mean(x)
        k = np.mean(y)
        
        # Rough estimates for a and b
        a = np.std(x)
        b = np.std(y)
        
        # Calculate foci (approximately)
        c = np.sqrt(a**2 + b**2)
        focus1 = (h - c, k)
        focus2 = (h + c, k)
        
        return (focus1, focus2), (h, k, a, b)
    except:
        return None, None

def fit_ellipse(points):
    """Fit ellipse to points"""
    if len(points) < 5:
        return None, None
    
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # Estimate center
        h = np.mean(x)
        k = np.mean(y)
        
        # Estimate semi-axes
        a = np.std(x)
        b = np.std(y)
        
        # Calculate foci
        if a > b:
            c = np.sqrt(a**2 - b**2)
            focus1 = (h - c, k)
            focus2 = (h + c, k)
        else:
            c = np.sqrt(b**2 - a**2)
            focus1 = (h, k - c)
            focus2 = (h, k + c)
        
        return (focus1, focus2), (h, k, a, b)
    except:
        return None, None

def create_reflector_plot(main_points, sub_points, main_type, sub_type, 
                         main_focus, sub_foci, show_rays=False):
    """Create interactive plot of dual reflector system"""
    
    fig = go.Figure()
    
    # Plot main reflector
    if len(main_points) > 0:
        fig.add_trace(go.Scatter(
            x=main_points[:, 0],
            y=main_points[:, 1],
            mode='lines+markers',
            name=f'Main Reflector ({main_type})',
            line=dict(color='blue', width=3),
            marker=dict(size=4)
        ))
    
    # Plot sub reflector
    if len(sub_points) > 0:
        fig.add_trace(go.Scatter(
            x=sub_points[:, 0],
            y=sub_points[:, 1],
            mode='lines+markers',
            name=f'Sub Reflector ({sub_type})',
            line=dict(color='red', width=3),
            marker=dict(size=4)
        ))
    
    # Plot main reflector focus
    if main_focus is not None:
        fig.add_trace(go.Scatter(
            x=[main_focus[0]],
            y=[main_focus[1]],
            mode='markers',
            name='Main Focus',
            marker=dict(size=12, color='yellow', symbol='star')
        ))
    
    # Plot sub reflector foci
    if sub_foci is not None:
        if isinstance(sub_foci[0], tuple):  # Two foci (hyperbola/ellipse)
            fig.add_trace(go.Scatter(
                x=[sub_foci[0][0], sub_foci[1][0]],
                y=[sub_foci[0][1], sub_foci[1][1]],
                mode='markers',
                name='Sub Foci',
                marker=dict(size=10, color='orange', symbol='diamond')
            ))
        else:  # Single focus (parabola)
            fig.add_trace(go.Scatter(
                x=[sub_foci[0]],
                y=[sub_foci[1]],
                mode='markers',
                name='Sub Focus',
                marker=dict(size=10, color='orange', symbol='diamond')
            ))
    
    # Add ray tracing if requested
    if show_rays and main_focus is not None and sub_foci is not None:
        # Simple ray tracing example - you can expand this
        for i in range(0, len(main_points), max(1, len(main_points)//10)):
            point = main_points[i]
            # Draw ray from focus to main reflector point
            fig.add_trace(go.Scatter(
                x=[main_focus[0], point[0]],
                y=[main_focus[1], point[1]],
                mode='lines',
                line=dict(color='lightblue', width=1, dash='dash'),
                showlegend=False
            ))
    
    fig.update_layout(
        title='Dual Reflector System Analysis',
        xaxis_title='X (mm)',
        yaxis_title='Y (mm)',
        template='plotly_dark',
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    
    return fig

def generate_theoretical_curve(curve_type, params, x_range, num_points=100):
    """Generate theoretical curve based on type and parameters"""
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    if curve_type == "Parabola":
        # y = ax² + bx + c
        a, b, c = params
        y = a * x**2 + b * x + c
    elif curve_type == "Hyperbola":
        # Simplified hyperbola generation
        h, k, a, b = params
        y = k + b * np.sqrt((x - h)**2 / a**2 - 1)
        # Handle both branches if needed
    elif curve_type == "Ellipse":
        # Simplified ellipse generation
        h, k, a, b = params
        y_pos = k + b * np.sqrt(1 - (x - h)**2 / a**2)
        y_neg = k - b * np.sqrt(1 - (x - h)**2 / a**2)
        return np.column_stack([np.concatenate([x, x[::-1]]), 
                               np.concatenate([y_pos, y_neg[::-1]])])
    
    return np.column_stack([x, y])

def main():
    # Header
    st.markdown('<h1 class="main-header">📡 Dual Reflector Focus Finder</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Purpose:</strong> Analyze dual reflector antenna systems (Cassegrain/Gregorian) to find focus points.
    Upload DXF files with 2D projected curves or input points manually.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Reflector Configuration")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Input Method:",
        ["DXF Upload", "Manual Points", "Theoretical Curves"]
    )
    
    # Initialize data containers
    main_points = np.array([[0, 0]])
    sub_points = np.array([[0, 0]])
    
    if input_method == "DXF Upload":
        st.sidebar.subheader("📁 File Upload")
        
        main_dxf = st.sidebar.file_uploader(
            "Main Reflector DXF:",
            type=['dxf'],
            help="Upload DXF file with main reflector curve"
        )
        
        sub_dxf = st.sidebar.file_uploader(
            "Sub Reflector DXF:",
            type=['dxf'],
            help="Upload DXF file with sub reflector curve"
        )
        
        if main_dxf is not None:
            dxf_content = main_dxf.getvalue().decode('utf-8', errors='ignore')
            main_points = parse_dxf_points(dxf_content)
            st.sidebar.success(f"Main: {len(main_points)} points loaded")
        
        if sub_dxf is not None:
            dxf_content = sub_dxf.getvalue().decode('utf-8', errors='ignore')
            sub_points = parse_dxf_points(dxf_content)
            st.sidebar.success(f"Sub: {len(sub_points)} points loaded")
    
    elif input_method == "Manual Points":
        st.sidebar.subheader("✏️ Manual Input")
        
        # Text areas for point input
        main_text = st.sidebar.text_area(
            "Main Reflector Points (x,y per line):",
            value="0,0\n10,2\n20,8\n30,18",
            help="Enter x,y coordinates, one pair per line"
        )
        
        sub_text = st.sidebar.text_area(
            "Sub Reflector Points (x,y per line):",
            value="5,15\n10,12\n15,15",
            help="Enter x,y coordinates, one pair per line"
        )
        
        # Parse manual input
        try:
            main_lines = [line.strip() for line in main_text.split('\n') if line.strip()]
            main_points = np.array([[float(x) for x in line.split(',')] for line in main_lines])
        except:
            st.sidebar.error("Invalid main reflector points format")
            main_points = np.array([[0, 0]])
        
        try:
            sub_lines = [line.strip() for line in sub_text.split('\n') if line.strip()]
            sub_points = np.array([[float(x) for x in line.split(',')] for line in sub_lines])
        except:
            st.sidebar.error("Invalid sub reflector points format")
            sub_points = np.array([[0, 0]])
    
    else:  # Theoretical Curves
        st.sidebar.subheader("📐 Theoretical Parameters")
        
        # Main reflector parameters
        st.sidebar.write("**Main Reflector (Parabola):**")
        main_focal = st.sidebar.number_input("Focal Length:", value=25.0, step=1.0)
        main_vertex_x = st.sidebar.number_input("Vertex X:", value=0.0, step=1.0)
        main_vertex_y = st.sidebar.number_input("Vertex Y:", value=0.0, step=1.0)
        main_width = st.sidebar.number_input("Width:", value=50.0, step=1.0)
        
        # Generate parabola points
        x_range = np.linspace(-main_width/2, main_width/2, 50)
        y_parabola = (x_range**2) / (4 * main_focal)
        main_points = np.column_stack([x_range + main_vertex_x, y_parabola + main_vertex_y])
        
        # Sub reflector parameters
        sub_type_theoretical = st.sidebar.selectbox(
            "Sub Reflector Type:",
            ["Hyperbola", "Ellipse"]
        )
        
        if sub_type_theoretical == "Hyperbola":
            st.sidebar.write("**Sub Reflector (Hyperbola):**")
            hyp_a = st.sidebar.number_input("Semi-major axis (a):", value=5.0, step=0.5)
            hyp_b = st.sidebar.number_input("Semi-minor axis (b):", value=3.0, step=0.5)
            hyp_h = st.sidebar.number_input("Center X:", value=0.0, step=1.0)
            hyp_k = st.sidebar.number_input("Center Y:", value=15.0, step=1.0)
            
            # Generate hyperbola points (upper branch)
            x_hyp = np.linspace(-8, 8, 50)
            y_hyp = hyp_k + hyp_b * np.sqrt((x_hyp - hyp_h)**2 / hyp_a**2 + 1)
            sub_points = np.column_stack([x_hyp, y_hyp])
        
        else:  # Ellipse
            st.sidebar.write("**Sub Reflector (Ellipse):**")
            ell_a = st.sidebar.number_input("Semi-major axis (a):", value=8.0, step=0.5)
            ell_b = st.sidebar.number_input("Semi-minor axis (b):", value=5.0, step=0.5)
            ell_h = st.sidebar.number_input("Center X:", value=0.0, step=1.0)
            ell_k = st.sidebar.number_input("Center Y:", value=15.0, step=1.0)
            
            # Generate ellipse points
            theta = np.linspace(0, 2*np.pi, 100)
            x_ell = ell_h + ell_a * np.cos(theta)
            y_ell = ell_k + ell_b * np.sin(theta)
            sub_points = np.column_stack([x_ell, y_ell])
    
    # Reflector type selection
    st.sidebar.header("🔍 Analysis Options")
    
    main_type = st.sidebar.selectbox(
        "Main Reflector Type:",
        ["Parabola", "Hyperbola", "Ellipse"]
    )
    
    sub_type = st.sidebar.selectbox(
        "Sub Reflector Type:",
        ["Hyperbola", "Ellipse", "Parabola"]
    )
    
    show_rays = st.sidebar.checkbox("Show Ray Tracing", value=False)
    show_fits = st.sidebar.checkbox("Show Curve Fits", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Reflector Analysis")
        
        # Analyze main reflector
        main_focus = None
        main_params = None
        
        if main_type == "Parabola":
            main_focus, main_params = fit_parabola(main_points)
        elif main_type == "Hyperbola":
            main_focus, main_params = fit_hyperbola(main_points)
        elif main_type == "Ellipse":
            main_focus, main_params = fit_ellipse(main_points)
        
        # Analyze sub reflector
        sub_foci = None
        sub_params = None
        
        if sub_type == "Parabola":
            sub_foci, sub_params = fit_parabola(sub_points)
        elif sub_type == "Hyperbola":
            sub_foci, sub_params = fit_hyperbola(sub_points)
        elif sub_type == "Ellipse":
            sub_foci, sub_params = fit_ellipse(sub_points)
        
        # Create plot
        fig = create_reflector_plot(
            main_points, sub_points, main_type, sub_type,
            main_focus, sub_foci, show_rays
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Results")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            if st.button("📊 Export Analysis Data"):
                # Create comprehensive analysis report
                analysis_data = {
                    'Main_Reflector_Type': [main_type],
                    'Sub_Reflector_Type': [sub_type],
                    'Main_Points_Count': [len(main_points)],
                    'Sub_Points_Count': [len(sub_points)]
                }
                
                if main_focus is not None:
                    if isinstance(main_focus, tuple) and len(main_focus) == 2:
                        analysis_data['Main_Focus_X'] = [main_focus[0]]
                        analysis_data['Main_Focus_Y'] = [main_focus[1]]
                    elif isinstance(main_focus[0], tuple):  # Two foci
                        analysis_data['Main_Focus1_X'] = [main_focus[0][0]]
                        analysis_data['Main_Focus1_Y'] = [main_focus[0][1]]
                        analysis_data['Main_Focus2_X'] = [main_focus[1][0]]
                        analysis_data['Main_Focus2_Y'] = [main_focus[1][1]]
                
                if sub_foci is not None:
                    if isinstance(sub_foci[0], tuple):  # Two foci
                        analysis_data['Sub_Focus1_X'] = [sub_foci[0][0]]
                        analysis_data['Sub_Focus1_Y'] = [sub_foci[0][1]]
                        analysis_data['Sub_Focus2_X'] = [sub_foci[1][0]]
                        analysis_data['Sub_Focus2_Y'] = [sub_foci[1][1]]
                    else:  # Single focus
                        analysis_data['Sub_Focus_X'] = [sub_foci[0]]
                        analysis_data['Sub_Focus_Y'] = [sub_foci[1]]
                
                df = pd.DataFrame(analysis_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download Analysis CSV",
                    data=csv,
                    file_name="reflector_analysis.csv",
                    mime="text/csv"
                )
        
        with col_exp2:
            if st.button("📐 Export Point Data"):
                # Export all point coordinates
                main_df = pd.DataFrame(main_points, columns=['X', 'Y'])
                main_df['Reflector'] = 'Main'
                
                sub_df = pd.DataFrame(sub_points, columns=['X', 'Y'])
                sub_df['Reflector'] = 'Sub'
                
                combined_df = pd.concat([main_df, sub_df], ignore_index=True)
                csv = combined_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Points CSV",
                    data=csv,
                    file_name="reflector_points.csv",
                    mime="text/csv"
                )
    
    with col2:
        st.header("📈 Results")
        
        # Display main reflector results
        st.subheader("Main Reflector")
        st.write(f"**Type:** {main_type}")
        st.write(f"**Points:** {len(main_points)}")
        
        if main_focus is not None:
            if isinstance(main_focus, tuple) and len(main_focus) == 2 and not isinstance(main_focus[0], tuple):
                st.success(f"**Focus:** ({main_focus[0]:.3f}, {main_focus[1]:.3f})")
            elif isinstance(main_focus[0], tuple):  # Two foci
                st.success(f"**Focus 1:** ({main_focus[0][0]:.3f}, {main_focus[0][1]:.3f})")
                st.success(f"**Focus 2:** ({main_focus[1][0]:.3f}, {main_focus[1][1]:.3f})")
        else:
            st.warning("Could not determine focus")
        
        if main_params is not None:
            st.write("**Parameters:**")
            if main_type == "Parabola":
                st.write(f"a = {main_params[0]:.6f}")
                st.write(f"b = {main_params[1]:.6f}")
                st.write(f"c = {main_params[2]:.6f}")
        
        # Display sub reflector results
        st.subheader("Sub Reflector")
        st.write(f"**Type:** {sub_type}")
        st.write(f"**Points:** {len(sub_points)}")
        
        if sub_foci is not None:
            if isinstance(sub_foci[0], tuple):  # Two foci
                st.success(f"**Focus 1:** ({sub_foci[0][0]:.3f}, {sub_foci[0][1]:.3f})")
                st.success(f"**Focus 2:** ({sub_foci[1][0]:.3f}, {sub_foci[1][1]:.3f})")
                
                # Calculate distance between foci
                dist = np.sqrt((sub_foci[1][0] - sub_foci[0][0])**2 + 
                              (sub_foci[1][1] - sub_foci[0][1])**2)
                st.info(f"**Focal Distance:** {dist:.3f}")
            else:  # Single focus
                st.success(f"**Focus:** ({sub_foci[0]:.3f}, {sub_foci[1]:.3f})")
        else:
            st.warning("Could not determine foci")
        
        # System analysis
        st.subheader("System Analysis")
        
        if main_focus is not None and sub_foci is not None:
            # Analyze focal relationships
            if isinstance(main_focus, tuple) and len(main_focus) == 2 and not isinstance(main_focus[0], tuple):
                main_f = main_focus
            else:
                main_f = None
            
            if isinstance(sub_foci[0], tuple):
                # Check if main focus coincides with either sub focus
                for i, sub_f in enumerate(sub_foci):
                    if main_f is not None:
                        dist = np.sqrt((main_f[0] - sub_f[0])**2 + (main_f[1] - sub_f[1])**2)
                        if dist < 1.0:  # Tolerance
                            st.success(f"✅ Main focus aligns with sub focus {i+1}")
                            st.write(f"Distance: {dist:.3f}")
                        else:
                            st.info(f"Distance to sub focus {i+1}: {dist:.3f}")
            
            # Determine system type
            if sub_type == "Hyperbola":
                st.write("**System Type:** Cassegrain")
            elif sub_type == "Ellipse":
                st.write("**System Type:** Gregorian")
        
        # Data quality metrics
        st.subheader("Data Quality")
        
        if len(main_points) > 0:
            main_range_x = np.ptp(main_points[:, 0])
            main_range_y = np.ptp(main_points[:, 1])
            st.write(f"Main X range: {main_range_x:.2f}")
            st.write(f"Main Y range: {main_range_y:.2f}")
        
        if len(sub_points) > 0:
            sub_range_x = np.ptp(sub_points[:, 0])
            sub_range_y = np.ptp(sub_points[:, 1])
            st.write(f"Sub X range: {sub_range_x:.2f}")
            st.write(f"Sub Y range: {sub_range_y:.2f}")

if __name__ == "__main__":
    main()