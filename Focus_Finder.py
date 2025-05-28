import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize_scalar, curve_fit, least_squares
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
    """Extract points from DXF content - parser for LWPOLYLINE, LINE, and SPLINE entities"""
    points = []
    lines = dxf_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for SPLINE entities (most common from CAD exports)
        if line == 'SPLINE':
            i += 1
            # Parse SPLINE control points
            while i < len(lines):
                if lines[i].strip() == '10':  # X coordinate code for control points
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
                    # Break if we hit another entity or end of section
                    if i < len(lines) and lines[i].strip() in ['SPLINE', 'LWPOLYLINE', 'LINE', 'CIRCLE', 'ARC', 'ENDSEC']:
                        break
        
        # Look for LWPOLYLINE or LINE entities
        elif line == 'LWPOLYLINE' or line == 'LINE':
            i += 1
            # Parse coordinates within this entity
            while i < len(lines):
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
                    if i < len(lines) and lines[i].strip() in ['LWPOLYLINE', 'LINE', 'CIRCLE', 'ARC', 'SPLINE', 'ENDSEC']:
                        break
        else:
            i += 1
    
    return np.array(points) if points else np.array([[0, 0]])

def rotate_points(points, angle_deg):
    """Rotate points by angle_deg around origin"""
    if len(points) == 0:
        return points
    
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    return np.dot(points, rotation_matrix.T)

def fit_parabola(points):
    """Fit parabola to points - handles both orientations"""
    if len(points) < 3:
        return None, None
    
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # Determine if parabola opens vertically or horizontally
        x_range = np.ptp(x)
        y_range = np.ptp(y)
        
        if y_range > x_range:  # Vertical parabola: x = ay² + by + c
            # Fit x = ay² + by + c
            def parabola_vertical(y, a, b, c):
                return a * y**2 + b * y + c
            
            popt, _ = curve_fit(parabola_vertical, y, x)
            a, b, c = popt
            
            # Calculate focus for vertical parabola x = ay² + by + c
            k = -b / (2 * a)  # vertex y
            h = c - b**2 / (4 * a)  # vertex x
            
            # Focus is at (h + 1/(4a), k)
            focus_x = h + 1 / (4 * a)
            focus_y = k
            
        else:  # Horizontal parabola: y = ax² + bx + c
            def parabola_horizontal(x, a, b, c):
                return a * x**2 + b * x + c
            
            popt, _ = curve_fit(parabola_horizontal, x, y)
            a, b, c = popt
            
            # Calculate focus for horizontal parabola y = ax² + bx + c
            h = -b / (2 * a)  # vertex x
            k = c - b**2 / (4 * a)  # vertex y
            
            # Focus is at (h, k + 1/(4a))
            focus_x = h
            focus_y = k + 1 / (4 * a)
        
        return (focus_x, focus_y), popt
    except:
        return None, None

def fit_ellipse_robust(points):
    """Robust ellipse fitting using direct least squares method"""
    if len(points) < 5:
        return None, None
    
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # Center the data
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        # Use direct least squares fitting for ellipse
        # Fit: Ax² + Bxy + Cy² + Dx + Ey + F = 0 with constraint B²-4AC < 0
        
        # Design matrix
        D = np.column_stack([
            x_centered**2,      # A
            x_centered * y_centered,  # B
            y_centered**2,      # C
            x_centered,         # D
            y_centered,         # E
            np.ones(len(x))     # F
        ])
        
        # Constraint: 4AC - B² = 1 (ellipse constraint)
        # Use SVD to solve constrained problem
        U, s, Vt = np.linalg.svd(D)
        
        # Take the smallest singular vector as solution
        coeffs = Vt[-1, :]
        A, B, C, D_coeff, E, F = coeffs
        
        # Ensure it's an ellipse (discriminant test)
        discriminant = B**2 - 4*A*C
        if discriminant >= 0:
            # Try alternative fitting method
            return fit_ellipse_simple(points)
        
        # Convert to standard form
        # Calculate center
        denom = B**2 - 4*A*C
        if abs(denom) < 1e-12:
            return fit_ellipse_simple(points)
        
        h = (2*C*D_coeff - B*E) / denom + x_mean
        k = (2*A*E - B*D_coeff) / denom + y_mean
        
        # Calculate semi-axes and rotation
        theta = 0.5 * np.arctan2(B, A - C) if abs(A - C) > 1e-12 else 0
        
        # Transform to principal axes
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Calculate eigenvalues to get axes lengths
        M = np.array([[A, B/2], [B/2, C]])
        eigenvals = np.linalg.eigvals(M)
        
        if np.any(eigenvals <= 0):
            return fit_ellipse_simple(points)
        
        # Semi-axes lengths
        sqrt_eig = np.sqrt(eigenvals)
        a = max(sqrt_eig)  # semi-major axis
        b = min(sqrt_eig)  # semi-minor axis
        
        # Calculate focal distance
        if a > b:
            c_focal = np.sqrt(a**2 - b**2)
            # Foci along major axis
            focus1_x = h + c_focal * cos_theta
            focus1_y = k + c_focal * sin_theta
            focus2_x = h - c_focal * cos_theta
            focus2_y = k - c_focal * sin_theta
        else:
            focus1_x = focus2_x = h
            focus1_y = focus2_y = k
        
        focus1 = (focus1_x, focus1_y)
        focus2 = (focus2_x, focus2_y)
        
        return (focus1, focus2), (h, k, a, b, theta)
        
    except Exception as e:
        return fit_ellipse_simple(points)

def fit_ellipse_simple(points):
    """Simple ellipse fitting using bounding ellipse approximation"""
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # Estimate center as centroid
        h = np.mean(x)
        k = np.mean(y)
        
        # Estimate semi-axes from data spread
        a = np.std(x) * 2  # Rough estimate
        b = np.std(y) * 2
        
        # Ensure a >= b for proper focus calculation
        if b > a:
            a, b = b, a
            
        # Calculate foci
        if a > b and a > 0 and b > 0:
            c = np.sqrt(a**2 - b**2)
            focus1 = (h + c, k)
            focus2 = (h - c, k)
        else:
            focus1 = focus2 = (h, k)
        
        return (focus1, focus2), (h, k, a, b, 0)
        
    except:
        return None, None

def generate_ellipse_curve(center, a, b, angle=0, num_points=100):
    """Generate full ellipse curve from parameters"""
    h, k = center
    t = np.linspace(0, 2*np.pi, num_points)
    
    # Parametric ellipse equations
    x_ellipse = a * np.cos(t)
    y_ellipse = b * np.sin(t)
    
    # Apply rotation if needed
    if angle != 0:
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = x_ellipse * cos_angle - y_ellipse * sin_angle
        y_rot = x_ellipse * sin_angle + y_ellipse * cos_angle
        x_ellipse, y_ellipse = x_rot, y_rot
    
    # Translate to center
    x_ellipse += h
    y_ellipse += k
    
    return np.column_stack([x_ellipse, y_ellipse])

def fit_polynomial_curve(points, degree=3):
    """Fit polynomial curve to points - good for partial curves"""
    if len(points) < degree + 1:
        return None, None
    
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # Fit polynomial y = p(x)
        coeffs = np.polyfit(x, y, degree)
        poly_func = np.poly1d(coeffs)
        
        # For visualization, extend the curve slightly
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min
        x_extended = np.linspace(x_min - x_range*0.2, x_max + x_range*0.2, 100)
        y_extended = poly_func(x_extended)
        
        # Find approximate focus for polynomial (estimate)
        # Use the vertex of the fitted curve
        if degree >= 2:
            # For quadratic and higher, find critical points
            poly_deriv = np.polyder(poly_func)
            critical_points = np.roots(poly_deriv)
            
            # Find real critical points within reasonable range
            real_critical = []
            for cp in critical_points:
                if np.isreal(cp):
                    cp_real = np.real(cp)
                    if x_min - x_range <= cp_real <= x_max + x_range:
                        real_critical.append(cp_real)
            
            if real_critical:
                # Use the critical point closest to data center
                x_center = np.mean(x)
                best_cp = min(real_critical, key=lambda cp: abs(cp - x_center))
                focus_x = best_cp
                focus_y = poly_func(best_cp)
                
                # Estimate focal distance based on curvature
                second_deriv = np.polyder(poly_deriv)
                curvature = abs(second_deriv(best_cp))
                if curvature > 1e-6:
                    focal_offset = 1 / (4 * curvature)  # Approximate
                    focus_y += focal_offset
                
                return (focus_x, focus_y), (coeffs, x_extended, y_extended)
        
        # Fallback: use center of data
        focus_x = np.mean(x)
        focus_y = np.mean(y)
        return (focus_x, focus_y), (coeffs, x_extended, y_extended)
        
    except Exception as e:
        return None, None

def fit_ellipse_partial(points):
    """Fit ellipse to partial curve data with better constraints"""
    if len(points) < 5:
        return None, None
    
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # For partial curves, use a simpler geometric approach
        # Find approximate center from data bounds
        x_center = (np.min(x) + np.max(x)) / 2
        y_center = (np.min(y) + np.max(y)) / 2
        
        # Estimate semi-axes from data spread
        # For partial ellipse, this is approximate
        x_span = np.max(x) - np.min(x)
        y_span = np.max(y) - np.min(y)
        
        # Assume the partial curve represents a significant portion
        a = x_span * 1.5  # Semi-major axis estimate
        b = y_span * 2.0  # Semi-minor axis estimate
        
        # Ensure a >= b
        if b > a:
            a, b = b, a
        
        # Calculate foci
        if a > b and a > 0 and b > 0:
            c = np.sqrt(a**2 - b**2)
            focus1 = (x_center + c, y_center)
            focus2 = (x_center - c, y_center)
        else:
            focus1 = focus2 = (x_center, y_center)
        
        return (focus1, focus2), (x_center, y_center, a, b, 0)
        
    except:
        return None, None

def create_reflector_plot(main_points, sub_points, main_type, sub_type, 
                         main_focus, sub_foci, show_rays=False, ray_data=None,
                         sub_params=None, fitted_curve=None):
    """Create interactive plot of dual reflector system"""
    
    fig = go.Figure()
    
    # Plot main reflector - original curve only
    if len(main_points) > 0:
        fig.add_trace(go.Scatter(
            x=main_points[:, 0],
            y=main_points[:, 1],
            mode='lines+markers',
            name=f'Main Reflector ({main_type})',
            line=dict(color='blue', width=3),
            marker=dict(size=3)
        ))
    
    # Plot sub reflector - original curve only
    if len(sub_points) > 0:
        fig.add_trace(go.Scatter(
            x=sub_points[:, 0],
            y=sub_points[:, 1],
            mode='lines+markers',
            name=f'Sub Reflector ({sub_type}) - Data',
            line=dict(color='red', width=4),
            marker=dict(size=6, color='red')
        ))
    
    # Plot fitted curve based on type
    if fitted_curve is not None:
        if sub_type == "Polynomial":
            coeffs, x_extended, y_extended = fitted_curve
            fig.add_trace(go.Scatter(
                x=x_extended,
                y=y_extended,
                mode='lines',
                name=f'Fitted {sub_type}',
                line=dict(color='cyan', width=2, dash='dash'),
                opacity=0.8
            ))
        elif sub_type == "Ellipse" and sub_params is not None:
            if len(sub_params) == 5:  # Full parameters with rotation
                h, k, a, b, theta = sub_params
            else:  # Simple parameters
                h, k, a, b = sub_params
                theta = 0
                
            ellipse_points = generate_ellipse_curve((h, k), a, b, theta)
            
            fig.add_trace(go.Scatter(
                x=ellipse_points[:, 0],
                y=ellipse_points[:, 1],
                mode='lines',
                name='Fitted Ellipse (Full)',
                line=dict(color='pink', width=2, dash='dash'),
                opacity=0.7
            ))
    
    # Plot main reflector focus
    if main_focus is not None:
        fig.add_trace(go.Scatter(
            x=[main_focus[0]],
            y=[main_focus[1]],
            mode='markers',
            name='Main Focus',
            marker=dict(size=14, color='yellow', symbol='star')
        ))
    
    # Plot sub reflector foci
    if sub_foci is not None:
        if isinstance(sub_foci[0], tuple):  # Two foci (hyperbola/ellipse)
            fig.add_trace(go.Scatter(
                x=[sub_foci[0][0], sub_foci[1][0]],
                y=[sub_foci[0][1], sub_foci[1][1]],
                mode='markers',
                name='Sub Foci',
                marker=dict(size=12, color='orange', symbol='diamond')
            ))
            
            # Add lines connecting the foci
            fig.add_trace(go.Scatter(
                x=[sub_foci[0][0], sub_foci[1][0]],
                y=[sub_foci[0][1], sub_foci[1][1]],
                mode='lines',
                name='Focal Axis',
                line=dict(color='orange', width=1, dash='dot'),
                showlegend=False
            ))
        else:  # Single focus (parabola/polynomial)
            fig.add_trace(go.Scatter(
                x=[sub_foci[0]],
                y=[sub_foci[1]],
                mode='markers',
                name='Sub Focus',
                marker=dict(size=12, color='orange', symbol='diamond')
            ))
    
    # Add axis of symmetry
    if len(main_points) > 0 or len(sub_points) > 0:
        all_points = np.vstack([p for p in [main_points, sub_points] if len(p) > 0])
        y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        y_range = y_max - y_min
        
        fig.add_trace(go.Scatter(
            x=[0, 0],
            y=[y_min - y_range*0.1, y_max + y_range*0.1],
            mode='lines',
            name='Axis of Symmetry',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))
    
    # Determine title based on sub type
    curve_desc = "Polynomial Fit" if sub_type == "Polynomial" else "Ellipse Fit"
    
    fig.update_layout(
        title=f'Dual Reflector System Analysis - {curve_desc}',
        xaxis_title='X (mm)',
        yaxis_title='Y (mm)',
        template='plotly_dark',
        height=700,
        showlegend=True,
        hovermode='closest'
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    
    return fig

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
    
    # Reflector type selection and orientation
    st.sidebar.header("🔍 Analysis Options")
    
    # Global rotation control
    st.sidebar.subheader("🔄 System Orientation")
    global_rotation = st.sidebar.slider(
        "Rotate Entire System (°):",
        min_value=-180,
        max_value=180,
        value=-90,  # Default to -90° 
        step=5,
        help="Rotate the entire coordinate system (both reflectors together)"
    )
    
    main_type = st.sidebar.selectbox(
        "Main Reflector Type:",
        ["Parabola", "Hyperbola", "Ellipse"]
    )
    
    sub_type = st.sidebar.selectbox(
        "Sub Reflector Type:",
        ["Polynomial", "Ellipse", "Hyperbola", "Parabola"]  # Add polynomial option
    )
    
    # Apply global rotation to both reflectors
    if global_rotation != 0:
        main_points = rotate_points(main_points, global_rotation)
        sub_points = rotate_points(sub_points, global_rotation)
    
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
        
        # Analyze sub reflector
        sub_foci = None
        sub_params = None
        
        if sub_type == "Ellipse":
            sub_foci, sub_params = fit_ellipse_robust(sub_points)
        
        # Create plot
        fig = create_reflector_plot(
            main_points, sub_points, main_type, sub_type,
            main_focus, sub_foci, show_rays=False, ray_data=None, sub_params=sub_params
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("📈 Results")
        
        # Display main reflector results
        st.subheader("Main Reflector")
        st.write(f"**Type:** {main_type}")
        st.write(f"**Points:** {len(main_points)}")
        if global_rotation != 0:
            st.write(f"**System Rotation:** {global_rotation}°")
        
        if main_focus is not None:
            st.success(f"**Focus:** ({main_focus[0]:.3f}, {main_focus[1]:.3f})")
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
        
        if sub_params is not None and sub_type == "Ellipse":
            st.write("**Ellipse Parameters:**")
            if len(sub_params) == 5:
                h, k, a, b, theta = sub_params
                st.write(f"Center: ({h:.3f}, {k:.3f})")
                st.write(f"Semi-major axis (a): {a:.3f}")
                st.write(f"Semi-minor axis (b): {b:.3f}")
                st.write(f"Rotation: {np.degrees(theta):.1f}°")
                eccentricity = np.sqrt(1 - (b**2)/(a**2)) if a > 0 else 0
                st.write(f"Eccentricity: {eccentricity:.3f}")
            else:
                h, k, a, b = sub_params
                st.write(f"Center: ({h:.3f}, {k:.3f})")
                st.write(f"Semi-major axis (a): {a:.3f}")
                st.write(f"Semi-minor axis (b): {b:.3f}")
                eccentricity = np.sqrt(1 - (b**2)/(a**2)) if a > 0 else 0
                st.write(f"Eccentricity: {eccentricity:.3f}")
        
        # System analysis
        st.subheader("System Analysis")
        
        if main_focus is not None and sub_foci is not None:
            # Analyze focal relationships
            if isinstance(main_focus, tuple) and len(main_focus) == 2:
                main_f = main_focus
            else:
                main_f = None
            
            if isinstance(sub_foci[0], tuple):
                # Check if main focus coincides with either sub focus
                for i, sub_f in enumerate(sub_foci):
                    if main_f is not None:
                        dist = np.sqrt((main_f[0] - sub_f[0])**2 + (main_f[1] - sub_f[1])**2)
                        if dist < 5.0:  # Increased tolerance for real-world data
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
