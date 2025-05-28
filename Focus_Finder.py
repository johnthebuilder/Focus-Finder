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

def mirror_curve(points, axis='x'):
    """Mirror curve across axis to create full reflector profile"""
    if len(points) == 0:
        return points
    
    if axis == 'x':
        # Mirror across x-axis (flip y coordinates)
        mirrored = points.copy()
        mirrored[:, 1] = -mirrored[:, 1]
    else:  # axis == 'y'
        # Mirror across y-axis (flip x coordinates)
        mirrored = points.copy()
        mirrored[:, 0] = -mirrored[:, 0]
    
    # Combine original and mirrored, removing duplicate center point if exists
    if np.allclose(points[0], [points[0, 0], 0]) or np.allclose(points[-1], [points[-1, 0], 0]):
        # Remove the center point from mirrored to avoid duplication
        if axis == 'x':
            mirrored = mirrored[1:] if abs(mirrored[0, 1]) < 1e-6 else mirrored
        else:
            mirrored = mirrored[1:] if abs(mirrored[0, 0]) < 1e-6 else mirrored
    
    # Combine: original + mirrored (reversed order for smooth curve)
    full_curve = np.vstack([points, mirrored[::-1]])
    return full_curve

def reflect_ray(incident_dir, normal):
    """Calculate reflected ray direction using law of reflection"""
    # R = I - 2(I·N)N where I is incident, N is normal, R is reflected
    incident_dir = np.array(incident_dir)
    normal = np.array(normal)
    
    # Ensure normal is normalized
    normal = normal / np.linalg.norm(normal)
    
    dot_product = np.dot(incident_dir, normal)
    reflected = incident_dir - 2 * dot_product * normal
    
    return reflected / np.linalg.norm(reflected)

def get_surface_normal(points, index):
    """Calculate surface normal at a point on the curve"""
    if index == 0:
        # Use forward difference at start
        dx = points[1, 0] - points[0, 0]
        dy = points[1, 1] - points[0, 1]
    elif index == len(points) - 1:
        # Use backward difference at end
        dx = points[-1, 0] - points[-2, 0]
        dy = points[-1, 1] - points[-2, 1]
    else:
        # Use central difference in middle
        dx = points[index + 1, 0] - points[index - 1, 0]
        dy = points[index + 1, 1] - points[index - 1, 1]
    
    # Tangent vector
    tangent = np.array([dx, dy])
    tangent = tangent / np.linalg.norm(tangent)
    
    # Normal is perpendicular to tangent (rotate 90 degrees)
    normal = np.array([-tangent[1], tangent[0]])
    
    return normal

def trace_ray_path(start_point, direction, main_points, sub_points, max_length=1000):
    """Trace a ray through the dual reflector system"""
    ray_path = [start_point]
    current_pos = np.array(start_point)
    current_dir = np.array(direction)
    current_dir = current_dir / np.linalg.norm(current_dir)
    
    # Trace ray to main reflector
    main_hit, main_index = find_ray_intersection(current_pos, current_dir, main_points)
    
    if main_hit is not None:
        ray_path.append(main_hit)
        
        # Calculate reflection off main reflector
        main_normal = get_surface_normal(main_points, main_index)
        reflected_dir = reflect_ray(current_dir, main_normal)
        
        # Trace reflected ray to sub reflector
        sub_hit, sub_index = find_ray_intersection(main_hit, reflected_dir, sub_points)
        
        if sub_hit is not None:
            ray_path.append(sub_hit)
            
            # Calculate reflection off sub reflector
            sub_normal = get_surface_normal(sub_points, sub_index)
            final_dir = reflect_ray(reflected_dir, sub_normal)
            
            # Extend final ray
            final_point = sub_hit + final_dir * 50  # Extend 50 units
            ray_path.append(final_point)
    
    return ray_path

def find_ray_intersection(ray_start, ray_dir, curve_points):
    """Find intersection of ray with curve using line-segment intersection"""
    ray_start = np.array(ray_start)
    ray_dir = np.array(ray_dir)
    
    min_distance = float('inf')
    closest_point = None
    closest_index = None
    
    # Check intersection with each line segment of the curve
    for i in range(len(curve_points) - 1):
        p1 = curve_points[i]
        p2 = curve_points[i + 1]
        
        # Parametric line equations:
        # Ray: ray_start + t * ray_dir
        # Segment: p1 + s * (p2 - p1), where 0 <= s <= 1
        
        # Solve: ray_start + t * ray_dir = p1 + s * (p2 - p1)
        seg_dir = p2 - p1
        
        # Set up system of equations
        # ray_start[0] + t * ray_dir[0] = p1[0] + s * seg_dir[0]
        # ray_start[1] + t * ray_dir[1] = p1[1] + s * seg_dir[1]
        
        # Rearrange to matrix form: [ray_dir, -seg_dir] * [t, s]^T = p1 - ray_start
        A = np.array([[ray_dir[0], -seg_dir[0]], 
                      [ray_dir[1], -seg_dir[1]]])
        b = p1 - ray_start
        
        try:
            if abs(np.linalg.det(A)) > 1e-10:  # Non-parallel lines
                params = np.linalg.solve(A, b)
                t, s = params[0], params[1]
                
                # Check if intersection is valid (forward ray, within segment)
                if t > 1e-6 and 0 <= s <= 1:  # t > small epsilon to avoid self-intersection
                    intersection = ray_start + t * ray_dir
                    distance = t
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = intersection
                        closest_index = i
        except np.linalg.LinAlgError:
            continue
    
    return closest_point, closest_index

def create_reflector_plot(main_points, sub_points, main_type, sub_type, 
                         main_focus, sub_foci, show_rays=False, ray_data=None,
                         show_mirror=False):
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
            marker=dict(size=3)
        ))
        
        # Show mirrored version if requested
        if show_mirror:
            mirrored_main = mirror_curve(main_points, 'x')
            fig.add_trace(go.Scatter(
                x=mirrored_main[:, 0],
                y=mirrored_main[:, 1],
                mode='lines',
                name='Main (Full Profile)',
                line=dict(color='lightblue', width=2, dash='dash'),
                showlegend=False
            ))
    
    # Plot sub reflector
    if len(sub_points) > 0:
        fig.add_trace(go.Scatter(
            x=sub_points[:, 0],
            y=sub_points[:, 1],
            mode='lines+markers',
            name=f'Sub Reflector ({sub_type})',
            line=dict(color='red', width=3),
            marker=dict(size=3)
        ))
        
        # Show mirrored version if requested
        if show_mirror:
            mirrored_sub = mirror_curve(sub_points, 'x')
            fig.add_trace(go.Scatter(
                x=mirrored_sub[:, 0],
                y=mirrored_sub[:, 1],
                mode='lines',
                name='Sub (Full Profile)',
                line=dict(color='pink', width=2, dash='dash'),
                showlegend=False
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
    
    # Plot ray paths
    if ray_data is not None and len(ray_data) > 0:
        for i, ray_path in enumerate(ray_data):
            if len(ray_path) > 1:
                ray_x = [p[0] for p in ray_path]
                ray_y = [p[1] for p in ray_path]
                
                fig.add_trace(go.Scatter(
                    x=ray_x,
                    y=ray_y,
                    mode='lines+markers',
                    name=f'Ray {i+1}' if len(ray_data) <= 5 else None,
                    line=dict(color='lime', width=2),
                    marker=dict(size=4, color='lime'),
                    showlegend=(i < 5)  # Only show legend for first 5 rays
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
    
    # Reflector type selection and orientation
    st.sidebar.header("🔍 Analysis Options")
    
    # Global rotation control
    st.sidebar.subheader("🔄 System Orientation")
    global_rotation = st.sidebar.slider(
        "Rotate Entire System (°):",
        min_value=-180,
        max_value=180,
        value=90,  # Default to 90° clockwise
        step=5,
        help="Rotate the entire coordinate system (both reflectors together)"
    )
    
    main_type = st.sidebar.selectbox(
        "Main Reflector Type:",
        ["Parabola", "Hyperbola", "Ellipse"]
    )
    
    sub_type = st.sidebar.selectbox(
        "Sub Reflector Type:",
        ["Hyperbola", "Ellipse", "Parabola"]
    )
    
    # Apply global rotation to both reflectors
    if global_rotation != 0:
        main_points = rotate_points(main_points, global_rotation)
        sub_points = rotate_points(sub_points, global_rotation)
    
    # Mirroring option
    show_mirror = st.sidebar.checkbox(
        "Show Full Profile (Mirrored)",
        value=True,
        help="Show complete reflector by mirroring curve across x-axis"
    )
    
    # Ray tracing controls
    st.sidebar.subheader("🌟 Ray Tracing")
    show_rays = st.sidebar.checkbox("Enable Ray Tracing", value=False)
    
    if show_rays:
        num_rays = st.sidebar.slider(
            "Number of Rays:",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of parallel rays to trace"
        )
        
        ray_spacing = st.sidebar.slider(
            "Ray Spacing:",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            help="Spacing between parallel rays"
        )
        
        ray_angle = st.sidebar.slider(
            "Incident Angle (°):",
            min_value=-90.0,
            max_value=90.0,
            value=-90.0,  # Default to vertical downward rays
            step=5.0,
            help="Angle of incoming rays (-90° = from above, 0° = horizontal)"
        )
        
        ray_start_distance = st.sidebar.slider(
            "Ray Start Distance:",
            min_value=50.0,
            max_value=200.0,
            value=100.0,
            step=10.0,
            help="Distance to start rays from reflectors"
        )
    
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
        
        # Generate ray data if ray tracing is enabled
        ray_data = None
        if show_rays and len(main_points) > 0 and len(sub_points) > 0:
            ray_data = []
            
            # Calculate ray starting positions
            all_points = np.vstack([main_points, sub_points])
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            
            # Ray direction from angle
            ray_angle_rad = np.radians(ray_angle)
            ray_direction = [np.sin(ray_angle_rad), -np.cos(ray_angle_rad)]  # Fixed direction calculation
            
            # Starting positions for parallel rays
            if abs(ray_angle) > 45:  # Nearly vertical rays
                # Start from above/below
                center_x = (x_min + x_max) / 2
                start_y = y_max + ray_start_distance if ray_angle < 0 else y_min - ray_start_distance
                
                for i in range(num_rays):
                    offset = (i - num_rays // 2) * ray_spacing
                    start_point = [center_x + offset, start_y]
                    
                    ray_path = trace_ray_path(start_point, ray_direction, main_points, sub_points)
                    if len(ray_path) > 1:
                        ray_data.append(ray_path)
            else:  # Nearly horizontal rays
                # Start from left/right
                center_y = (y_min + y_max) / 2
                start_x = x_min - ray_start_distance if ray_angle >= 0 else x_max + ray_start_distance
                
                for i in range(num_rays):
                    offset = (i - num_rays // 2) * ray_spacing
                    start_point = [start_x, center_y + offset]
                    
                    ray_path = trace_ray_path(start_point, ray_direction, main_points, sub_points)
                    if len(ray_path) > 1:
                        ray_data.append(ray_path)
        
        # Create plot
        fig = create_reflector_plot(
            main_points, sub_points, main_type, sub_type,
            main_focus, sub_foci, show_rays, ray_data, show_mirror
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
        if global_rotation != 0:
            st.write(f"**System Rotation:** {global_rotation}°")
        
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
        
        # Ray tracing results
        if show_rays and ray_data:
            st.subheader("Ray Tracing")
            st.write(f"**Rays Traced:** {len(ray_data)}")
            st.write(f"**Incident Angle:** {ray_angle}°")
            
            # Analyze ray convergence
            final_points = []
            for ray_path in ray_data:
                if len(ray_path) >= 4:  # Start, main hit, sub hit, final
                    final_points.append(ray_path[-1])
            
            if len(final_points) > 1:
                final_points = np.array(final_points)
                convergence_x = np.std(final_points[:, 0])
                convergence_y = np.std(final_points[:, 1])
                st.info(f"**Convergence (σx):** {convergence_x:.3f}")
                st.info(f"**Convergence (σy):** {convergence_y:.3f}")
        
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
            st.write(f"Sub Y range: {sub_range_y:.2f}")("Sub Reflector")
        st.write(f"**Type:** {sub_type}")
        st.write(f"**Points:** {len(sub_points)}")
        if sub_rotation != 0:
            st.write(f"**Rotation:** {sub_rotation}°")
        
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
        
        # Ray tracing results
        if show_rays and ray_data:
            st.subheader("Ray Tracing")
            st.write(f"**Rays Traced:** {len(ray_data)}")
            st.write(f"**Incident Angle:** {ray_angle}°")
            
            # Analyze ray convergence
            final_points = []
            for ray_path in ray_data:
                if len(ray_path) >= 4:  # Start, main hit, sub hit, final
                    final_points.append(ray_path[-1])
            
            if len(final_points) > 1:
                final_points = np.array(final_points)
                convergence_x = np.std(final_points[:, 0])
                convergence_y = np.std(final_points[:, 1])
                st.info(f"**Convergence (σx):** {convergence_x:.3f}")
                st.info(f"**Convergence (σy):** {convergence_y:.3f}")
        
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
            st.write(f"Sub Y range: {sub_range_y:.2f}")("Sub Reflector")
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
