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

def fit_parabola_advanced(points):
    """Advanced parabola fitting for polynomial-optimized curves"""
    if len(points) < 3:
        return None, None
    
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # For your main reflector, it's a dish opening upward (concave up)
        # The polynomial is: y = Am/100000×x³ + Bm/1000×x² + Cm×x/100 + Dm/10 - 30
        # For a dish, we expect the focus to be above the vertex
        
        # Find the vertex (lowest point for upward-opening dish)
        vertex_idx = np.argmin(y)
        vertex_x = x[vertex_idx]
        vertex_y = y[vertex_idx]
        
        # For polynomial curves, estimate the effective focal length
        # by fitting a parabola to points near the vertex
        
        # Use points within 20% of the total span around the vertex
        x_span = np.max(x) - np.min(x)
        mask = np.abs(x - vertex_x) <= x_span * 0.2
        
        if np.sum(mask) >= 3:
            x_local = x[mask]
            y_local = y[mask]
            
            # Translate to vertex coordinates
            x_rel = x_local - vertex_x
            y_rel = y_local - vertex_y
            
            # Fit parabola y = ax² near vertex
            try:
                # Use only the x² term for better focus estimation
                A = x_rel[:, np.newaxis]**2
                a_coeff = np.linalg.lstsq(A, y_rel, rcond=None)[0][0]
                
                if a_coeff > 0:  # Upward opening parabola
                    # For y = ax², focus is at (0, 1/(4a)) relative to vertex
                    focal_distance = 1 / (4 * a_coeff)
                    focus_x = vertex_x
                    focus_y = vertex_y + focal_distance
                    
                    return (focus_x, focus_y), (a_coeff, vertex_x, vertex_y)
            except:
                pass
        
        # Fallback: estimate focus from curvature
        # For a dish, focus should be above the vertex
        y_range = np.max(y) - np.min(y)
        estimated_focal_length = x_span / 4  # Rough estimate
        focus_x = vertex_x
        focus_y = vertex_y + estimated_focal_length
        
        return (focus_x, focus_y), None
        
    except Exception as e:
        return None, None

def trace_ray_path_improved(start_point, direction, main_points, sub_points):
    """Improved ray tracing with better visualization"""
    ray_path = [np.array(start_point)]
    current_pos = np.array(start_point)
    current_dir = np.array(direction)
    current_dir = current_dir / np.linalg.norm(current_dir)
    
    # Trace ray to main reflector first
    main_hit, main_index = find_ray_intersection(current_pos, current_dir, main_points)
    
    if main_hit is not None and main_index is not None:
        ray_path.append(main_hit)
        
        # Calculate reflection off main reflector
        main_normal = get_surface_normal(main_points, main_index)
        
        # Ensure normal points outward (for a dish, upward and outward)
        # Check if normal points toward incident ray, if so flip it
        if np.dot(main_normal, -current_dir) < 0:
            main_normal = -main_normal
        
        reflected_dir = reflect_ray(current_dir, main_normal)
        
        # Trace reflected ray to sub reflector
        sub_hit, sub_index = find_ray_intersection(main_hit, reflected_dir, sub_points)
        
        if sub_hit is not None and sub_index is not None:
            ray_path.append(sub_hit)
            
            # Calculate reflection off sub reflector
            sub_normal = get_surface_normal(sub_points, sub_index)
            
            # For sub-reflector, normal should point away from main dish
            # Check orientation and flip if needed
            to_main = main_hit - sub_hit
            if np.dot(sub_normal, to_main) < 0:
                sub_normal = -sub_normal
            
            final_dir = reflect_ray(reflected_dir, sub_normal)
            
            # Extend final ray - this should converge to the system focus
            final_point = sub_hit + final_dir * 15  # Extend 15 units
            ray_path.append(final_point)
        else:
            # If no sub hit, extend the reflected ray
            extended_point = main_hit + reflected_dir * 20
            ray_path.append(extended_point)
    
    return ray_path

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

def fit_ellipse_through_points(points):
    """Fit ellipse that passes through the given points (partial curve on full ellipse)"""
    if len(points) < 5:
        return None, None
    
    try:
        # Use 5 well-distributed points for better fitting
        n_points = len(points)
        if n_points > 5:
            # Select 5 evenly distributed points
            indices = np.linspace(0, n_points-1, 5, dtype=int)
            selected_points = points[indices]
        else:
            selected_points = points
        
        x = selected_points[:, 0]
        y = selected_points[:, 1]
        
        # Algebraic ellipse fitting: Ax² + Bxy + Cy² + Dx + Ey + F = 0
        # We want the ellipse that these points lie on
        
        # Build the constraint matrix - each point gives us one equation
        # For n points, we have n equations in 6 unknowns (A,B,C,D,E,F)
        # We set F = -1 to avoid trivial solution and solve for the rest
        
        D_matrix = np.column_stack([
            x**2,           # A
            x * y,          # B  
            y**2,           # C
            x,              # D
            y,              # E
        ])
        
        # Right hand side (since F = -1)
        b = -np.ones(len(x))
        
        # Solve the overdetermined system using least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(D_matrix, b, rcond=None)
        A, B, C, D, E = coeffs
        F = -1
        
        # Check if this represents an ellipse (discriminant < 0)
        discriminant = B**2 - 4*A*C
        if discriminant >= 0:
            st.warning("Fitted curve is not an ellipse (discriminant >= 0)")
            # Try alternative method
            return fit_ellipse_force(points)
        
        # Convert to center and axis form
        # Calculate center
        denom = B**2 - 4*A*C
        if abs(denom) < 1e-12:
            return fit_ellipse_geometric(points)
        
        h = (2*C*D - B*E) / denom  # center x
        k = (2*A*E - B*D) / denom  # center y
        
        # Calculate the semi-axes and rotation angle
        # This involves solving the eigenvalue problem for the quadratic form
        theta = 0.5 * np.arctan2(B, A - C) if abs(A - C) > 1e-10 else 0
        
        # Calculate the semi-axis lengths using the eigenvalues
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Rotate to principal axes
        A_rot = A * cos_theta**2 + B * cos_theta * sin_theta + C * sin_theta**2
        C_rot = A * sin_theta**2 - B * cos_theta * sin_theta + C * cos_theta**2
        
        # Calculate semi-axes lengths
        # The general form becomes: A_rot*(x')² + C_rot*(y')² = constant
        constant = A*h**2 + B*h*k + C*k**2 + D*h + E*k + F
        
        if A_rot <= 0 or C_rot <= 0 or constant >= 0:
            return fit_ellipse_geometric(points)
        
        a = np.sqrt(-constant / A_rot)  # semi-axis along rotated x
        b = np.sqrt(-constant / C_rot)  # semi-axis along rotated y
        
        # Ensure a >= b (a is semi-major axis)
        if b > a:
            a, b = b, a
            theta += np.pi/2
        
        # Calculate foci
        if a > b:
            c_focal = np.sqrt(a**2 - b**2)
            # Foci are along the major axis
            focus1_x = h + c_focal * np.cos(theta)
            focus1_y = k + c_focal * np.sin(theta)
            focus2_x = h - c_focal * np.cos(theta)
            focus2_y = k - c_focal * np.sin(theta)
        else:
            focus1_x = focus2_x = h
            focus1_y = focus2_y = k
        
        focus1 = (focus1_x, focus1_y)
        focus2 = (focus2_x, focus2_y)
        
        # Return results
        return (focus1, focus2), (h, k, a, b, theta)
        
    except Exception as e:
        st.error(f"Ellipse fitting error: {str(e)}")
        return fit_ellipse_geometric(points)

def fit_ellipse_force(points):
    """Force fit any curve to an ellipse shape - even if it's not truly elliptical"""
    if len(points) < 3:
        return None, None
    
    try:
        x = points[:, 0]
        y = points[:, 1]
        
        # Method 1: Bounding box approach with center estimation
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Find the center by analyzing the curve shape
        # Use weighted centroid based on distance from extremes
        weights = 1.0 / (1.0 + 0.1 * (np.abs(x - (x_min + x_max)/2) + np.abs(y - (y_min + y_max)/2)))
        h = np.average(x, weights=weights)
        k = np.average(y, weights=weights)
        
        # Calculate distances from center to all points
        distances = np.sqrt((x - h)**2 + (y - k)**2)
        angles = np.arctan2(y - k, x - h)
        
        # Fit an ellipse by finding the best a and b that minimize distance errors
        # For each point, calculate what the ellipse radius should be at that angle
        
        # Estimate initial semi-axes
        x_spread = (x_max - x_min) / 2
        y_spread = (y_max - y_min) / 2
        
        # Use the maximum extent as initial guess
        a_init = max(x_spread, np.max(distances)) * 1.1
        b_init = max(y_spread, np.max(distances)) * 0.9
        
        # Ensure a >= b
        if b_init > a_init:
            a_init, b_init = b_init, a_init
        
        # Refine using the actual data points
        # Calculate what a and b should be to best fit the points
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # For each point, if it were on an ellipse: r = ab/sqrt((b*cos(θ))² + (a*sin(θ))²)
        # Rearrange to estimate a and b
        
        # Use robust estimation
        a = np.percentile(distances / np.sqrt(sin_angles**2 + (cos_angles * b_init/a_init)**2), 80)
        b = np.percentile(distances / np.sqrt((sin_angles * a_init/b_init)**2 + cos_angles**2), 80)
        
        # Ensure reasonable values
        a = max(a, x_spread * 1.2)
        b = max(b, y_spread * 1.2)
        
        if b > a:
            a, b = b, a
        
        # Calculate foci
        if a > b and a > 1e-6:
            c = np.sqrt(max(0, a**2 - b**2))
            focus1 = (h + c, k)
            focus2 = (h - c, k)
        else:
            focus1 = focus2 = (h, k)
        
        return (focus1, focus2), (h, k, a, b, 0)
        
    except Exception as e:
        # Simple fallback
        h, k = np.mean(x), np.mean(y)
        a = np.std(x) * 3
        b = np.std(y) * 3
        if b > a:
            a, b = b, a
        
        if a > b:
            c = np.sqrt(a**2 - b**2)
            focus1 = (h + c, k)
            focus2 = (h - c, k)
        else:
            focus1 = focus2 = (h, k)
        
        return (focus1, focus2), (h, k, a, b, 0)

def reflect_ray(incident_dir, normal):
    """Calculate reflected ray direction using law of reflection"""
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
    if dx == 0 and dy == 0:
        return np.array([0, 1])  # Default normal
    
    tangent = np.array([dx, dy])
    tangent = tangent / np.linalg.norm(tangent)
    
    # Normal is perpendicular to tangent (rotate 90 degrees)
    normal = np.array([-tangent[1], tangent[0]])
    
    return normal

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
        
        seg_dir = p2 - p1
        
        # Set up system of equations
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

def trace_ray_path(start_point, direction, main_points, sub_points):
    """Trace a ray through the dual reflector system"""
    ray_path = [start_point]
    current_pos = np.array(start_point)
    current_dir = np.array(direction)
    current_dir = current_dir / np.linalg.norm(current_dir)
    
    # Trace ray to main reflector first
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
            final_point = sub_hit + final_dir * 20  # Extend 20 units
            ray_path.append(final_point)
    
    return ray_path

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
    
    # Plot ray paths with different colors for each segment
    if ray_data is not None and len(ray_data) > 0:
        for i, ray_path in enumerate(ray_data):
            if len(ray_path) > 1:
                ray_x = [p[0] for p in ray_path]
                ray_y = [p[1] for p in ray_path]
                
                # Different colors for different ray segments
                if len(ray_path) >= 4:  # Complete path: start -> main -> sub -> final
                    # Incident ray (start to main)
                    fig.add_trace(go.Scatter(
                        x=[ray_x[0], ray_x[1]],
                        y=[ray_y[0], ray_y[1]],
                        mode='lines',
                        name=f'Incident Ray {i+1}' if i == 0 else None,
                        line=dict(color='cyan', width=3),
                        showlegend=(i == 0)
                    ))
                    
                    # Reflected ray (main to sub)
                    fig.add_trace(go.Scatter(
                        x=[ray_x[1], ray_x[2]],
                        y=[ray_y[1], ray_y[2]],
                        mode='lines',
                        name=f'Main→Sub Ray {i+1}' if i == 0 else None,
                        line=dict(color='yellow', width=3),
                        showlegend=(i == 0)
                    ))
                    
                    # Final ray (sub to focus)
                    fig.add_trace(go.Scatter(
                        x=[ray_x[2], ray_x[3]],
                        y=[ray_y[2], ray_y[3]],
                        mode='lines',
                        name=f'Final Ray {i+1}' if i == 0 else None,
                        line=dict(color='lime', width=3),
                        showlegend=(i == 0)
                    ))
                    
                    # Add markers at reflection points
                    fig.add_trace(go.Scatter(
                        x=[ray_x[1], ray_x[2]],
                        y=[ray_y[1], ray_y[2]],
                        mode='markers',
                        name='Reflection Points' if i == 0 else None,
                        marker=dict(size=8, color='orange', symbol='circle'),
                        showlegend=(i == 0)
                    ))
                else:
                    # Simpler path
                    fig.add_trace(go.Scatter(
                        x=ray_x,
                        y=ray_y,
                        mode='lines+markers',
                        name=f'Ray {i+1}' if len(ray_data) <= 5 else None,
                        line=dict(color='lime', width=2),
                        marker=dict(size=4, color='lime'),
                        showlegend=(i < 3)
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
    <div class="info-box" style="background-color: #2d3142; color: #ffffff; border: 1px solid #4a4e69;">
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
    
    # Initialize all variables early
    show_rays = False
    num_rays = 3
    ray_spacing = 1.0
    ray_angle = -70.0
    ray_start_distance = 10.0
    
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
        
        # Option to input polynomial coefficients directly
        input_type = st.sidebar.radio(
            "Input Type:",
            ["XY Coordinates", "Polynomial Coefficients"]
        )
        
        if input_type == "Polynomial Coefficients":
            st.sidebar.write("**Choose Reflector:**")
            poly_target = st.sidebar.selectbox("Polynomial for:", ["Sub Reflector (4th order)", "Main Reflector (3rd order)"])
            
            if poly_target == "Sub Reflector (4th order)":
                st.sidebar.write("**Sub: y = As/10000×x⁴ + Bs/100×x³ + Cs/10×x² + Ds/10×x + Es**")
                
                # Sub reflector coefficients from your data
                As = st.sidebar.number_input("As:", value=-1.257614832, format="%.9f")
                Bs = st.sidebar.number_input("Bs:", value=0.919978748, format="%.9f") 
                Cs = st.sidebar.number_input("Cs:", value=-1.768773487, format="%.9f")
                Ds = st.sidebar.number_input("Ds:", value=7.000282416, format="%.9f")
                Es = st.sidebar.number_input("Es:", value=2.23545, format="%.5f")
                
                # X range for sub reflector
                x_min = st.sidebar.number_input("X Min:", value=0.0, step=0.1)
                x_max = st.sidebar.number_input("X Max:", value=5.0, step=0.1)
                num_points = st.sidebar.slider("Number of Points:", 10, 100, 50)
                
                # Generate sub reflector curve
                x_vals = np.linspace(x_min, x_max, num_points)
                y_vals = (As/10000.0 * x_vals**4 + 
                         Bs/100.0 * x_vals**3 + 
                         Cs/10.0 * x_vals**2 + 
                         Ds/10.0 * x_vals + 
                         Es)
                
                sub_points = np.column_stack([x_vals, y_vals])
                st.sidebar.success(f"Sub: {len(sub_points)} points generated from 4th order polynomial")
                
            else:  # Main Reflector
                st.sidebar.write("**Main: y = Am/100000×x³ + Bm/1000×x² + Cm×x/100 + Dm/10 - 30**")
                
                # Main reflector coefficients from your data
                Am = st.sidebar.number_input("Am:", value=-1.409451197, format="%.9f")
                Bm = st.sidebar.number_input("Bm:", value=9.21575011, format="%.8f") 
                Cm = st.sidebar.number_input("Cm:", value=-8.360930461, format="%.9f")
                Dm = st.sidebar.number_input("Dm:", value=1.894145843, format="%.9f")
                
                # X range for main reflector
                x_min = st.sidebar.number_input("X Min:", value=5.0, step=0.5)
                x_max = st.sidebar.number_input("X Max:", value=47.5, step=0.5)
                num_points = st.sidebar.slider("Number of Points:", 20, 200, 100)
                
                # Generate main reflector curve
                x_vals = np.linspace(x_min, x_max, num_points)
                y_vals = (Am/100000.0 * x_vals**3 + 
                         Bm/1000.0 * x_vals**2 + 
                         Cm * x_vals/100.0 + 
                         Dm/10.0 - 30.0)
                
                main_points = np.column_stack([x_vals, y_vals])
                st.sidebar.success(f"Main: {len(main_points)} points generated from 3rd order polynomial")
            
            # Option to generate both at once
            if st.sidebar.button("🔄 Generate Both Reflectors"):
                # Sub reflector
                As, Bs, Cs, Ds, Es = -1.257614832, 0.919978748, -1.768773487, 7.000282416, 2.23545
                x_sub = np.linspace(0.0, 5.0, 50)
                y_sub = (As/10000.0 * x_sub**4 + Bs/100.0 * x_sub**3 + 
                        Cs/10.0 * x_sub**2 + Ds/10.0 * x_sub + Es)
                sub_points = np.column_stack([x_sub, y_sub])
                
                # Main reflector  
                Am, Bm, Cm, Dm = -1.409451197, 9.21575011, -8.360930461, 1.894145843
                x_main = np.linspace(5.0, 47.5, 100)
                y_main = (Am/100000.0 * x_main**3 + Bm/1000.0 * x_main**2 + 
                         Cm * x_main/100.0 + Dm/10.0 - 30.0)
                main_points = np.column_stack([x_main, y_main])
                
                st.sidebar.success("✅ Both reflectors generated!")
                st.sidebar.write(f"Main: {len(main_points)} points")
                st.sidebar.write(f"Sub: {len(sub_points)} points")
        
        else:
            # Original XY coordinate input
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
    
    # Curve fitting options
    st.sidebar.subheader("🔧 Curve Fitting")
    if st.sidebar.button("🔄 Refresh Fit"):
        st.rerun()
    
    fit_info = st.sidebar.info("For ellipse: Uses 5 distributed points to find the full ellipse that your partial curve lies on.")
    
    show_fits = st.sidebar.checkbox("Show Curve Fits", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Reflector Analysis")
        
        # Analyze main reflector
        main_focus = None
        main_params = None
        
        if main_type == "Parabola":
            main_focus, main_params = fit_parabola_advanced(main_points)
        
        # Analyze sub reflector
        sub_foci = None
        sub_params = None
        fitted_curve = None
        
        if sub_type == "Force Ellipse":
            sub_foci, sub_params = fit_ellipse_force(sub_points)
        elif sub_type == "Ellipse":
            sub_foci, sub_params = fit_ellipse_through_points(sub_points)
        elif sub_type == "Polynomial":
            sub_foci, fitted_curve = fit_polynomial_curve(sub_points, degree=3)
        
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
            ray_direction = [np.sin(ray_angle_rad), -np.cos(ray_angle_rad)]
            
            # Starting positions for parallel rays
            if abs(ray_angle) > 45:  # Nearly vertical rays
                # Start from above
                center_x = (x_min + x_max) / 2
                start_y = y_max + ray_start_distance
                
                for i in range(num_rays):
                    offset = (i - num_rays // 2) * ray_spacing
                    start_point = [center_x + offset, start_y]
                    
                    ray_path = trace_ray_path_improved(start_point, ray_direction, main_points, sub_points)
                    if len(ray_path) > 1:
                        ray_data.append(ray_path)
            else:  # Nearly horizontal rays
                # Start from left/right
                center_y = (y_min + y_max) / 2
                start_x = x_min - ray_start_distance if ray_angle >= 0 else x_max + ray_start_distance
                
                for i in range(num_rays):
                    offset = (i - num_rays // 2) * ray_spacing
                    start_point = [start_x, center_y + offset]
                    
                    ray_path = trace_ray_path_improved(start_point, ray_direction, main_points, sub_points)
                    if len(ray_path) > 1:
                        ray_data.append(ray_path)
        
        # Create plot
        fig = create_reflector_plot(
            main_points, sub_points, main_type, sub_type,
            main_focus, sub_foci, show_rays, ray_data, 
            sub_params=sub_params, fitted_curve=fitted_curve
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
        
        if sub_params is not None and (sub_type == "Ellipse" or sub_type == "Force Ellipse"):
            st.write("**Ellipse Parameters:**")
            if len(sub_params) == 5:
                h, k, a, b, theta = sub_params
                st.write(f"Center: ({h:.3f}, {k:.3f})")
                st.write(f"Semi-major axis (a): {a:.3f}")
                st.write(f"Semi-minor axis (b): {b:.3f}")
                st.write(f"Rotation: {np.degrees(theta):.1f}°")
                eccentricity = np.sqrt(1 - (b**2)/(a**2)) if a > 0 else 0
                st.write(f"Eccentricity: {eccentricity:.3f}")
                if sub_type == "Force Ellipse":
                    st.warning("⚠️ Force-fitted ellipse - may not represent true geometry")
            else:
                h, k, a, b = sub_params
                st.write(f"Center: ({h:.3f}, {k:.3f})")
                st.write(f"Semi-major axis (a): {a:.3f}")
                st.write(f"Semi-minor axis (b): {b:.3f}")
                eccentricity = np.sqrt(1 - (b**2)/(a**2)) if a > 0 else 0
                st.write(f"Eccentricity: {eccentricity:.3f}")
                if sub_type == "Force Ellipse":
                    st.warning("⚠️ Force-fitted ellipse - may not represent true geometry")
        
        elif fitted_curve is not None and sub_type == "Polynomial":
            st.write("**Polynomial Parameters:**")
            coeffs, x_extended, y_extended = fitted_curve
            st.write(f"Degree: {len(coeffs)-1}")
            for i, coeff in enumerate(coeffs):
                power = len(coeffs) - 1 - i
                if power == 0:
                    st.write(f"Constant: {coeff:.6f}")
                elif power == 1:
                    st.write(f"Linear: {coeff:.6f}")
                else:
                    st.write(f"x^{power}: {coeff:.6f}")
            st.info("⚠️ Polynomial fit - focus is approximate")
        
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
                
                # Estimate antenna gain based on convergence
                # Tighter convergence = higher gain (rough approximation)
                convergence_total = np.sqrt(convergence_x**2 + convergence_y**2)
                if convergence_total < 1.0:
                    gain_estimate = 55 + (1.0 - convergence_total) * 5
                    st.success(f"**Estimated Gain:** ~{gain_estimate:.1f} dBi")
                elif convergence_total < 3.0:
                    gain_estimate = 52 + (3.0 - convergence_total) * 1.5
                    st.info(f"**Estimated Gain:** ~{gain_estimate:.1f} dBi")
                else:
                    st.warning(f"**Poor Focus:** Convergence = {convergence_total:.2f}")
            
            st.write("**Ray Path Info:**")
            for i, ray_path in enumerate(ray_data[:3]):  # Show first 3 rays
                if len(ray_path) >= 3:
                    st.write(f"Ray {i+1}: {len(ray_path)} segments")
        
        # Optimization context
        st.subheader("📊 Optimization Context")
        st.info("""
        **Your Data Context:**
        - **Main Reflector:** 3rd order polynomial optimized for gain
        - **Sub Reflector:** 4th order polynomial with diameter optimization
        - **Gain Results:** 51-57 dBi depending on frequency and sub size
        - **Optimization:** Post-processed for maximum directivity
        """)
        
        if sub_type == "Force Ellipse" or sub_type == "Ellipse":
            st.warning("""
            ⚠️ **Note:** Your curves are polynomial-optimized, not true conics.
            Ellipse fitting provides approximate focus locations for analysis.
            """)
        
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
