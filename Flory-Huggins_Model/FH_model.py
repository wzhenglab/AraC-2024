import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import binary_erosion, generate_binary_structure

def s_value(a, b):
    return 1 - a - b
    
def F(a, b, A, B, chi_AS, chi_AB, chi_BS):
    s = s_value(a, b)
    return chi_AS * a * s + chi_AB * a * b + chi_BS * b * s + a * np.log(a) / A + b * np.log(b) / B + s * np.log(s)

def d2F_da2(a, b, A, B, chi_AS, chi_AB, chi_BS):
    s = s_value(a, b)
    return -2 * chi_AS + 1 / (a * A) + 1 / s

def d2F_db2(a, b, A, B, chi_AS, chi_AB, chi_BS):
    s = s_value(a, b)
    return -2 * chi_BS + 1 / (b * B) + 1 / s

def d2F_dadb(a, b, A, B, chi_AS, chi_AB, chi_BS):
    s = s_value(a, b)
    return -chi_AS + chi_AB - chi_BS + 1 / s

def Hessian(a, b, A, B, chi_AS, chi_AB, chi_BS):
    H11 = d2F_da2(a, b, A, B, chi_AS, chi_AB, chi_BS)
    H12 = d2F_dadb(a, b, A, B, chi_AS, chi_AB, chi_BS)
    H21 = d2F_dadb(a, b, A, B, chi_AS, chi_AB, chi_BS)
    H22 = d2F_db2(a, b, A, B, chi_AS, chi_AB, chi_BS)
    return np.array([[H11,H12],[H21,H22]])

def Volume_fraction(A, B, numbers):
    V = numbers[0] * A + numbers[1] * B + numbers[2]
    return numbers[0] * A / V, numbers[1] * B /V, V

def find_negative_boundary(matrix):
    """
    Find and order the outermost points of a contiguous negative-valued region in a 2D matrix.
    Excludes matrix boundary points when the negative area touches the edges.
    
    Args:
        matrix (np.ndarray): Input 2D matrix (M,N)
        
    Returns:
        list: Ordered list of (row, col) points forming the boundary polygon,
              or empty list if no valid boundary points found
    """
    # Create binary mask of negative values
    mask = matrix < 0
    
    # Check if any negative values exist
    if not np.any(mask):
        return []
    
    # Create structure for 8-connected neighborhood
    struct = generate_binary_structure(2, 2)
    
    # Erode the mask to find interior points
    eroded = binary_erosion(mask, structure=struct)
    
    # Boundary is where mask is True and eroded is False
    boundary_mask = mask & ~eroded
    
    # Get boundary points (excluding matrix edges)
    m, n = matrix.shape
    boundary_points = []
    for i in range(4, m-4):  # Exclude first and last rows
        for j in range(4, n-4):  # Exclude first and last columns
            if (i/m)+(j/n)>0.95:
                continue
            elif boundary_mask[i, j]:
                boundary_points.append([i, j])
    
    # Order points to form a continuous polygon
    ordered_points = order_boundary_points(boundary_points)
    
    return ordered_points

def order_boundary_points(points):
    """
    Order boundary points to form a continuous polygon.
    
    Args:
        points (list): List of [row, col] boundary points
        
    Returns:
        list: Ordered list of points
    """
    if len(points) < 3:
        return points
    
    # Start with the first point
    ordered = [points[0]]
    remaining = points[1:]
    
    while remaining:
        last = ordered[-1]
        # Find the closest point in remaining
        closest_idx = None
        min_dist = float('inf')
        
        for i, p in enumerate(remaining):
            dist = (p[0]-last[0])**2 + (p[1]-last[1])**2  # Squared Euclidean distance
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        if closest_idx is not None:
            next_point = remaining.pop(closest_idx)
            ordered.append(next_point)
        else:
            break
    
    # Close the polygon if we have enough points
    if len(ordered) >= 3:
        ordered.append(ordered[0])
    
    return ordered

chi_list = [[1.2, 0.0, 0.0],\
            [1.2, 0.0,-0.6],\
            [1.2, 0.0,-1.2],\
            [1.2,-0.6,-0.6],\
            [1.2,-0.6, 0.0],\
            [1.2,-0.6, 0.6],\
            [1.2, 0.6,-0.6],\
            [1.2, 0.6, 0.0],\
            [1.2, 0.6, 0.6],\
            [1.2, 1.2,-0.6],\
            [1.2, 1.2, 0.0],\
            [1.2, 1.2, 0.6],\
            [0.0, 0.0,-4.0],\
            [0.0, 0.0, 1.0]]

A = 10
B = 10
V_total = 1002
stride_search = 0.2

for chi in chi_list:
    print(chi)
    chi_AS = chi[0]
    chi_BS = chi[1]
    chi_AB = chi[2]
    possible_inits = []

    Max_A = int((V_total-2)/A)
    Max_B = int((V_total-2)/B)

    for NA in range(2,Max_A+1):
        for NB in range(2,Max_B+1):
            if NA*A+NB*B<V_total-1:
                possible_inits.append([NA, NB, V_total-(NA*A+NB*B)])

    init_volume_fracs = []
    for init in possible_inits:
        init_volume_fracs.append(Volume_fraction(A, B, init))

    detHs = []
    for volume_frac in init_volume_fracs:
        detHs.append(np.linalg.det(Hessian(volume_frac[0], volume_frac[1], A, B, chi_AS, chi_AB, chi_BS)))

    print(np.min(np.array(detHs)),np.max(np.array(detHs)))

    detH_matrix = np.ones((Max_A+1, Max_B+1))
    for k, init in enumerate(possible_inits):
        detH_matrix[init[0], init[1]] = detHs[k]

    boundary = find_negative_boundary(detH_matrix)

    spin_bd_inits = []
    spin_bd = []

    for bd in boundary:
        NA = bd[0]
        NB = bd[1]
        spin_bd_inits.append([NA, NB, V_total-(NA*A+NB*B)])
        spin_bd.append(Volume_fraction(A, B, [NA, NB, V_total-(NA*A+NB*B)]))

    phases_min = []
    for init in tqdm(spin_bd_inits):
        phase_1s = []
        phase_2s = []
        for NA1 in np.arange(1, init[0], stride_search):
            for NB1 in np.arange(1, init[1], stride_search):
                for NS1 in np.arange(1, init[2], stride_search):
                    volume_frac_1 = Volume_fraction(A, B, [NA1, NB1, NS1])
                    volume_frac_2 = Volume_fraction(A, B, [init[0]-NA1, init[1]-NB1, init[2]-NS1])
                    phase_1s.append(volume_frac_1)
                    phase_2s.append(volume_frac_2)

        phase_1s = np.array(phase_1s)
        phase_2s = np.array(phase_2s)
        F_totals = phase_1s[:,2] * F(phase_1s[:,0], phase_1s[:,1], A, B, chi_AS, chi_AB, chi_BS) +\
                phase_2s[:,2] * F(phase_2s[:,0], phase_2s[:,1], A, B, chi_AS, chi_AB, chi_BS)

        min_index = np.argmin(F_totals)
        phase_1_min = phase_1s[min_index]
        phase_2_min = phase_2s[min_index]
        if np.linalg.det(Hessian(phase_1_min[0], phase_1_min[1], A, B, chi_AS, chi_AB, chi_BS)) > 0 and\
        np.linalg.det(Hessian(phase_2_min[0], phase_2_min[1], A, B, chi_AS, chi_AB, chi_BS)) > 0:
            phases_min.append(np.concatenate([phase_1_min[:], phase_2_min[:]]))

    np.save('FH-V_' + str(V_total) + '-size_' + str(A) + '_' + str(A) + '-chi_' + str(chi_AS) + '_' + str(chi_BS) + '_' + str(chi_AB) + '-finer_stride' + str(stride_search) + '.npy', phases_min)
