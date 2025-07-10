from geqtrain.data import AtomicDataDict
import torch
import itertools
from e3nn import o3
from e3nn.o3._wigner import _so3_clebsch_gordan
from typing import Dict, Tuple, Optional

# ==============================================================================
# SECTION 1: DYNAMIC IRREPS AND INTERACTION MAP GENERATION
# ==============================================================================

def get_irreps(basis_def: Dict[int, Dict[str, int]]) -> Tuple[o3.Irreps, o3.Irreps, Dict, Dict]:
    """
    Dynamically generates Irreps and interaction maps for nodes and edges
    based on the provided basis set definition. This is a robust implementation
    based on physical coupling rules.

    Args:
        basis_def: A dictionary mapping atomic numbers to their orbital definitions.
                   e.g., {8: {'1s': 1, '2s': 1, '2p': 1}, 1: {'1s': 1}}

    Returns:
        A tuple containing:
        - node_irreps (o3.Irreps): The required Irreps for node features.
        - edge_irreps (o3.Irreps): The required Irreps for edge features.
        - node_interaction_map (Dict): Maps on-site interactions to feature slices.
        - edge_interaction_map (Dict): Maps inter-site interactions to feature slices.
    """
    
    def orbital_str_to_l(s: str) -> int:
        if 's' in s: return 0
        if 'p' in s: return 1
        if 'd' in s: return 2
        if 'f' in s: return 3
        return -1

    node_map = {}
    edge_map = {}
    
    # Generate a map of all required interactions for on-site (node) blocks
    for Z, orbitals in basis_def.items():
        orbital_names = [name for name, present in orbitals.items() if present]
        for orb1_name, orb2_name in itertools.combinations_with_replacement(orbital_names, 2):
            l1, l2 = orbital_str_to_l(orb1_name), orbital_str_to_l(orb2_name)
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                key = f"node:{Z}:{orb1_name}-{orb2_name}_l{l3}"
                if key not in node_map:
                    node_map[key] = {'l': l3}
    
    # Generate a map of all required interactions for inter-site (edge) blocks
    element_pairs = itertools.combinations_with_replacement(list(basis_def.keys()), 2)
    for Z1, Z2 in element_pairs:
        orbitals1 = [name for name, present in basis_def[Z1].items() if present]
        orbitals2 = [name for name, present in basis_def[Z2].items() if present]
        for orb1_name in orbitals1:
            for orb2_name in orbitals2:
                l1, l2 = orbital_str_to_l(orb1_name), orbital_str_to_l(orb2_name)
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    key = f"edge:{Z1}-{Z2}:{orb1_name}-{orb2_name}_l{l3}"
                    if key not in edge_map:
                        edge_map[key] = {'l': l3}

    def finalize_map_and_get_irreps(interaction_map: Dict) -> Tuple[Dict, o3.Irreps]:
        final_map = {}
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5:0} # Support up to l=5
        
        for key in sorted(interaction_map.keys()):
            l = interaction_map[key]['l']
            final_map[key] = {'l': l, 'idx': counts[l]}
            counts[l] += 1
        
        irreps_str_list = [f"{count}x{l}e" if l % 2 == 0 else f"{count}x{l}o"
                           for l, count in counts.items() if count > 0]
        return final_map, o3.Irreps("+".join(irreps_str_list)).sort().irreps

    node_interaction_map, node_irreps = finalize_map_and_get_irreps(node_map)
    edge_interaction_map, edge_irreps = finalize_map_and_get_irreps(edge_map)
    
    return node_irreps, edge_irreps, node_interaction_map, edge_interaction_map

# ==============================================================================
# SECTION 2: HAMILTONIAN ASSEMBLY
# ==============================================================================

def construct_interaction_block(
    l1: int,
    l2: int,
    predicted_components: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """
    Constructs an interaction matrix block M^{l1,l2} from its predicted irreducible components.
    This implements the sum of tensor product expansions as described in Eq. 19
    of the PhiSNet paper, where each expansion (Eq. 8) is done using
    e3nn.o3.clebsch_gordan.

    Args:
        l1 (int): Angular momentum of the first orbital (e.g., 0 for s, 1 for p).
        l2 (int): Angular momentum of the second orbital.
        predicted_components (Dict[int, torch.Tensor]): A dictionary where keys are
            l3 values (int) and values are the corresponding predicted irrep
            tensors a^(l3). Each tensor should have shape (2*l3+1,).
        device (Optional[torch.device]): Device for the output tensor.
        dtype (Optional[torch.dtype]): Dtype for the output tensor.

    Returns:
        torch.Tensor: The constructed matrix block of shape (2*l1+1, 2*l2+1).
    """
    dim1 = 2 * l1 + 1
    dim2 = 2 * l2 + 1
    matrix_block = torch.zeros(dim1, dim2, device=torch.get_default_device(), dtype=torch.get_default_dtype())

    # Sum over all valid l3 components that contribute to this l1-l2 interaction
    # l3 ranges from |l1 - l2| to l1 + l2
    for l3_val in range(abs(l1 - l2), l1 + l2 + 1):
        if l3_val in predicted_components:
            a_l3_tensor = predicted_components[l3_val]
            if a_l3_tensor.shape != (2 * l3_val + 1,):
                raise ValueError(
                    f"Tensor for l3={l3_val} has incorrect shape {a_l3_tensor.shape}. "
                    f"Expected ({2*l3_val + 1},)."
                )

            # Get the Clebsch-Gordan coefficients from e3nn
            # W has shape (2*l1+1, 2*l2+1, 2*l3+1)
            W_cgc = _so3_clebsch_gordan(l1, l2, l3_val).to(torch.get_default_dtype())

            # Perform the tensor product expansion for this a_l3 component
            # M_contrib_{m1,m2} = sum_{m3} W_{m1,m2,m3} (a_l3)_{m3}
            matrix_contribution = torch.einsum('ijk,k->ij', W_cgc, a_l3_tensor.to(device=torch.get_default_device(), dtype=torch.get_default_dtype()))
            matrix_block += matrix_contribution
            
    return matrix_block

def assemble_hamiltonian(
    node_components: torch.Tensor,
    node_irreps: o3.Irreps,
    edge_components: torch.Tensor,
    edge_irreps: o3.Irreps,
    edge_index: torch.Tensor,
    node_species: torch.Tensor,
    basis_def: Dict[int, Dict[str, int]],
    node_interaction_map: Dict,
    edge_interaction_map: Dict,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Assembles the full Hamiltonian matrix using the dynamic interaction maps.
    This is a new, cleaner, and more robust implementation.
    """
    if batch is None:
        batch = torch.zeros_like(node_species)

    def orbital_str_to_l(s: str) -> int:
        if 's' in s: return 0
        if 'p' in s: return 1
        if 'd' in s: return 2
        return -1

    # 1. Prepare data structures for assembly
    atom_orbital_shells = []
    for i in range(len(node_species)):
        Z = node_species[i].item()
        orbitals = [name for name, present in basis_def[Z].items() if present]
        atom_orbital_shells.append(orbitals)

    n_basis_per_node = torch.tensor([
        sum(2 * orbital_str_to_l(orb_name) + 1 for orb_name in shells)
        for shells in atom_orbital_shells
    ], device=node_components.device)
    
    total_basis_size = n_basis_per_node.sum().item()
    atom_offsets = torch.cat([torch.tensor([0]), torch.cumsum(n_basis_per_node, dim=0)[:-1]])
    
    H = torch.zeros(total_basis_size, total_basis_size, 
                    device=node_components.device, dtype=node_components.dtype)

    node_slices = {ir.ir.l: s for ir, s in zip(node_irreps, node_irreps.slices())}
    edge_slices = {ir.ir.l: s for ir, s in zip(edge_irreps, edge_irreps.slices())}

    # 2. Off-site blocks
    edge_lookup = {(i.item(), j.item()): k for k, (i, j) in enumerate(zip(edge_index[0], edge_index[1]))}
    
    for i, j in itertools.combinations(range(len(node_species)), 2):
        k = edge_lookup.get((i, j), edge_lookup.get((j, i)))
        if k is None: continue

        Z_i, Z_j = node_species[i].item(), node_species[j].item()
        orbitals_i, orbitals_j = atom_orbital_shells[i], atom_orbital_shells[j]
        atom_offset_i, atom_offset_j = atom_offsets[i], atom_offsets[j]

        for orb_i_idx, orb_i_name in enumerate(orbitals_i):
            for orb_j_idx, orb_j_name in enumerate(orbitals_j):
                l_i, l_j = orbital_str_to_l(orb_i_name), orbital_str_to_l(orb_j_name)
                
                predicted_components = {}
                for l3 in range(abs(l_i - l_j), l_i + l_j + 1):
                    key1 = f"edge:{Z_i}-{Z_j}:{orb_i_name}-{orb_j_name}_l{l3}"
                    key2 = f"edge:{Z_j}-{Z_i}:{orb_j_name}-{orb_i_name}_l{l3}"
                    key = key1 if key1 in edge_interaction_map else key2
                    
                    if key in edge_interaction_map:
                        map_info = edge_interaction_map[key]
                        channel_idx, l_val = map_info['idx'], map_info['l']
                        l_tensor = edge_components[k, edge_slices[l_val]]
                        start = channel_idx * (2 * l_val + 1)
                        predicted_components[l3] = l_tensor[start : start + 2 * l_val + 1]
                
                if not predicted_components: continue
                
                block = construct_interaction_block(l_j, l_i, predicted_components)
                
                offset_i_orb = atom_offset_i + sum(2*orbital_str_to_l(o)+1 for o_idx, o in enumerate(orbitals_i) if o_idx < orb_i_idx)
                offset_j_orb = atom_offset_j + sum(2*orbital_str_to_l(o)+1 for o_idx, o in enumerate(orbitals_j) if o_idx < orb_j_idx)
                
                # We are building H_ji where j > i (lower triangle)
                H[offset_j_orb:offset_j_orb+2*l_j+1, offset_i_orb:offset_i_orb+2*l_i+1] = block

    # 3. Symmetrization
    H = H + H.T

    # 4. On-site blocks (add to the diagonal)
    for i in range(len(node_species)):
        Z = node_species[i].item()
        orbitals = atom_orbital_shells[i]
        
        for orb1_idx, orb1_name in enumerate(orbitals):
            for orb2_idx, orb2_name in enumerate(orbitals):
                if orb1_idx < orb2_idx: continue # Build only the lower triangle of blocks within the diagonal block

                l1, l2 = orbital_str_to_l(orb1_name), orbital_str_to_l(orb2_name)
                
                predicted_components = {}
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    key1 = f"node:{Z}:{orb1_name}-{orb2_name}_l{l3}"
                    key2 = f"node:{Z}:{orb2_name}-{orb1_name}_l{l3}"
                    key = key1 if key1 in node_interaction_map else key2
                    
                    if key in node_interaction_map:
                        map_info = node_interaction_map[key]
                        channel_idx, l_val = map_info['idx'], map_info['l']
                        l_tensor_slice = node_components[i, node_slices[l_val]]
                        start = channel_idx * (2 * l_val + 1)
                        predicted_components[l3] = l_tensor_slice[start : start + 2 * l_val + 1]
                    else:
                        pass

                if not predicted_components: continue

                block = construct_interaction_block(l1, l2, predicted_components)
                
                offset1 = atom_offsets[i] + sum(2*orbital_str_to_l(o)+1 for o_idx, o in enumerate(orbitals) if o_idx < orb1_idx)
                offset2 = atom_offsets[i] + sum(2*orbital_str_to_l(o)+1 for o_idx, o in enumerate(orbitals) if o_idx < orb2_idx)
                
                H[offset1:offset1+2*l1+1, offset2:offset2+2*l2+1] = block
                # If it's not a diagonal block (l1!=l2), add its transpose
                if orb1_idx != orb2_idx:
                    H[offset2:offset2+2*l2+1, offset1:offset1+2*l1+1] = block.T
    
    return H

# ==============================================================================
# SECTION 3: HELPER AND DEBUGGING FUNCTIONS (Unchanged)
# ==============================================================================

def rotate_feature_tensor(irreps: o3.Irreps, tensor: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return tensor @ irreps.D_from_matrix(R).squeeze(0).T

def check_eigenvector_transformation(
    original_H: torch.Tensor, 
    rotated_H: torch.Tensor, 
    U: torch.Tensor
):
    """
    Checks if eigenvectors transform correctly under rotation, properly handling
    degenerate eigenvalues by comparing subspaces instead of individual vectors.

    Theory: If H' = U H U^†, then eigenvectors should transform as |ψ'> = U|ψ>.
    For degenerate subspaces, we check if the projector P' = P_rot is equal to U P_orig U^†.
    
    Args:
        original_H (torch.Tensor): The Hamiltonian matrix before rotation.
        rotated_H (torch.Tensor): The Hamiltonian matrix after rotation.
        U (torch.Tensor): The Wigner D-matrix for the rotation, U = D(R).
    """
    print("\n" + "="*60)
    print("ROBUST EIGENVECTOR TRANSFORMATION CHECK")
    print("="*60)

    # Ensure tensors are complex for eigenvalue decomposition
    U = U.to(torch.complex64)
    original_H_c = original_H.to(torch.complex64)
    rotated_H_c = rotated_H.to(torch.complex64)
    
    # 1. Compute eigenvectors and eigenvalues
    eigenvals_orig, eigenvecs_orig = torch.linalg.eig(original_H_c)
    eigenvals_rot, eigenvecs_rot = torch.linalg.eig(rotated_H_c)
    
    # 2. Sort by real part of eigenvalues to align corresponding vectors/spaces
    idx_orig = torch.argsort(eigenvals_orig.real)
    eigenvals_orig_sorted = eigenvals_orig[idx_orig]
    eigenvecs_orig_sorted = eigenvecs_orig[:, idx_orig]
    
    idx_rot = torch.argsort(eigenvals_rot.real)
    eigenvals_rot_sorted = eigenvals_rot[idx_rot]
    eigenvecs_rot_sorted = eigenvecs_rot[:, idx_rot]
    
    # Check eigenvalue invariance (should always be true for a correct H transformation)
    eigenval_diff = torch.max(torch.abs(eigenvals_orig_sorted.real - eigenvals_rot_sorted.real))
    print(f"Eigenvalue matching check: Max absolute difference = {eigenval_diff:.2e}")
    assert torch.allclose(eigenvals_orig_sorted.real, eigenvals_rot_sorted.real, atol=1e-5), "Eigenvalues do not match!"

    # 3. Identify groups of degenerate eigenvalues
    degeneracy_tol = 1e-5
    diffs = torch.diff(eigenvals_orig_sorted.real)
    # Get indices where a new energy level starts
    splits = torch.where(diffs > degeneracy_tol)[0] + 1
    degenerate_groups = torch.tensor_split(torch.arange(len(eigenvals_orig_sorted)), splits)

    print(f"Found {len(degenerate_groups)} distinct energy levels (groups of degenerate orbitals).")

    # 4. Check transformation for each group (either single vector or subspace)
    all_checks_pass = True
    for group_indices in degenerate_groups:
        if len(group_indices) == 1:
            # --- Non-degenerate case ---
            i = group_indices[0]
            # Predict how the original eigenvector should transform
            vec_orig = eigenvecs_orig_sorted[:, i]
            vec_predicted = U @ vec_orig
            
            # Get the actual eigenvector from the rotated Hamiltonian
            vec_actual = eigenvecs_rot_sorted[:, i]
            
            # Check the overlap |<actual|predicted>|. It should be 1.0.
            # This is robust to the arbitrary global phase factor e^(i*phi).
            overlap = torch.abs(torch.vdot(vec_actual, vec_predicted)).to(torch.get_default_dtype())
            
            print(f"  Non-degenerate Eigenvector {i+1}: |overlap| = {overlap:.4f}")
            if not torch.allclose(overlap, torch.tensor(1.0, dtype=torch.get_default_dtype()), atol=1e-5):
                all_checks_pass = False
        else:
            # --- Degenerate case ---
            start, end = group_indices[0], group_indices[-1]
            print(f"  Degenerate Subspace (Eigenvectors {start+1} to {end+1}):")
            
            # Get the basis vectors for the original and actual rotated subspaces
            V_orig = eigenvecs_orig_sorted[:, group_indices]
            V_actual = eigenvecs_rot_sorted[:, group_indices]

            # The projector onto the original subspace is P_orig = V_orig @ V_orig^†
            P_orig = V_orig @ V_orig.conj().T
            
            # The predicted projector for the rotated subspace is P_pred = U @ P_orig @ U^†
            P_predicted = U @ P_orig @ U.conj().T
            
            # The actual projector from the rotated eigensolver is P_actual = V_actual @ V_actual^†
            P_actual = V_actual @ V_actual.conj().T

            # Check if the projectors are the same
            error = torch.norm(P_predicted - P_actual)
            print(f"    Projector norm difference ||P_predicted - P_actual||: {error:.2e}")
            if error > 1e-5:
                all_checks_pass = False

    print("-" * 60)
    if all_checks_pass:
        print("✅ SUCCESS: Eigenvector transformation check passed (accounting for degeneracy).")
    else:
        print("❌ FAILURE: Eigenvector transformation check FAILED.")

# ==============================================================================
# SECTION 4: MAIN TEST SCRIPT
# ==============================================================================

def compute_hamiltonian(data):
    node_species = data[AtomicDataDict.NODE_TYPE_KEY]
    
    basis_def = {
        1: {'1s': 1},
        8: {'1s': 1, '2s': 1, '2p': 1},
        # Add other elements as needed, e.g. 6: {'1s': 1, ...} for Carbon
    }

    # --- Dynamic Setup ---
    # 1. Dynamically generate the required Irreps and interaction maps
    node_irreps, edge_irreps, node_interaction_map, edge_interaction_map = get_irreps(basis_def)
    print("--- Dynamically Generated Irreps ---")
    print(f"Node Irreps: {node_irreps}")
    print(f"Edge Irreps: {edge_irreps}")

    # 2. Extract predicted tensors matching the dynamic Irreps
    node_components = data[AtomicDataDict.NODE_FEATURES_KEY]
    edge_components = data[AtomicDataDict.EDGE_FEATURES_KEY]
    edge_index      = data[AtomicDataDict.EDGE_INDEX_KEY]
    
    # 3. Assemble the original Hamiltonian
    H_original = assemble_hamiltonian(
        node_components, node_irreps, edge_components, edge_irreps,
        edge_index, node_species, basis_def, 
        node_interaction_map, edge_interaction_map
    )

    print("\n--- Original Hamiltonian ---")
    print(f"Shape of the Hamiltonian: {H_original.shape}")
    print("Is the Hamiltonian symmetric?", torch.allclose(H_original, H_original.T, atol=1e-6))
    
    # --- EQUIVARIANCE CHECK ---
    print("\n--- EQUIVARIANCE CHECK ---")
    R = o3.rand_matrix()
    print("Generated a random 3x3 rotation matrix R.")

    # 4. METHOD A: Transform the fully assembled Hamiltonian (Ground Truth)
    def orbital_str_to_l(s: str) -> int:
        if 's' in s: return 0
        if 'p' in s: return 1
        if 'd' in s: return 2
        return -1
        
    basis_irreps_list = []
    for z_val in node_species:
        z = z_val.item()
        orbitals = [name for name, present in basis_def[z].items() if present]
        for orb_name in orbitals:
            l = orbital_str_to_l(orb_name)
            parity = "e" if l % 2 == 0 else "o"
            basis_irreps_list.append(f"1x{l}{parity}")
    
    actual_basis_irreps_str = "+".join(basis_irreps_list)
    print(f"Generated basis irreps for D-matrix: {actual_basis_irreps_str}")
    basis_irreps_for_D = o3.Irreps(actual_basis_irreps_str)
    D_matrix_H = basis_irreps_for_D.D_from_matrix(R)
    H_transformed_truth = D_matrix_H @ H_original @ D_matrix_H.T

    # 5. METHOD B: Rotate the input feature tensors, then assemble
    rotated_node_components = rotate_feature_tensor(node_irreps, node_components, R)
    rotated_edge_components = rotate_feature_tensor(edge_irreps, edge_components, R)
    
    H_from_rotated_inputs = assemble_hamiltonian(
        rotated_node_components, node_irreps,
        rotated_edge_components, edge_irreps,
        edge_index, node_species, basis_def,
        node_interaction_map, edge_interaction_map
    )
    
    # 6. VERIFICATION
    is_equivariant = torch.allclose(H_transformed_truth, H_from_rotated_inputs, atol=1e-5)
    print("\n--- VERIFICATION ---")
    print(f"Are the two methods for rotating the Hamiltonian equivalent? --> {is_equivariant}")
    if not is_equivariant:
        print("Difference matrix (should be close to zero):")
        print(H_transformed_truth - H_from_rotated_inputs)
    
    # 8. Check Eigenvector Transformation
    check_eigenvector_transformation(H_original, H_transformed_truth, D_matrix_H)

    return H_original

# ==============================================================================
# SECTION 5: PROPERTY CALCULATION
# ==============================================================================

def _placeholder_compute_overlap_matrix(total_basis_size, device, dtype) -> torch.Tensor:
    """
    Placeholder for overlap matrix S. In a real application, this would be
    computed by a quantum chemistry library (e.g., PySCF) based on the
    atomic positions and basis set.
    Here we return a matrix that is close to the identity but not exactly,
    to properly test the orthogonalization procedure.
    """
    S = torch.eye(total_basis_size, device=device, dtype=dtype)
    # Add small off-diagonal elements to simulate non-orthogonality
    S += torch.randn_like(S) * 0.01
    S = 0.5 * (S + S.T) # Ensure symmetry
    S.fill_diagonal_(1.0) # Ensure diagonal is 1
    return S

def _placeholder_compute_dipole_integrals(total_basis_size, device, dtype) -> torch.Tensor:
    """
    Placeholder for dipole integral matrices D. In a real application, this would
    be computed by a quantum chemistry library. Returns a zero tensor.
    """
    return torch.zeros(3, total_basis_size, total_basis_size, device=device, dtype=dtype)

def compute_properties(
    H: torch.Tensor,
    node_species: torch.Tensor,
    atom_positions: torch.Tensor,
    basis_def: Dict
) -> Dict:
    """
    Computes electronic properties from the Hamiltonian matrix.
    """
    device, dtype = H.device, H.dtype
    total_basis_size = H.shape[0]

    # 1. Get Overlap Matrix S (using placeholder)
    S = _placeholder_compute_overlap_matrix(total_basis_size, device, dtype)

    # 2. Transform the Generalized Eigenvalue Problem to Standard Form
    # First, find the transformation matrix X from S, such that X.T @ S @ X = I.
    # This is typically done via canonical orthogonalization.
    try:
        s_evals, s_evecs = torch.linalg.eigh(S)
        # Numerical stability: filter out very small eigenvalues of S
        # to avoid division by zero when computing s_evals**(-0.5)
        cutoff = 1e-7
        good_indices = s_evals > cutoff
        s_inv_sqrt = torch.zeros_like(s_evals)
        s_inv_sqrt[good_indices] = s_evals[good_indices]**(-0.5)
        # Transformation matrix X = U @ s^(-1/2)
        X = s_evecs @ torch.diag(s_inv_sqrt)
    except torch.linalg.LinAlgError:
        print("Warning: Eigendecomposition of overlap matrix S failed. Assuming S=I.")
        X = torch.eye(total_basis_size, device=device, dtype=dtype)
    
    # 3. Transform the Hamiltonian: H' = X^† H X
    H_prime = X.T.conj() @ H @ X

    # 4. Solve the Standard Eigenvalue Problem: H'C' = εC'
    try:
        eigenvalues, C_prime = torch.linalg.eigh(H_prime)
    except torch.linalg.LinAlgError:
        print("Warning: Eigenvalue decomposition of transformed H' failed. Returning empty results.")
        return {}

    # 5. Back-transform eigenvectors to the original AO basis: C = XC'
    C = X @ C_prime
    
    # 6. Calculate HOMO, LUMO, and Gap
    num_electrons = node_species.sum().item()
    num_occupied_orbitals = num_electrons // 2
    
    if num_occupied_orbitals >= len(eigenvalues):
        homo, lumo, gap = None, None, None
    else:
        homo = eigenvalues[num_occupied_orbitals - 1]
        lumo = eigenvalues[num_occupied_orbitals]
        gap = lumo - homo

    # 4. Calculate Dipole Moment
    # a. Get Dipole Integrals D (using placeholder)
    D = _placeholder_compute_dipole_integrals(total_basis_size, device, dtype)
    
    # b. Calculate Density Matrix P
    C_occupied = C[:, :num_occupied_orbitals]
    P = 2 * (C_occupied @ C_occupied.T.conj()) # conj().T for complex case
    
    # c. Calculate electronic contribution
    mu_elec = -torch.einsum('xij,ji->x', D, P)

    # d. Calculate nuclear contribution
    mu_nuc = torch.einsum('i,ix->x', node_species.to(dtype), atom_positions)
    
    # e. Total dipole moment
    mu = mu_elec + mu_nuc

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": C,
        "HOMO_energy": homo,
        "LUMO_energy": lumo,
        "HOMO_LUMO_gap": gap,
        "dipole_moment_vector": mu,
        "dipole_moment_total": torch.norm(mu)
    }

# ==============================================================================
# SECTION 3: ITERATIVE EIGENSOLVER FOR HOMO/LUMO
# ==============================================================================

def find_homo_lumo_iterative(
    H: torch.Tensor,
    num_electrons: int,
    energy_shift_guess: torch.Tensor,
    k: int = 6,
    num_iterations: int = 20,
) -> Dict:
    """
    Finds HOMO and LUMO energies using the Shift-Invert Lanczos algorithm.

    Args:
        H (torch.Tensor): The Hamiltonian matrix.
        num_electrons (int): Total number of electrons in the system.
        energy_shift_guess (torch.Tensor): A scalar tensor, the model's prediction
            for the energy at the midpoint of the HOMO-LUMO gap.
        k (int): The number of eigenvalues to find around the shift.
        num_iterations (int): The number of Lanczos iterations.

    Returns:
        A dictionary containing HOMO energy, LUMO energy, and the gap.
    """
    device, dtype = H.device, H.dtype
    n_basis = H.shape[0]

    # The operator for the linear system is A = (H - sigma * I)
    A = H - energy_shift_guess * torch.eye(n_basis, device=device, dtype=dtype)
    
    # --- Lanczos Iteration ---
    # We build a small tridiagonal matrix T
    T = torch.zeros(num_iterations, num_iterations, device=device, dtype=dtype)
    
    # Start with a random vector
    q = torch.randn(n_basis, device=device, dtype=dtype)
    q = q / torch.norm(q)
    
    # Store the basis vectors of the Krylov subspace
    Q = torch.zeros(n_basis, num_iterations, device=device, dtype=dtype)
    
    beta = 0.0
    for j in range(num_iterations):
        Q[:, j] = q
        
        # This is the key step: apply the shift-invert operator
        # Instead of inverting A, we solve the linear system A*w = q
        # This finds w = A_inv * q
        w = torch.linalg.solve(A, q)
        
        alpha = torch.dot(q, w)
        T[j, j] = alpha
        
        # Re-orthogonalize w against previous vectors (Full reorthogonalization for stability)
        w = w - Q[:, :j+1] @ (Q[:, :j+1].T @ w)
        
        beta = torch.norm(w)
        if beta < 1e-8 or j == num_iterations - 1:
            break
            
        T[j, j-1] = beta
        T[j-1, j] = beta
        
        q = w / beta

    # Truncate T if we broke early
    T = T[:j+1, :j+1]

    # --- Post-processing ---
    # Find eigenvalues (theta) of the small matrix T
    ritz_values_theta = torch.linalg.eigvalsh(T)
    
    # Convert them back to eigenvalues (lambda) of the original Hamiltonian H
    # theta = 1 / (lambda - sigma)  =>  lambda = sigma + 1 / theta
    eigenvalues_lambda = energy_shift_guess + 1.0 / ritz_values_theta
    
    eigenvalues_sorted = torch.sort(eigenvalues_lambda)[0]

    # Find HOMO and LUMO from the calculated eigenvalues
    num_occupied = num_electrons // 2
    
    # Find all eigenvalues below the shift guess (potential occupied)
    # and all above (potential virtual)
    occupied_candidates = eigenvalues_sorted[eigenvalues_sorted < energy_shift_guess]
    virtual_candidates = eigenvalues_sorted[eigenvalues_sorted >= energy_shift_guess]
    
    if len(occupied_candidates) == 0 or len(virtual_candidates) == 0:
        print("Warning: Iterative solver did not find eigenvalues on both sides of the gap.")
        # Fallback: take the two eigenvalues closest to the shift
        if len(eigenvalues_sorted) > 1:
             # This is not physically guaranteed but a reasonable fallback
            homo = eigenvalues_sorted[k//2 - 1]
            lumo = eigenvalues_sorted[k//2]
        else:
            return {'HOMO_energy': None, 'LUMO_energy': None, 'HOMO_LUMO_gap': None}
    else:
        homo = occupied_candidates[-1]
        lumo = virtual_candidates[0]
        
    gap = lumo - homo

    return {
        'HOMO_energy': homo,
        'LUMO_energy': lumo,
        'HOMO_LUMO_gap': gap,
        'found_eigenvalues': eigenvalues_sorted,
    }


if __name__ == '__main__':
    # --- Define System and Basis Set Here ---
    torch.set_default_dtype(torch.float64)
    
    # Define the molecule by a list of atomic numbers (Z)
    node_types = torch.tensor([8, 1, 1]) # Example: O-O-H
    N_nodes = len(node_types)

    basis_def = {
        1: {'1s': 1},
        8: {'1s': 1, '2s': 1, '2p': 1},
        # Add other elements as needed, e.g. 6: {'1s': 1, ...} for Carbon
    }
    node_irreps, edge_irreps, _, _ = get_irreps(basis_def)

    edge_index = torch.combinations(torch.arange(N_nodes), r=2).t()
    N_edges = edge_index.shape[1]
    
    data = {
        AtomicDataDict.NODE_TYPE_KEY: node_types,
        AtomicDataDict.NODE_FEATURES_KEY: torch.randn(N_nodes, node_irreps.dim),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.randn(N_edges, edge_irreps.dim),
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
    }

    compute_hamiltonian(data)