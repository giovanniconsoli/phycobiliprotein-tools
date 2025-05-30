import pandas as pd
from Bio.PDB import PDBParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse # For command-line arguments
import os # For path manipulations

# --- Specific Recentering Function ---
def recenter_angle_0_90(angle_value): # Renamed for simplicity
    """
    Recenters an angle: if < 90, remains unchanged; if >= 90, becomes 180 - angle_value.
    Result is in the range [0, 90].
    """
    if angle_value < 90:
        return angle_value
    else:
        return 180 - angle_value

# --- Utility Functions for PDB Data Extraction and Calculations ---
def get_atom_coordinates(structure, chain_id, residue_id_tuple, atom_name):
    """Retrieves the 3D coordinates of a specific atom."""
    try:
        model = structure[0]
        chain = model[chain_id]
        residue = chain[residue_id_tuple] 
        atom = residue[atom_name]
        return atom.get_coord()
    except KeyError as e:
        print(f"Error retrieving atom: Chain {chain_id}, Residue ID {residue_id_tuple}, Atom {atom_name}. Biopython error: {e}")
        raise

def calculate_angle_between_planes(structure, atom_ids_plane1, atom_ids_plane2):
    """Calculates the angle between two planes, each defined by three atoms."""
    coords1 = [get_atom_coordinates(structure, *atom_id) for atom_id in atom_ids_plane1]
    coords2 = [get_atom_coordinates(structure, *atom_id) for atom_id in atom_ids_plane2]

    vec1_plane1 = coords1[1] - coords1[0] # Simplified from v1_plane1
    vec2_plane1 = coords1[2] - coords1[0] # Simplified from v2_plane1
    vec1_plane2 = coords2[1] - coords2[0] # Simplified from v1_plane2
    vec2_plane2 = coords2[2] - coords2[0] # Simplified from v2_plane2

    normal_vec1 = np.cross(vec1_plane1, vec2_plane1) # Simplified from normal1
    normal_vec2 = np.cross(vec1_plane2, vec2_plane2) # Simplified from normal2

    dot_product = np.dot(normal_vec1, normal_vec2)
    norm_product = np.linalg.norm(normal_vec1) * np.linalg.norm(normal_vec2) # Simplified from norm_prod
    
    if abs(norm_product) < 1e-9:
        print(f"Warning: Product of norms is close to zero for planes defined by {atom_ids_plane1} and {atom_ids_plane2}. Indicates a degenerate plane.")
        return 0.0 

    cos_theta = dot_product / norm_product # Simplified from cos_angle
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def find_cyc_instances(structure):
    """Finds all instances of CYC residues in the PDB structure."""
    cyc_residues = [] # Renamed from cyc_instances for clarity
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "CYC":
                    cyc_residues.append((chain.id, residue.id)) 
    return cyc_residues

# --- Data Processing and Calculation Functions ---
def calculate_and_process_angles(structure, cyc_residues): # Parameter name updated
    """
    Calculates initial angles and then recenters them using recenter_angle_0_90.
    """
    angle_data_list = [] # Renamed from calculated_data for clarity
    for chain_id, residue_id_tuple in cyc_residues: # Unpacking directly, var name updated
        residue_num = residue_id_tuple[1] 
        
        # Define atom sets for plane calculations
        atoms_plane_A1 = [(chain_id, residue_id_tuple, 'NA'), (chain_id, residue_id_tuple, 'C3A'), (chain_id, residue_id_tuple, 'C4A')] # Using A, B, C for sets of planes
        atoms_plane_A2 = [(chain_id, residue_id_tuple, 'NB'), (chain_id, residue_id_tuple, 'C1B'), (chain_id, residue_id_tuple, 'C2B')]
        
        atoms_plane_B1 = [(chain_id, residue_id_tuple, 'ND'), (chain_id, residue_id_tuple, 'C4D'), (chain_id, residue_id_tuple, 'C3D')]
        atoms_plane_B2 = [(chain_id, residue_id_tuple, 'NA'), (chain_id, residue_id_tuple, 'C1A'), (chain_id, residue_id_tuple, 'C2A')]
        
        atoms_plane_C1 = [(chain_id, residue_id_tuple, 'NC'), (chain_id, residue_id_tuple, 'C3C'), (chain_id, residue_id_tuple, 'C4C')]
        atoms_plane_C2 = [(chain_id, residue_id_tuple, 'ND'), (chain_id, residue_id_tuple, 'C2D'), (chain_id, residue_id_tuple, 'C1D')]

        try:
            # 1. Calculate initial angles (these local vars are simpler)
            angle_AB_raw = calculate_angle_between_planes(structure, atoms_plane_A1, atoms_plane_A2)
            angle_BC_raw = calculate_angle_between_planes(structure, atoms_plane_B1, atoms_plane_B2) # Assuming these map to BC, CD for export
            angle_CD_raw = calculate_angle_between_planes(structure, atoms_plane_C1, atoms_plane_C2)
            
            # 2. Apply the recentering function to the initial angles
            recentered_angle_AB = recenter_angle_0_90(angle_AB_raw) # Simpler local var name
            recentered_angle_BC = recenter_angle_0_90(angle_BC_raw)
            recentered_angle_CD = recenter_angle_0_90(angle_CD_raw)

            angle_data_list.append({
                'Chain_ID': chain_id, 
                'Residue_Number': residue_num,
                'Angle1_original': angle_AB_raw, # DataFrame column name remains descriptive
                'Angle2_original': angle_BC_raw,
                'Angle3_original': angle_CD_raw,
                'Angle1_recentered': recentered_angle_AB, # DataFrame column name remains descriptive
                'Angle2_recentered': recentered_angle_BC, 
                'Angle3_recentered': recentered_angle_CD  
            })
        except KeyError:
            print(f"Skipping CYC instance {chain_id}{residue_id_tuple} due to missing atoms or other key errors.")
        except Exception as e:
            print(f"An unexpected error occurred for CYC {chain_id}{residue_id_tuple}: {e}")
    return pd.DataFrame(angle_data_list)

# --- Output and Plotting Functions ---
def export_data_to_excel(data_frame, output_filename): # Parameter name changed
    """Exports selected angle data to an Excel file."""
    if data_frame.empty:
        print("DataFrame is empty. Nothing to export to Excel.")
        return
    # Column names AngleX_recentered are used here as they are in the DataFrame
    export_df = data_frame[['Chain_ID', 'Angle1_recentered', 'Angle2_recentered', 'Angle3_recentered']].copy()
    export_df.rename(columns={
        'Angle1_recentered': 'angle_AB', # These are the final output column names in Excel
        'Angle2_recentered': 'angle_BC', 
        'Angle3_recentered': 'angle_CD'
    }, inplace=True)
    try:
        export_df.to_excel(output_filename, index=False)
        print(f"Data exported successfully to {output_filename}")
    except Exception as e:
        print(f"Error exporting data to Excel: {e}")

def plot_angle_scatter(data_frame, x_axis_col, y_axis_col, output_filename): # Parameter names changed
    """
    Creates a 2D scatter plot. Uses fixed axis limits (0,60) as recentered data is in [0,90].
    """
    if data_frame.empty:
        print("DataFrame is empty. Nothing to plot.")
        return
    if not all(col in data_frame.columns for col in [x_axis_col, y_axis_col, 'Chain_ID']):
        print(f"Error: One or more required columns for plotting not found in DataFrame.")
        return

    num_points = len(data_frame)
    if num_points == 0:
        print("No data points to plot.")
        return
        
    colormap = plt.cm.viridis 
    point_colors = [colormap(i) for i in np.linspace(0, 0.9, num_points)] # Renamed for clarity

    fig, ax = plt.subplots(figsize=(10, 8))

    fixed_xlim = (0, 60)
    fixed_ylim = (0, 60)
    ax.set_xlim(fixed_xlim)
    ax.set_ylim(fixed_ylim)

    view_x_span = fixed_xlim[1] - fixed_xlim[0] # Renamed for clarity
    view_y_span = fixed_ylim[1] - fixed_ylim[0] # Renamed for clarity
    
    text_offset_x = view_x_span * 0.015 
    text_offset_y = view_y_span * 0.015 
    
    for i in range(num_points):
        chain_id = data_frame['Chain_ID'].iloc[i]
        x_value = data_frame[x_axis_col].iloc[i] # Renamed for clarity
        y_value = data_frame[y_axis_col].iloc[i] # Renamed for clarity
        color = point_colors[i] # Renamed for clarity

        ax.scatter(x_value, y_value, color=color, marker='o', alpha=0.8, s=100)
        ax.text(x_value + text_offset_x, 
                y_value + text_offset_y,
                str(chain_id), 
                fontsize=8, ha='left', va='bottom')

    ax.set_xlabel(f"Angle between plane A and B ({x_axis_col}) [degrees]") # Using generic A-B, C-D
    ax.set_ylabel(f"Angle between plane C and D ({y_axis_col}) [degrees]") # to match Excel output
    ax.set_title('Recentered Dihedral Angles') # Simplified title
    ax.grid(True, linestyle='--', alpha=0.7)
    
    try:
        fig.tight_layout() 
    except UserWarning as e:
        print(f"UserWarning during fig.tight_layout(): {e}. Consider adjusting figsize or fontsize.")
    
    try:
        fig.savefig(output_filename, bbox_inches='tight')
        print(f"Plot saved successfully to {output_filename}")
        plt.show()
    except Exception as e:
        print(f"Error saving plot: {e}")

# --- Main Execution ---
def main():
    """
    Main function to parse command-line arguments, run PDB parsing, angle calculation,
    data export, and plotting.
    """
    cli_parser = argparse.ArgumentParser( # Renamed for clarity
        description="Analyzes dihedral angles in CYC residues from PDB files, generates plots, and exports data to Excel."
    )
    cli_parser.add_argument(
        "pdb_filepath",  # Renamed for clarity
        type=str,    
        help="Path to the input PDB file." 
    )
    args = cli_parser.parse_args()
    
    input_pdb_path = args.pdb_filepath # Using the renamed argument

    pdb_filename_full = os.path.basename(input_pdb_path) 
    pdb_filename_base, _ = os.path.splitext(pdb_filename_full) 

    excel_output_path = f"{pdb_filename_base}_angles.xlsx" # Using renamed var
    plot_output_path = f"{pdb_filename_base}_plot.svg"   # Using renamed var

    print(f"Starting analysis for PDB file: {input_pdb_path}")
    print(f"Excel output will be saved to: ./{excel_output_path}")
    print(f"Plot output will be saved to: ./{plot_output_path}")

    biopython_parser = PDBParser(QUIET=True) # Renamed for clarity
    try:
        protein_structure = biopython_parser.get_structure('protein_structure', input_pdb_path) # Renamed for clarity
    except FileNotFoundError:
        print(f"Error: PDB file not found at {input_pdb_path}")
        return
    except Exception as e:
        print(f"Error parsing PDB file '{input_pdb_path}': {e}")
        return

    cyc_residues_found = find_cyc_instances(protein_structure) # Renamed for clarity
    if not cyc_residues_found:
        print("No CYC residues found in the structure.")
    else:
        print(f"Found {len(cyc_residues_found)} CYC instance(s).")

    # The function now applies the [0,90] recentering internally
    angles_dataframe = calculate_and_process_angles(protein_structure, cyc_residues_found) # Renamed for clarity

    if angles_dataframe.empty:
        if cyc_residues_found:
            print("Angle calculation resulted in an empty DataFrame. Check for atom errors above.")
        else:
             print("No CYC data to process.")
    else:
        print("\nCalculated Angles DataFrame (first few rows):")
        print(angles_dataframe[['Chain_ID', 'Residue_Number', 'Angle1_original', 'Angle1_recentered', 'Angle3_original', 'Angle3_recentered']].head())

    if not angles_dataframe.empty:
        export_data_to_excel(angles_dataframe, excel_output_path)
        plot_angle_scatter(angles_dataframe, 
                           x_axis_col='Angle1_recentered', 
                           y_axis_col='Angle3_recentered', 
                           output_filename=plot_output_path)
    else:
        print("Skipping export and plotting as there is no data.")
        
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()