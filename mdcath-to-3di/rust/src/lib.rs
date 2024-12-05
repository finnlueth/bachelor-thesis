use numpy::PyReadonlyArray3;
use pyo3::prelude::*;
use std::fmt::Write;

#[pymodule]
fn rust_modules(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(replace_pdb_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bytestrings, m)?)?;
    m.add_function(wrap_pyfunction!(strings_to_fasta, m)?)?;
    Ok(())
}

#[pyfunction]
fn strings_to_fasta(strings: Vec<String>, name: String) -> PyResult<String> {
    let mut fasta = String::new();
    for (i, sequence) in strings.iter().enumerate() {
        writeln!(&mut fasta, ">{}|frame_{}", name, i).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to write header: {}", e)
            )
        })?;
        writeln!(&mut fasta, "{}", sequence.trim()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to write sequence: {}", e) 
            )
        })?;
    }
    Ok(fasta)
}


#[pyfunction]
fn decode_bytestrings(byte_strings: Vec<Vec<u8>>) -> PyResult<Vec<String>> {
    byte_strings
        .into_iter()
        .map(|bytes| {
            String::from_utf8(bytes)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyUnicodeDecodeError, _>(
                    format!("Failed to decode bytes to UTF-8: {}", e)
                ))
        })
        .collect()
}

#[pyfunction]
fn replace_pdb_coordinates(pdb_template: String, coordinates: PyReadonlyArray3<f32>) -> PyResult<Vec<String>> {
    let dims = coordinates.dims();
    let n_frames = dims[0];
    let n_atoms = dims[1];
    
    let line_count = pdb_template.lines().count();
    let estimated_capacity = line_count * 81;
    
    let mut atom_lines: Vec<(usize, &str)> = pdb_template
        .lines()
        .enumerate()
        .filter(|(_, line)| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .collect();
    
    if atom_lines.len() != n_atoms {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Number of coordinates ({}) doesn't match number of ATOM/HETATM records ({}) in PDB", 
                n_atoms, atom_lines.len())
        ));
    }
    
    let template_segments: Vec<&str> = pdb_template.lines().collect();
    
    let mut pdbs = Vec::with_capacity(n_frames);
    
    for frame_idx in 0..n_frames {
        let mut new_pdb = String::with_capacity(estimated_capacity);
        let mut current_line = 0;
        
        for (atom_count, &(line_idx, atom_line)) in atom_lines.iter().enumerate() {
            while current_line < line_idx {
                new_pdb.push_str(template_segments[current_line]);
                new_pdb.push('\n');
                current_line += 1;
            }
            
            let x = coordinates.get([frame_idx, atom_count, 0]).unwrap();
            let y = coordinates.get([frame_idx, atom_count, 1]).unwrap();
            let z = coordinates.get([frame_idx, atom_count, 2]).unwrap();
            
            write!(new_pdb, "{}{:8.3}{:8.3}{:8.3}{}\n",
                &atom_line[0..30],
                x, y, z,
                &atom_line[54..])
                .unwrap();
            
            current_line += 1;
        }
        
        while current_line < template_segments.len() {
            new_pdb.push_str(template_segments[current_line]);
            new_pdb.push('\n');
            current_line += 1;
        }
        
        if new_pdb.ends_with('\n') {
            new_pdb.pop();
        }
        
        pdbs.push(new_pdb);
    }
    Ok(pdbs)
}