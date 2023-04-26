use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::types::{PyString, PyList};
use std::path::Path;


#[pymodule]
fn read_layers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    fn read_layers<'py>(py: Python<'py>, folder: &PyString) -> &'py PyArray2<f64> {
        rust_fn::read_layers(folder.to_str().unwrap()).into_pyarray(py)
    }

    #[pyfn(m)]
    fn read_selected_layers<'py>(py: Python<'py>, file_list: &PyList) -> &'py PyArray2<f64> {
        rust_fn::read_selected_layers(
            file_list.iter().map(|x| Path::new(&(*x).str().unwrap().to_string()).to_path_buf()).collect()
    ).into_pyarray(py)
    }

    #[pyfn(m)]
    fn read_layer<'py>(py: Python<'py>, file: &PyString) -> &'py PyArray2<f64> {
        rust_fn::read_layer(file.to_str().unwrap()).into_pyarray(py)
    }

    Ok(())
}


mod rust_fn {
    use rayon::prelude::*;
    use glob::{glob, GlobError};
    use std::path::{Path, PathBuf};
    use std::error::Error;
    use std::fs::File;
    use csv::{ReaderBuilder};
    use ndarray::{Array2, ArrayView2, concatenate, Axis, Slice};
    use ndarray_csv::Array2Reader;
    use indicatif::ProgressBar;

    pub fn read_layers(folder: &str) -> Array2<f64> {
        let glob_string: String = folder.to_owned() + "/*.pcd";
        let mut glob_iterator: Vec<PathBuf> = glob(glob_string.as_str()).expect("Files not found!")
                                                                        .collect::<Result<Vec<PathBuf>, GlobError>>()
                                                                        .unwrap();
        glob_iterator.par_sort_unstable_by(|a, b| get_z(a).partial_cmp(&get_z(b)).unwrap());
        let len: usize = glob_iterator.len();
        let bar = ProgressBar::new(len as u64);
        let mut arrays: Vec<Array2<f64>> = vec![Array2::<f64>::zeros((0, 0)); len];
        let mut z_vals: Vec<f64> = vec![0.; len];
        let mut z_lens: Vec<usize> = vec![0; len];
        glob_iterator.par_iter()
                     .zip(arrays.par_iter_mut())
                     .zip(z_vals.par_iter_mut())
                     .zip(z_lens.par_iter_mut())
                     .for_each(|(((filepath, array_element), z_vals_element), z_lens_element)| {
            let (array, z, z_len) = read_file(filepath.to_path_buf()).unwrap();
            *array_element = array;
            *z_vals_element = z;
            *z_lens_element = z_len;
            bar.inc(1)
        });

        let mut padding_arrays: Vec<Array2<f64>> = Vec::<Array2<f64>>::new();
        for (z, z_len) in z_vals.iter().zip(z_lens) {
            let z_array: Array2::<f64> = Array2::from_elem((z_len, 1), *z);
            padding_arrays.push(z_array);
        }

        let padding_array_views: Vec<ArrayView2<f64>> = padding_arrays.iter()
                                                                      .map(|x| x.view())
                                                                      .collect();
        let array_views: Vec<ArrayView2<f64>> = arrays.iter()
                                                      .map(|x| x.view())
                                                      .collect();

        let mut out_array = concatenate(
            Axis(1),
            &[ concatenate(Axis(0), &array_views).unwrap().slice_axis(Axis(1), Slice::from(0..2)),
               concatenate(Axis(0), &padding_array_views).unwrap().view(),
               concatenate(Axis(0), &array_views).unwrap().slice_axis(Axis(1), Slice::from(2..4)) ]
        ).unwrap();

        out_array.column_mut(0).par_map_inplace(correct_x);
        out_array.column_mut(1).par_map_inplace(correct_y);

        out_array
    }

    pub fn read_selected_layers(file_list: Vec<PathBuf>) -> Array2<f64> {
        let len: usize = file_list.len();
        let bar = ProgressBar::new(len as u64);
        let mut arrays: Vec<Array2<f64>> = vec![Array2::<f64>::zeros((0, 0)); len];
        let mut z_vals: Vec<f64> = vec![0.; len];
        let mut z_lens: Vec<usize> = vec![0; len];
        file_list.par_iter()
                     .zip(arrays.par_iter_mut())
                     .zip(z_vals.par_iter_mut())
                     .zip(z_lens.par_iter_mut())
                     .for_each(|(((filepath, array_element), z_vals_element), z_lens_element)| {
            let (array, z, z_len) = read_file(filepath.to_path_buf()).unwrap();
            *array_element = array;
            *z_vals_element = z;
            *z_lens_element = z_len;
            bar.inc(1)
        });

        let mut padding_arrays: Vec<Array2<f64>> = Vec::<Array2<f64>>::new();
        for (z, z_len) in z_vals.iter().zip(z_lens) {
            let z_array: Array2::<f64> = Array2::from_elem((z_len, 1), *z);
            padding_arrays.push(z_array);
        }

        let padding_array_views: Vec<ArrayView2<f64>> = padding_arrays.iter()
                                                                      .map(|x| x.view())
                                                                      .collect();
        let array_views: Vec<ArrayView2<f64>> = arrays.iter()
                                                      .map(|x| x.view())
                                                      .collect();

        let mut out_array = concatenate(
            Axis(1),
            &[ concatenate(Axis(0), &array_views).unwrap().slice_axis(Axis(1), Slice::from(0..2)),
               concatenate(Axis(0), &padding_array_views).unwrap().view(),
               concatenate(Axis(0), &array_views).unwrap().slice_axis(Axis(1), Slice::from(2..4)) ]
        ).unwrap();

        out_array.column_mut(0).par_map_inplace(correct_x);
        out_array.column_mut(1).par_map_inplace(correct_y);

        out_array
    }

    pub fn read_layer(file: &str) -> Array2<f64> {
        let (array, z, z_len) = read_file(Path::new(file).to_path_buf()).unwrap();
        let z_array: Array2::<f64> = Array2::from_elem((z_len, 1), z);
        let z_array_view: ArrayView2<f64> = z_array.view();
        let array_view: ArrayView2<f64> = array.view();

        let mut out_array = concatenate(
            Axis(1),
            &[ array_view,
               z_array_view ]
        ).unwrap();

        out_array.column_mut(0).par_map_inplace(correct_x);
        out_array.column_mut(1).par_map_inplace(correct_y);

        out_array
    }

    fn read_file(filepath: PathBuf) -> Result<(Array2<f64>, f64, usize), Box<dyn Error>> {
        let z: f64 = get_z(&filepath);
        let file = File::open(filepath)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b' ')
            .from_reader(file);
        let array_read: Array2<f64> = rdr.deserialize_array2_dynamic()?;
        let z_len: usize = array_read.shape()[0];
        
        Ok((array_read, z, z_len))
    }

    fn get_z(filepath: &PathBuf) -> f64 {
        filepath.file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .parse::<f64>()
                .unwrap()
    }

    fn correct_x(x: &mut f64) -> () {
        *x = - ((((*x + 16384.) * 0.009155273) - 87.) / 1.01);
    }

    fn correct_y(y: &mut f64) -> () {
        *y = (((*y + 16384.) * 0.009155273) - 91.) / 1.02;
    }
}
