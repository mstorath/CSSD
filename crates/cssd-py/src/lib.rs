//! PyO3 bindings for cssd-core.
//!
//! Exposes a minimal extension module `_cssd_core` consumed by the Python
//! package `cssd`. Heavy lifting (input validation, output construction) lives
//! on the Python side; this layer is a thin numpy <-> Rust adapter.
#![allow(clippy::too_many_arguments, clippy::useless_conversion)]

use cssd_core::{cssd as core_cssd, cssd_cvscore as core_cvscore, Preconditioning, Pruning};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn parse_pruning(s: &str) -> PyResult<Pruning> {
    match s {
        "FPVI" | "fpvi" => Ok(Pruning::Fpvi),
        "PELT" | "pelt" => Ok(Pruning::Pelt),
        other => Err(PyValueError::new_err(format!(
            "unknown pruning '{other}', expected 'FPVI' or 'PELT'"
        ))),
    }
}

fn parse_precondition(s: &str) -> PyResult<Preconditioning> {
    Preconditioning::parse(s).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown precondition '{s}', expected 'none' or 'local'"
        ))
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, p, gamma, delta=None, pruning="FPVI", precondition="none"))]
fn cssd<'py>(
    py: Python<'py>,
    x: Option<PyReadonlyArray1<'py, f64>>,
    y: PyReadonlyArray2<'py, f64>,
    p: f64,
    gamma: f64,
    delta: Option<PyReadonlyArray1<'py, f64>>,
    pruning: &str,
    precondition: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let pruning = parse_pruning(pruning)?;
    let precondition = parse_precondition(precondition)?;
    let y_arr = y.as_array().to_owned();
    let x_owned = x.as_ref().map(|a| a.as_array().to_owned());
    let delta_owned = delta.as_ref().map(|a| a.as_array().to_owned());

    let out = core_cssd(
        x_owned.as_ref().map(|a| a.view()),
        y_arr.view(),
        p,
        gamma,
        delta_owned.as_ref().map(|a| a.view()),
        pruning,
        precondition,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("breaks", out.pp.breaks.into_pyarray_bound(py))?;
    dict.set_item("coefs", out.pp.coefs.into_pyarray_bound(py))?;
    dict.set_item("dim", out.pp.dim)?;
    dict.set_item("order", out.pp.order)?;
    dict.set_item("discont", out.discont.into_pyarray_bound(py))?;
    let discont_idx_i64: Vec<i64> = out.discont_idx.iter().map(|&i| i as i64).collect();
    dict.set_item(
        "discont_idx",
        ndarray::Array1::from_vec(discont_idx_i64).into_pyarray_bound(py),
    )?;
    dict.set_item("x", out.x.into_pyarray_bound(py))?;
    dict.set_item("y", out.y.into_pyarray_bound(py))?;
    dict.set_item("complexity_counter", out.complexity_counter as u64)?;

    let intervals = PyList::empty_bound(py);
    for iv in &out.interval_cell {
        let v: Vec<i64> = iv.iter().map(|&i| i as i64).collect();
        intervals.append(ndarray::Array1::from_vec(v).into_pyarray_bound(py))?;
    }
    dict.set_item("interval_cell", intervals)?;

    let pps = PyList::empty_bound(py);
    for pp in &out.pp_cell {
        let d = PyDict::new_bound(py);
        d.set_item("breaks", pp.breaks.clone().into_pyarray_bound(py))?;
        d.set_item("coefs", pp.coefs.clone().into_pyarray_bound(py))?;
        d.set_item("dim", pp.dim)?;
        d.set_item("order", pp.order)?;
        pps.append(d)?;
    }
    dict.set_item("pp_cell", pps)?;

    dict.set_item("F", ndarray::Array1::from_vec(out.f).into_pyarray_bound(py))?;
    let part_i64: Vec<i64> = out.partition.iter().map(|&i| i as i64).collect();
    dict.set_item(
        "partition",
        ndarray::Array1::from_vec(part_i64).into_pyarray_bound(py),
    )?;

    let precondition_str = match out.precondition {
        Preconditioning::None => "none",
        Preconditioning::Local => "local",
    };
    dict.set_item("precondition", precondition_str)?;
    dict.set_item("tau", out.tau.into_pyarray_bound(py))?;

    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (x, y, p, gamma, delta, folds, pruning="FPVI", precondition="none"))]
fn cssd_cvscore<'py>(
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    p: f64,
    gamma: f64,
    delta: PyReadonlyArray1<'py, f64>,
    folds: Vec<Vec<i64>>,
    pruning: &str,
    precondition: &str,
) -> PyResult<f64> {
    let pruning = parse_pruning(pruning)?;
    let precondition = parse_precondition(precondition)?;
    let folds: Vec<Vec<usize>> = folds
        .into_iter()
        .map(|f| f.into_iter().map(|i| i as usize).collect())
        .collect();
    Ok(core_cvscore(
        x.as_array(),
        y.as_array(),
        p,
        gamma,
        delta.as_array(),
        &folds,
        pruning,
        precondition,
    ))
}

#[pymodule]
fn _cssd_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cssd, m)?)?;
    m.add_function(wrap_pyfunction!(cssd_cvscore, m)?)?;
    Ok(())
}
