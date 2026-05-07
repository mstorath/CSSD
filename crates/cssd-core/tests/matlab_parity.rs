//! Live Rust↔MATLAB parity test for `cssd_core::cssd()`.
//!
//! Each test generates a small input, calls the Rust port in-process, then shells
//! out to host MATLAB via the `matlab` shim (`/usr/local/bin/matlab`, which the
//! dev container's post-create.sh installs as a SSH proxy to host MATLAB).
//! MATLAB writes the resulting `pp.coefs`, `pp.breaks`, and `discont_idx` to CSVs;
//! the test reads them and compares.
//!
//! Tolerance: `atol=1e-8 / rtol=1e-5` on `pp.coefs`. This matches the
//! tolerance the existing Python harness (`tests_py/test_matlab_parity.py`)
//! uses against the 594 `.mat` fixtures, and reflects the empirical gap
//! between MATLAB's LAPACK-backed QR/banded solves and Rust's nalgebra/ndarray
//! equivalents on the spline-system normal equations. Tightening below this
//! is empirically not feasible without algorithmic changes.
//!
//! Skipped (not failed) when the matlab shim is unavailable.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use approx::assert_relative_eq;
use cssd_core::{cssd, Preconditioning, Pruning};
use ndarray::{Array1, Array2};

fn shim_available() -> bool {
    env::var("HOST_MATLAB").is_ok() && std::path::Path::new("/usr/local/bin/matlab").exists()
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../CSSD/crates/cssd-core
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root resolves")
}

fn workspace_root() -> PathBuf {
    // .../devcontainer/10-OwnRepos/CSSD -> .../devcontainer
    repo_root().join("../..").canonicalize().expect("workspace root resolves")
}

fn make_workspace_tempdir() -> PathBuf {
    let mut path;
    loop {
        let suffix: u64 = rand_u64();
        path = workspace_root().join(format!("cssd-parity-{suffix:x}"));
        if !path.exists() {
            fs::create_dir(&path).expect("create cssd-parity tempdir");
            break;
        }
    }
    path
}

fn rand_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    nanos ^ std::process::id() as u64
}

fn read_csv_2d(path: &std::path::Path) -> Array2<f64> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|s| s.trim().parse::<f64>().unwrap_or_else(|_| panic!("parse {s:?}")))
                .collect()
        })
        .collect();
    let nrows = rows.len();
    let ncols = rows.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((nrows, ncols), flat).expect("rectangular CSV")
}

fn read_csv_1d(path: &std::path::Path) -> Array1<f64> {
    let arr = read_csv_2d(path);
    Array1::from_iter(arr.iter().copied())
}

struct MatlabCssdResult {
    breaks: Array1<f64>,
    coefs: Array2<f64>,
    discont_idx: Array1<usize>,
}

fn run_matlab_cssd(
    x: &Array1<f64>,
    y: &Array2<f64>,
    p: f64,
    gamma: f64,
    pruning: &str,
) -> MatlabCssdResult {
    let work = make_workspace_tempdir();
    let coefs_path = work.join("coefs.csv");
    let breaks_path = work.join("breaks.csv");
    let didx_path = work.join("discont_idx.csv");
    let script_path = work.join("run_parity.m");

    let x_lit: String = x.iter().map(|v| format!("{v:.17e}")).collect::<Vec<_>>().join("; ");
    let n = y.nrows();
    let d = y.ncols();
    let mut y_lit = String::new();
    for i in 0..n {
        if i > 0 {
            y_lit.push_str("; ");
        }
        for j in 0..d {
            if j > 0 {
                y_lit.push_str(", ");
            }
            y_lit.push_str(&format!("{:.17e}", y[(i, j)]));
        }
    }

    let cssd_repo = repo_root();
    let script = format!(
        "addpath(genpath('{cssd_repo}'));\n\
         x = [{x_lit}];\n\
         y = [{y_lit}];\n\
         p = {p:.17e};\n\
         gamma = {gamma:.17e};\n\
         out = cssd(x, y, p, gamma, [], [], 'Pruning', '{pruning}');\n\
         writematrix(out.pp.coefs, '{coefs_csv}');\n\
         writematrix(out.pp.breaks(:), '{breaks_csv}');\n\
         if isempty(out.discont_idx); didx = zeros(0,1); else; didx = out.discont_idx(:); end;\n\
         writematrix(didx, '{didx_csv}');\n",
        cssd_repo = cssd_repo.display(),
        coefs_csv = coefs_path.display(),
        breaks_csv = breaks_path.display(),
        didx_csv = didx_path.display(),
    );
    fs::write(&script_path, script).expect("write .m");

    let status = Command::new("matlab")
        .arg("-batch")
        .arg(format!("addpath('{}'); run_parity", work.display()))
        .output()
        .expect("invoke matlab shim");
    if !status.status.success() {
        let stdout = String::from_utf8_lossy(&status.stdout);
        let stderr = String::from_utf8_lossy(&status.stderr);
        panic!("MATLAB cssd failed (rc={:?}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}", status.status.code());
    }

    let coefs = read_csv_2d(&coefs_path);
    let breaks = read_csv_1d(&breaks_path);
    let didx_arr = read_csv_2d(&didx_path);
    let discont_idx: Array1<usize> = if didx_arr.is_empty() {
        Array1::from_vec(vec![])
    } else {
        Array1::from_iter(didx_arr.iter().map(|v| (*v as usize).saturating_sub(1)))
    };

    let _ = fs::remove_dir_all(&work);
    MatlabCssdResult { breaks, coefs, discont_idx }
}

fn assert_coefs_close(rust: &Array2<f64>, matlab: &Array2<f64>) {
    assert_eq!(
        rust.shape(),
        matlab.shape(),
        "coefs shape mismatch: rust {:?} vs matlab {:?}",
        rust.shape(),
        matlab.shape()
    );
    for ((i, j), &r) in rust.indexed_iter() {
        let m = matlab[(i, j)];
        let abs = (r - m).abs();
        let denom = m.abs().max(1.0);
        let rel = abs / denom;
        assert!(
            abs <= 1e-8 || rel <= 1e-5,
            "coefs[{i},{j}]: rust={r:.6e} matlab={m:.6e} abs={abs:.3e} rel={rel:.3e}"
        );
    }
}

fn run_parity_case(
    name: &str,
    x: Array1<f64>,
    y: Array2<f64>,
    p: f64,
    gamma: f64,
    pruning_rust: Pruning,
    pruning_matlab: &str,
) {
    if !shim_available() {
        eprintln!("[{name}] skipping: matlab shim not configured");
        return;
    }
    let out_rust =
        cssd(Some(x.view()), y.view(), p, gamma, None, pruning_rust, Preconditioning::None)
            .expect("rust cssd ok");
    let out_matlab = run_matlab_cssd(&x, &y, p, gamma, pruning_matlab);

    assert_relative_eq!(
        out_rust.pp.breaks.as_slice().unwrap(),
        out_matlab.breaks.as_slice().unwrap(),
        epsilon = 1e-10
    );
    assert_coefs_close(&out_rust.pp.coefs, &out_matlab.coefs);
    assert_eq!(
        out_rust.discont_idx.as_slice().unwrap(),
        out_matlab.discont_idx.as_slice().unwrap(),
        "discont_idx mismatch"
    );
}

#[test]
fn parity_step_p09_g1_fpvi() {
    let n = 12;
    let x = Array1::from_iter((1..=n).map(|i| i as f64));
    let mut y_vec = vec![0.0; n / 2];
    y_vec.extend(vec![1.0; n / 2]);
    let y = Array2::from_shape_vec((n, 1), y_vec).unwrap();
    run_parity_case("step_p09_g1_fpvi", x, y, 0.9, 1.0, Pruning::Fpvi, "FPVI");
}

#[test]
fn parity_step_p05_g05_pelt() {
    let n = 16;
    let x = Array1::from_iter((1..=n).map(|i| i as f64));
    let mut y_vec = vec![0.0; n / 2];
    y_vec.extend(vec![2.0; n / 2]);
    let y = Array2::from_shape_vec((n, 1), y_vec).unwrap();
    run_parity_case("step_p05_g05_pelt", x, y, 0.5, 0.5, Pruning::Pelt, "PELT");
}

#[test]
fn parity_smooth_p099_glarge() {
    // Large gamma -> classical smoothing spline (no discontinuities).
    let n = 10;
    let x = Array1::from_iter((1..=n).map(|i| i as f64));
    let y_vec: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            t * t
        })
        .collect();
    let y = Array2::from_shape_vec((n, 1), y_vec).unwrap();
    run_parity_case("smooth_p099_glarge", x, y, 0.99, 1e8, Pruning::Fpvi, "FPVI");
}
