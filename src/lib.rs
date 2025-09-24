use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::SystemTime;

#[pyclass]
struct FFmpegCapture {
    child: std::process::Child,
    stdout: std::process::ChildStdout,
    width: usize,
    height: usize,
    frame_size: usize,
    frame_count: usize,
    start_time: SystemTime,
}

#[pymethods]
impl FFmpegCapture {
    /// Create a capture that pipes raw RGB24 frames from ffmpeg stdout.
    /// device: avfoundation device index as a string, e.g. "0"
    #[new]
    #[pyo3(signature = (device=None, width=None, height=None, framerate=None, pix_in=None))]
    fn new(
        device: Option<String>,
        width: Option<usize>,
        height: Option<usize>,
        framerate: Option<u32>,
        pix_in: Option<String>,
    ) -> PyResult<Self> {
        let device = device.unwrap_or_else(|| "0".to_string());
        let width = width.unwrap_or(640);
        let height = height.unwrap_or(480);
        let framerate = framerate.unwrap_or(5);
        let _pix_in = pix_in.unwrap_or_else(|| "uyvy422".to_string());

        // Spawn ffmpeg to capture raw frames
        let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner","-loglevel","error",
            "-f","avfoundation",
            "-framerate",&framerate.to_string(),
            "-video_size",&format!("{width}x{height}"),
            "-i",&device,
            "-f","rawvideo","-pix_fmt","rgb24","-"
        ])        
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to spawn ffmpeg: {e}")))?;

        let stdout = child.stdout.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Failed to capture ffmpeg stdout")
        })?;

        let frame_size = width * height * 3;
        let start_time = SystemTime::now();

        Ok(Self { 
            child, 
            stdout, 
            width, 
            height, 
            frame_size,
            frame_count: 0,
            start_time,
        })
    }

    /// Read one frame. Returns `bytes` of length width*height*3 (RGB), or None on EOF.
    fn next_frame<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let mut buf = vec![0u8; self.frame_size];
        let mut read_bytes = 0usize;

        while read_bytes < self.frame_size {
            match self.stdout.read(&mut buf[read_bytes..]) {
                Ok(0) => {
                    // EOF
                    if read_bytes == 0 {
                        return Ok(None);
                    } else {
                        return Err(pyo3::exceptions::PyEOFError::new_err(
                            "Unexpected EOF in middle of frame",
                        ));
                    }
                }
                Ok(n) => read_bytes += n,
                Err(e) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Read error: {e}"
                    )))
                }
            }
        }

        self.frame_count += 1;
        Ok(Some(PyBytes::new_bound(py, &buf)))
    }

    /// Get frame information including timestamp and frame count
    fn frame_info(&self) -> PyResult<(f64, usize)> {
        let elapsed = self.start_time.elapsed()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Time error: {e}")))?;
        let timestamp = elapsed.as_secs_f64();
        Ok((timestamp, self.frame_count))
    }

    /// Return (width, height)
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Cleanly terminate ffmpeg
    fn close(&mut self) -> PyResult<()> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        Ok(())
    }

    /// Context manager support: with FFmpegCapture(...) as cap:
    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __exit__(&mut self, _ty: PyObject, _val: PyObject, _tb: PyObject) -> PyResult<()> {
        self.close()
    }
}

/// Utility functions for frame processing
#[pyfunction]
fn frame_to_pil_image(frame_bytes: &[u8], width: usize, height: usize) -> PyResult<Vec<u8>> {
    if frame_bytes.len() != width * height * 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Expected frame size {} but got {}", width * height * 3, frame_bytes.len())
        ));
    }
    Ok(frame_bytes.to_vec())
}

/// Validate frame dimensions and format
#[pyfunction]
fn validate_frame(frame_bytes: &[u8], expected_width: usize, expected_height: usize) -> PyResult<bool> {
    let expected_size = expected_width * expected_height * 3;
    Ok(frame_bytes.len() == expected_size)
}

/// Get available camera devices (macOS specific)
#[pyfunction]
fn list_camera_devices() -> PyResult<Vec<String>> {
    let output = Command::new("ffmpeg")
        .args(["-f", "avfoundation", "-list_devices", "true", "-i", ""])
        .output()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to list devices: {e}")))?;
    
    let output_str = String::from_utf8_lossy(&output.stderr);
    let mut devices = Vec::new();
    
    for line in output_str.lines() {
        if line.contains("[AVFoundation indev @") && line.contains("] [") {
            // Extract device index and name
            if let Some(start) = line.find("] [") {
                if let Some(end) = line[start + 3..].find("]") {
                    let device_info = &line[start + 3..start + 3 + end];
                    if let Some(colon_pos) = device_info.find(": ") {
                        let index = &device_info[..colon_pos].trim();
                        if index.parse::<usize>().is_ok() {
                            devices.push(index.to_string());
                        }
                    }
                }
            }
        }
    }
    
    Ok(devices)
}

#[pymodule]
fn camstream(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<FFmpegCapture>()?;
    m.add_function(wrap_pyfunction!(frame_to_pil_image, m)?)?;
    m.add_function(wrap_pyfunction!(validate_frame, m)?)?;
    m.add_function(wrap_pyfunction!(list_camera_devices, m)?)?;
    Ok(())
}
