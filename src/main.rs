use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::SystemTime;

/// Rust-driven camera capture that calls Python FastVLM
struct CameraAnalyzer {
    python_module: PyObject,
    ffmpeg_child: std::process::Child,
    ffmpeg_stdout: std::process::ChildStdout,
    width: usize,
    height: usize,
    frame_size: usize,
    frame_count: usize,
    start_time: SystemTime,
}

impl CameraAnalyzer {
    fn new(
        py: Python,
        device: &str,
        width: usize,
        height: usize,
        framerate: u32,
    ) -> PyResult<Self> {
        // Add current directory to Python path
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", ("/Users/nestor/armando_new/camstream",))?;
        
        // Print Python executable info
        let executable = sys.getattr("executable")?;
        let executable_str: String = executable.extract()?;
        println!("🐍 Using Python executable: {}", executable_str);
        
        println!("🔍 Attempting to import fastvlm_analyzer...");
        
        // Import the Python FastVLM module
        let fastvlm_module = match py.import_bound("fastvlm_analyzer") {
            Ok(module) => {
                println!("✅ fastvlm_analyzer imported successfully");
                module
            }
            Err(e) => {
                println!("❌ Failed to import fastvlm_analyzer: {}", e);
                return Err(e);
            }
        };
        
        // Spawn ffmpeg to capture raw frames
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner", "-loglevel", "error",
                "-f", "avfoundation",
                "-framerate", &framerate.to_string(),
                "-video_size", &format!("{width}x{height}"),
                "-i", device,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
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
            python_module: fastvlm_module.into(),
            ffmpeg_child: child,
            ffmpeg_stdout: stdout,
            width,
            height,
            frame_size,
            frame_count: 0,
            start_time,
        })
    }

    fn next_frame(&mut self, _py: Python) -> PyResult<Option<Vec<u8>>> {
        let mut buf = vec![0u8; self.frame_size];
        let mut read_bytes = 0usize;

        while read_bytes < self.frame_size {
            match self.ffmpeg_stdout.read(&mut buf[read_bytes..]) {
                Ok(0) => {
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
        Ok(Some(buf))
    }

    fn analyze_frame(&self, py: Python, frame_data: &[u8]) -> PyResult<String> {
        // Convert frame data to Python bytes
        let frame_bytes = PyBytes::new_bound(py, frame_data);
        
        // Call the Python FastVLM analyzer
        let result = self.python_module.call_method1(py, "analyze_frame", (frame_bytes, self.width, self.height))?;
        
        // Extract the result string
        let result_str: String = result.extract(py)?;
        Ok(result_str)
    }

    fn get_timing_info(&self) -> PyResult<(f64, usize)> {
        let elapsed = self.start_time.elapsed()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Time error: {e}")))?;
        let timestamp = elapsed.as_secs_f64();
        Ok((timestamp, self.frame_count))
    }

    fn cleanup(&mut self) -> PyResult<()> {
        let _ = self.ffmpeg_child.kill();
        let _ = self.ffmpeg_child.wait();
        Ok(())
    }
}

fn main() -> PyResult<()> {
    println!("🚀 Rust-driven Camera Stream Analyzer");
    println!("=====================================");

    // Initialize Python with virtual environment
    Python::with_gil(|py| {
        // Create the camera analyzer
        let mut analyzer = CameraAnalyzer::new(py, "0", 640, 480, 30)?;
        
        println!("✅ Camera analyzer initialized");
        println!("📹 Settings: 640x480 @ 30 FPS");
        println!("🤖 FastVLM model loaded");
        println!("Press Ctrl+C to stop gracefully\n");

        let mut processed_frames = 0;
        let frame_skip = 15; // Process every 15th frame (2 FPS analysis)
        let mut total_frames = 0;

        loop {
            // Read frame
            match analyzer.next_frame(py)? {
                Some(frame_data) => {
                    total_frames += 1;
                    
                    // Skip frames for processing efficiency
                    if total_frames % frame_skip != 0 {
                        continue;
                    }

                    // Analyze frame
                    match analyzer.analyze_frame(py, &frame_data) {
                        Ok(analysis) => {
                            let (timestamp, _frame_count) = analyzer.get_timing_info()?;
                            println!("[Frame {}] [t={:.1}s] {}", processed_frames, timestamp, analysis);
                            processed_frames += 1;
                        }
                        Err(e) => {
                            eprintln!("❌ Error analyzing frame: {}", e);
                        }
                    }
                }
                None => {
                    println!("📹 No more frames available");
                    break;
                }
            }
        }

        // Cleanup
        analyzer.cleanup()?;
        println!("👋 Goodbye!");
        Ok(())
    })
}
