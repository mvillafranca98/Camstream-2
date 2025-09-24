use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() {
    println!("🚀 Rust-driven Camera Stream Analyzer");
    println!("=====================================");
    println!("Rust is now orchestrating the Python FastVLM analysis!");
    println!("Press Ctrl+C to stop gracefully\n");
    
    // Set up signal handling for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
    ctrlc::set_handler(move || {
        println!("\n🛑 Shutdown requested...");
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl+C handler");
    
    // Get the current directory
    let current_dir = std::env::current_dir().expect("Failed to get current directory");
    
    // Run the Python script with real-time output streaming
    let mut child = Command::new(".venv/bin/python")
        .arg("rust_camera_cli.py")
        .arg("0")           // device
        .arg("640")         // width
        .arg("480")         // height
        .arg("30")          // framerate
        .arg("unlimited")   // max_frames (unlimited)
        .arg("15")          // frame_skip
        .current_dir(&current_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to execute Python script");
    
    // Stream the output in real-time
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if !running.load(Ordering::SeqCst) {
                break;
            }
            if let Ok(line) = line {
                println!("{}", line);
            }
        }
    }
    
    // Handle stderr
    if let Some(stderr) = child.stderr.take() {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("⚠️ {}", line);
            }
        }
    }
    
    // Wait for the process to complete or Ctrl+C
    let status = if running.load(Ordering::SeqCst) {
        child.wait().expect("Failed to wait for Python process")
    } else {
        let _ = child.kill();
        println!("\n🛑 Process terminated gracefully");
        return;
    };
    
    if status.success() {
        println!("\n✅ Rust successfully orchestrated the Python camera analysis!");
    } else {
        println!("\n❌ Python process exited with error code: {}", status);
    }
}
