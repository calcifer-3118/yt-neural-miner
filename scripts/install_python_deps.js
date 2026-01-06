const { spawn } = require("child_process");
const path = require("path");

console.log("🐍 Setting up Python environment...");

const requirementsPath = path.join(
  __dirname,
  "../lib/python-core/requirements.txt"
);

// On Windows, use 'python'. On Mac/Linux, try 'python3' first.
const pythonCmd = process.platform === "win32" ? "python" : "python3";

// We use 'python -m pip' instead of just 'pip' to avoid path issues
const args = ["-m", "pip", "install", "-r", requirementsPath];

console.log(`   Running: ${pythonCmd} ${args.join(" ")}`);

// shell: true is safer on Windows to handle path variables correctly
const install = spawn(pythonCmd, args, { shell: true });

install.stdout.on("data", (data) => console.log(data.toString()));
install.stderr.on("data", (data) => console.error(data.toString()));

install.on("error", (err) => {
  console.error("❌ Failed to start Python process:", err);
});

install.on("close", (code) => {
  if (code === 0) {
    console.log("✅ Python dependencies installed successfully.");
  } else {
    console.error(
      `❌ Failed with code ${code}. Please run this manually:\n   pip install -r lib/python-core/requirements.txt`
    );
  }
});
