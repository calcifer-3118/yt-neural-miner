const { spawn } = require("child_process");
const path = require("path");
const chalk = require("chalk");
const cliProgress = require("cli-progress");
const readline = require("readline");

const PYTHON_SCRIPT = path.join(__dirname, "python-core", "miner.py");

let activeProcess = null;

async function runScheduler(
  urls,
  stages,
  storageMode,
  cleanup,
  cookies,
  syncOnly,
  dbUrl
) {
  // --- HEADER ---
  console.clear();
  console.log(chalk.bold.hex("#00d4ff")(`\n ⚡ NEURAL MINER v2.7 `));
  console.log(
    chalk.gray(` ──────────────────────────────────────────────────`)
  );
  console.log(
    ` ${chalk.bold("Mode:")}    ${
      syncOnly
        ? chalk.magenta("SYNC ONLY")
        : chalk.green(storageMode.toUpperCase())
    }`
  );
  console.log(` ${chalk.bold("Targets:")} ${urls.length} Video(s)`);
  console.log(
    ` ${chalk.bold("DB Connection:")} ${
      dbUrl ? chalk.green("Active") : chalk.dim("None")
    }`
  );
  console.log(
    chalk.gray(` ──────────────────────────────────────────────────`)
  );
  console.log(chalk.dim(" Controls: [S]kip Current Stage | [K]ill Job\n"));

  // --- BARS ---
  const multiBar = new cliProgress.MultiBar(
    {
      clearOnComplete: false,
      hideCursor: true,
      // [Bar] 50% | ETA: 12s | Task Name
      format: `${chalk.cyan(
        "{bar}"
      )} {percentage}% | ETA: {eta_formatted} | {stage}`,
      barCompleteChar: "\u2588",
      barIncompleteChar: "\u2591",
      fps: 5,
      etaBuffer: 50,
    },
    cliProgress.Presets.shades_grey
  );

  const overallBar = multiBar.create(urls.length, 0, {
    stage: chalk.white("Overall Progress"),
  });
  const taskBar = multiBar.create(100, 0, {
    stage: chalk.gray("Waiting to start..."),
  });

  setupControls(taskBar);

  for (let i = 0; i < urls.length; i++) {
    const url = urls[i];

    // Log active video
    multiBar.log(chalk.white.bold(`\n▶ Processing: ${url}\n`));

    taskBar.setTotal(100);
    taskBar.update(0, { stage: "Initializing..." });

    try {
      await runPythonMiner(
        url,
        stages,
        storageMode,
        cleanup,
        cookies,
        syncOnly,
        dbUrl,
        taskBar,
        multiBar
      );
    } catch (e) {
      if (e.message !== "SKIPPED_VIDEO") {
        multiBar.log(chalk.red(`\n❌ Error: ${e.message}\n`));
      }
    }

    overallBar.increment();
  }

  taskBar.update(100, { stage: chalk.green("Done") });
  multiBar.stop();
  console.log(chalk.green.bold("\n✨ Pipeline Completed Successfully.\n"));
  process.exit(0);
}

function runPythonMiner(
  url,
  stages,
  storageMode,
  cleanup,
  cookies,
  syncOnly,
  dbUrl,
  taskBar,
  multiBar
) {
  return new Promise((resolve, reject) => {
    const args = [
      "-u",
      PYTHON_SCRIPT,
      "--url",
      url,
      "--mode",
      storageMode,
      "--non-interactive",
    ];

    if (syncOnly) args.push("--sync_only");
    else {
      const stageList = Array.isArray(stages) ? stages : [stages];
      stageList.forEach((s) => {
        args.push("--process");
        args.push(s);
      });
    }

    if (cleanup) args.push("--cleanup");
    if (cookies) {
      args.push("--cookies");
      args.push(cookies);
    }

    const envVars = {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
      PYTHONUNBUFFERED: "1",
      MINER_DB_URL: dbUrl || "",
    };

    const pythonCmd = process.platform === "win32" ? "python" : "python3";
    activeProcess = spawn(pythonCmd, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: envVars,
    });

    activeProcess.stdout.on("data", (data) => {
      const lines = data.toString().split("\n");
      lines.forEach((line) => {
        const clean = line.trim();
        if (!clean) return;

        // PROTOCOL: PRG:StageName:Current:Total
        if (clean.startsWith("PRG:")) {
          const parts = clean.split(":");
          if (parts.length >= 4) {
            const stageName = parts[1];
            const current = parseFloat(parts[2]);
            const total = parseFloat(parts[3]);

            // Colorize Stage
            let displayStage = chalk.yellow(stageName);
            if (stageName.includes("Audio"))
              displayStage = chalk.magenta(`Audio: ${parts[2]}%`);
            else if (stageName.includes("Video"))
              displayStage = chalk.blue(`Video: ${parts[2]}%`);
            else if (stageName.includes("Metadata"))
              displayStage = chalk.cyan("Metadata Analysis");
            else if (stageName.includes("DB"))
              displayStage = chalk.green("Database Sync");

            if (!isNaN(current)) {
              taskBar.setTotal(total);
              taskBar.update(current, { stage: displayStage });
            } else {
              // Text Status update (e.g. PRG:Audio:Transcribing...:100)
              taskBar.update(null, {
                stage: chalk.yellow(`${stageName}: ${parts[2]}`),
              });
            }
          }
        }
        // LOGS
        else {
          if (clean.includes("❌")) multiBar.log(chalk.red(`   ${clean}\n`));
          else if (clean.includes("✅"))
            multiBar.log(chalk.green(`   ${clean}\n`));
          else if (clean.includes("⏳"))
            multiBar.log(chalk.yellow(`   ${clean}\n`));
          else if (clean.includes("⏭"))
            multiBar.log(chalk.magenta(`   ${clean}\n`));
          else if (clean.includes("⬇️"))
            taskBar.update(null, { stage: "Downloading..." });
          else if (clean.includes("🧠"))
            taskBar.update(null, { stage: "Loading AI..." });
        }
      });
    });

    activeProcess.stderr.on("data", (data) => {
      const err = data.toString();
      if (
        !err.includes("UserWarning") &&
        !err.includes("FutureWarning") &&
        !err.includes("libpng") &&
        !err.includes("huggingface") &&
        !err.includes("tqdm")
      ) {
        multiBar.log(chalk.red(`   [PY] ${err.trim()}\n`));
      }
    });

    activeProcess.on("close", (code) => {
      activeProcess = null;
      if (code === 0) resolve();
      else reject(new Error(`Exit code ${code}`));
    });
  });
}

function setupControls(taskBar) {
  readline.emitKeypressEvents(process.stdin);
  if (process.stdin.isTTY) process.stdin.setRawMode(true);

  process.stdin.on("keypress", (str, key) => {
    if (!activeProcess) return;

    // KILL
    if (key.name === "k" || (key.ctrl && key.name === "c")) {
      activeProcess.kill();
      console.log(chalk.red("\n ⛔ Pipeline Killed by User."));
      process.exit(0);
    }

    // SKIP
    if (key.name === "s") {
      taskBar.update(null, { stage: chalk.magenta(">>> SKIPPING STAGE >>>") });
      if (activeProcess.stdin) activeProcess.stdin.write("skip\n");
    }
  });
}

module.exports = { runScheduler };
