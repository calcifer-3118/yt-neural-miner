#!/usr/bin/env node

const { program } = require("commander");
const { runScheduler } = require("../lib/index");
const inquirer = require("inquirer");
const fs = require("fs");

program.name("miner").description("Neural Miner").version("2.6.0");

function parseInput(input) {
  if (fs.existsSync(input)) {
    return fs
      .readFileSync(input, "utf-8")
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
  }
  return [input];
}

async function getDbUrl(provided) {
  if (provided) return provided;
  const ans = await inquirer.prompt([
    {
      type: "input",
      name: "db",
      message: "Database URL:",
      validate: (i) => i.length > 5,
    },
  ]);
  return ans.db;
}

program
  .command("run <input>")
  .option("-p, --process <items...>", "Stages", "all")
  .option("--mode <type>", "Mode: db/local", null)
  .option("--db <url>", "DB URL", null)
  .option("--cookies <source>", "Cookies path", null)
  .option("--keep", "Keep files", false)
  .action(async (input, options) => {
    let mode = options.mode;
    let dbUrl = options.db;
    if (!mode) {
      const ans = await inquirer.prompt([
        {
          type: "list",
          name: "mode",
          message: "Select Mode:",
          choices: [
            { name: "Local", value: "local" },
            { name: "Database", value: "db" },
          ],
        },
      ]);
      mode = ans.mode;
    }
    if (mode === "db") dbUrl = await getDbUrl(dbUrl);

    let stages = options.stages || "all";
    if (Array.isArray(stages) && stages.includes("all")) stages = "all";

    runScheduler(
      parseInput(input),
      stages,
      mode,
      !options.keep,
      options.cookies,
      false,
      dbUrl
    );
  });

program
  .command("sync <input>")
  .description("Sync existing data")
  .option("--db <url>", "DB URL", null)
  .option("--cookies <source>", "Cookies", null)
  .action(async (input, options) => {
    const dbUrl = await getDbUrl(options.db);
    const ans = await inquirer.prompt([
      { type: "confirm", name: "c", message: "Cleanup?", default: false },
    ]);
    runScheduler(
      parseInput(input),
      [],
      "db",
      ans.c,
      options.cookies,
      true,
      dbUrl
    );
  });

program.parse();
