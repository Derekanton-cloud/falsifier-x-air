import { execFile } from "child_process";
import { existsSync, readdirSync } from "fs";
import { join } from "path";
import { promisify } from "util";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
const execFileAsync = promisify(execFile);

function resolvePython(): string {
  const configured = process.env.FALSIFIER_PYTHON ?? process.env.PYTHON;
  if (configured) return configured;
  if (process.platform !== "win32") return "python";
  const installations = join(process.env.LOCALAPPDATA ?? "", "Programs", "Python");
  try {
    const executable = readdirSync(installations)
      .sort()
      .reverse()
      .map((directory) => join(installations, directory, "python.exe"))
      .find((candidate) => existsSync(candidate));
    if (executable) return executable;
  } catch {
    // The configured Python fallback below retains the original behavior.
  }
  return "python";
}

/** Executes the existing public Python demo path; no oracle or benchmark fields are returned. */
export async function POST(request: Request) {
  const python = resolvePython();
  const requestBody = await request.json().catch(() => ({})) as { case?: string };
  const investigationCase = requestBody.case === "indistinguishable" ? "indistinguishable" : "identifiable";
  try {
    const { stdout } = await execFileAsync(python, ["frontend/bridge/live_investigation.py", investigationCase], {
      cwd: process.cwd() + "/..", timeout: 15_000, windowsHide: true, maxBuffer: 1024 * 1024,
    });
    return NextResponse.json(JSON.parse(stdout));
  } catch (error) {
    return NextResponse.json({ error: "Live Python bridge unavailable", detail: String(error) }, { status: 503 });
  }
}
