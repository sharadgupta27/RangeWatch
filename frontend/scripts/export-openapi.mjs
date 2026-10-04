// Export the backend's OpenAPI schema to ./openapi.json using the repo's virtualenv
// (falls back to `python` on PATH when no .venv exists, e.g. in CI images).
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

const repo = resolve(import.meta.dirname, '..', '..')
const candidates = [
  resolve(repo, '.venv', 'Scripts', 'python.exe'),
  resolve(repo, '.venv', 'bin', 'python'),
]
const python = candidates.find((p) => existsSync(p)) ?? 'python'

execFileSync(python, ['-m', 'src.api.export_openapi', '../frontend/openapi.json'], {
  cwd: resolve(repo, 'backend'),
  stdio: 'inherit',
})
