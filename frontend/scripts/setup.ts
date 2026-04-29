import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const envPath = resolve(repoRoot, '.env');
const force = process.argv.includes('--force');

type EnvMap = Map<string, string>;

function run(command: string[], options: { allowFailure?: boolean; capture?: boolean } = {}) {
  const proc = Bun.spawnSync(command, {
    cwd: repoRoot,
    stdout: options.capture ? 'pipe' : 'inherit',
    stderr: options.capture ? 'pipe' : 'inherit',
    stdin: 'inherit',
  });
  if (proc.exitCode !== 0 && !options.allowFailure) {
    throw new Error(`command failed: ${command.join(' ')}`);
  }
  return proc;
}

function hasCommand(name: string): boolean {
  return run(['which', name], { allowFailure: true, capture: true }).exitCode === 0;
}

function readEnv(): EnvMap {
  const values: EnvMap = new Map();
  if (!existsSync(envPath)) return values;
  for (const line of readFileSync(envPath, 'utf8').split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$/);
    if (!match) continue;
    values.set(match[1], match[2]);
  }
  return values;
}

function writeEnv(values: EnvMap) {
  const existing = existsSync(envPath) ? readFileSync(envPath, 'utf8').split(/\r?\n/) : [];
  const written = new Set<string>();
  const lines = existing
    .filter((line) => line.trim() !== '')
    .map((line) => {
      const match = line.match(/^(\s*)([A-Za-z_][A-Za-z0-9_]*)(=.*)$/);
      if (!match) return line;
      const key = match[2];
      if (!values.has(key)) return line;
      written.add(key);
      return `${key}=${values.get(key)}`;
    });

  for (const [key, value] of values) {
    if (!written.has(key)) lines.push(`${key}=${value}`);
  }
  writeFileSync(envPath, `${lines.join('\n')}\n`);
}

function promptValue(label: string, current?: string): string {
  if (current && !force) return current;
  const suffix = current ? ' [already set, press enter to keep]' : '';
  const value = prompt(`${label}${suffix}:`)?.trim();
  return value || current || '';
}

function modalReady(): boolean {
  return run(['uv', 'run', 'modal', 'token', 'list'], {
    allowFailure: true,
    capture: true,
  }).exitCode === 0;
}

function deployModalApp(path: string) {
  run(['uv', 'run', 'modal', 'deploy', path]);
}

if (!existsSync(resolve(repoRoot, 'pyproject.toml'))) {
  throw new Error(`could not find repo root from ${scriptDir}`);
}

if (!hasCommand('uv')) {
  throw new Error('uv is required. Install it first: https://docs.astral.sh/uv/');
}

if (!hasCommand('bun')) {
  throw new Error('bun is required. Install it first: https://bun.sh/');
}

console.log('syncing Python dependencies');
run(['uv', 'sync', '--extra', 'server', '--extra', 'modal', '--extra', 'llm']);

const env = readEnv();
const openrouter = promptValue('OPENROUTER_API_KEY', env.get('OPENROUTER_API_KEY'));
const hfToken = promptValue('HF_TOKEN', env.get('HF_TOKEN') || env.get('HUGGINGFACE_HUB_TOKEN'));

if (openrouter) env.set('OPENROUTER_API_KEY', openrouter);
if (hfToken) {
  env.set('HF_TOKEN', hfToken);
  env.set('HUGGINGFACE_HUB_TOKEN', hfToken);
}
env.set(
  'RL_INTERN_DEFAULT_MODEL',
  env.get('RL_INTERN_DEFAULT_MODEL') || 'openrouter/anthropic/claude-sonnet-4.5',
);
writeEnv(env);

if (!modalReady()) {
  console.log('starting Modal authentication');
  run(['uv', 'run', 'modal', 'setup']);
}

console.log('deploying Modal apps');
deployModalApp('rl_intern/modal_jobs/generic.py');
deployModalApp('rl_intern/modal_jobs/sb3.py');

console.log('\nsetup complete');
console.log('start the app with:');
console.log('  cd frontend && bun run dev');
