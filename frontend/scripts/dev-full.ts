const host = '127.0.0.1';
const port = process.env.RL_INTERN_SERVER_PORT ?? '8765';
const apiUrl = process.env.VITE_RL_INTERN_API_URL ?? `http://${host}:${port}`;
const repoRoot = new URL('../..', import.meta.url).pathname;

async function isServerReady(): Promise<boolean> {
  try {
    const response = await fetch(`${apiUrl}/api/health`);
    return response.ok;
  } catch {
    return false;
  }
}

function spawnManaged(command: string[], cwd: string): Bun.Subprocess {
  return Bun.spawn(command, {
    cwd,
    stdout: 'inherit',
    stderr: 'inherit',
    stdin: 'inherit',
  });
}

const children = new Set<Bun.Subprocess>();

function stopChildren() {
  for (const child of children) {
    child.kill();
  }
}

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => {
    stopChildren();
    process.exit(signal === 'SIGINT' ? 130 : 143);
  });
}

if (!(await isServerReady())) {
  console.log(`starting rl-intern server on ${apiUrl}`);
  children.add(
    spawnManaged(['uv', 'run', 'rl-intern-server', '--host', host, '--port', port], repoRoot),
  );
}

children.add(spawnManaged(['bun', '--bun', 'vite', '--host', host], process.cwd()));

const exits = [...children].map(async (child) => {
  const code = await child.exited;
  children.delete(child);
  return code;
});

const code = await Promise.race(exits);
stopChildren();
process.exit(code ?? 0);
