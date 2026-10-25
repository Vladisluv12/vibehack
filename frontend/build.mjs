import { build } from 'esbuild';
import { copyFile, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

const result = await build({entryPoints:['frontend/app.js'],bundle:true,minify:true,format:'esm',
  outfile:'second_brain/static/app.js',legalComments:'linked',metafile:true});
await copyFile('frontend/app.css','second_brain/static/app.css');

// Notices for exactly the packages that ended up in the bundle.
const packages = new Map();
for (const input of Object.keys(result.metafile.inputs)) {
  const match = input.match(/^(.*node_modules[\\/](@[^\\/]+[\\/][^\\/]+|[^\\/]+))[\\/]/);
  if (match && !packages.has(match[2])) packages.set(match[2].replace('\\', '/'), match[1]);
}
let notices = '# Frontend third-party notices\n\nLocal bundled dependencies; see pnpm-lock.yaml for exact versions.\n';
for (const [name, dir] of [...packages].sort(([a], [b]) => a.localeCompare(b))) {
  const meta = JSON.parse(await readFile(join(dir, 'package.json'), 'utf8'));
  const file = (await readdir(dir)).find(file => /^(licen[cs]e|copying)/i.test(file));
  if (!file && !meta.license) throw new Error(`No license information in ${name}`);
  const author = typeof meta.author === 'object' ? meta.author?.name : meta.author;
  const text = file ? (await readFile(join(dir, file), 'utf8')).trim()
    : `License: ${meta.license} (declared in package.json; the package ships no license file).` +
      (author ? `\nAuthor: ${author}` : '') + (meta.repository?.url ? `\nSource: ${meta.repository.url}` : '');
  notices += `\n\n## ${name} ${meta.version}\n\n${text}\n`;
}
await writeFile(join(dirname(import.meta.dirname), 'THIRD_PARTY_NOTICES.md'), notices);
console.log(`Local graph assets built (${packages.size} bundled packages).`);
