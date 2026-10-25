import { build } from 'esbuild';
import { copyFile } from 'node:fs/promises';

await build({entryPoints:['frontend/app.js'],bundle:true,minify:true,format:'esm',
  outfile:'second_brain/static/app.js',legalComments:'linked'});
await copyFile('frontend/app.css','second_brain/static/app.css');
console.log('Local graph assets built.');
