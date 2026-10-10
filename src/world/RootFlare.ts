import * as THREE from 'three/webgpu';

/**
 * The foot of a conifer: the trunk swells into buttress roots that run out and down into the ground (round 1's
 * trunks went in like fence posts; the tree models only taper a few centimetres). Unit trunk radius at the ground,
 * y up; the top ring sits just inside the trunk. Attribute layout as in the tree GLBs (float position, normal, uv;
 * unorm8 color = baked visibility, wind = 0), so it draws with the bark program.
 */
export function rootFlareGeometry(seed: number): THREE.BufferGeometry {
  let r = seed >>> 0;
  const rnd = () => ((r = (Math.imul(r, 1664525) + 1013904223) >>> 0) / 4294967296);
  const SEG = 20;
  const ys = [-0.5, -0.24, -0.08, 0.04, 0.18, 0.38, 0.66, 1.05, 1.6];
  const nRoots = 4 + Math.floor(rnd() * 3);
  const roots = Array.from({ length: nRoots }, (_, i) => ({
    a: ((i + 0.3 * (rnd() - 0.5)) / nRoots) * Math.PI * 2,
    len: 0.8 + 1.1 * rnd(),
    w: 0.24 + 0.18 * rnd(),
  }));
  const pos: number[] = [], uv: number[] = [], col: number[] = [];
  for (let j = 0; j < ys.length; j++) {
    const y = ys[j];
    const top = j === ys.length - 1;
    // (forest polish: a stronger swell and longer roots, round 2's flare did not read at C5's distance)
    const swell = top ? 0.96 : 1 + 0.7 * Math.exp(-Math.max(y, 0) * 2.2);
    for (let i = 0; i <= SEG; i++) {
      const a = (i / SEG) * Math.PI * 2;
      let lobe = 0;
      for (const rt of roots) {
        let d = Math.abs(a - rt.a) % (Math.PI * 2);
        if (d > Math.PI) d = Math.PI * 2 - d;
        lobe = Math.max(lobe, rt.len * Math.exp(-((d / rt.w) ** 2)));
      }
      // the roots reach out near the ground and keep spreading under it
      const reach = top ? 0 : lobe * Math.exp(-Math.max(y + 0.05, 0) * 3.4) * (y < 0 ? 1 - y * 1.4 : 1);
      const R = swell + reach;
      pos.push(Math.cos(a) * R, y - (y < 0 ? reach * 0.15 : 0), Math.sin(a) * R);
      uv.push((i / SEG) * 1.4, y * 0.55);
      // darker toward the soil (litter and damp at the foot)
      const vis = Math.round(255 * (0.55 + 0.45 * Math.min(1, Math.max(0, (y + 0.1) / 1.2))));
      col.push(vis, vis, vis, 255);
    }
  }
  const index: number[] = [];
  const row = SEG + 1;
  for (let j = 0; j < ys.length - 1; j++) {
    for (let i = 0; i < SEG; i++) {
      const a = j * row + i, b = a + 1, c = a + row, d = c + 1;
      index.push(a, c, b, b, c, d);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uv), 2));
  g.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(col), 4, true));
  g.setAttribute('wind', new THREE.BufferAttribute(new Uint8Array((pos.length / 3) * 4), 4, true));
  g.setIndex(new THREE.BufferAttribute(new Uint16Array(index), 1));
  g.computeVertexNormals();
  // the seam column shares positions but not normals: average them
  const n = g.getAttribute('normal') as THREE.BufferAttribute;
  for (let j = 0; j < ys.length; j++) {
    const a = j * row, b = a + SEG;
    const x = n.getX(a) + n.getX(b), yy = n.getY(a) + n.getY(b), z = n.getZ(a) + n.getZ(b), l = Math.hypot(x, yy, z) || 1;
    n.setXYZ(a, x / l, yy / l, z / l);
    n.setXYZ(b, x / l, yy / l, z / l);
  }
  g.computeBoundingSphere();
  return g;
}
