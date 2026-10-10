/**
 * Instance capacity for an InstancedMesh holding `n` instances. three keeps up to 64 KB of instance matrices (1024)
 * in a uniform array whose length is written into the shader, so every distinct capacity compiled its own program
 * (round 1: 50 bark, 30 needle and 76 shadow-pass pipelines for 4 materials). Above that limit it reads the matrices
 * as instanced vertex attributes, with the same shader for every mesh of a material.
 */
export const instanceCapacity = (n: number) => Math.max(n, 1100);
