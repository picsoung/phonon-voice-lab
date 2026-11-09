// The WebAssembly SIMD the CPU tier can use here, without loading the CPU tier.
/** Modules that validate only with relaxed SIMD (i8x16.relaxed_swizzle), or with SIMD128. */
const RELAXED = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 0, 253, 15, 65, 0, 253, 15, 253,
  128, 2, 11,
]);
const SIMD = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
export function simdLevel(): "relaxed" | "simd" | null {
  if (WebAssembly.validate(RELAXED)) return "relaxed";
  if (WebAssembly.validate(SIMD)) return "simd";
  return null;
}
