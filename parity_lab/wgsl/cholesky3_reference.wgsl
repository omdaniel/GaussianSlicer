// Reference WGSL snippet for 3x3 SPD Cholesky + solve (lifted from AGENT-PARITYLAB.md).
struct Mat3 { c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32> };

fn m3_get(m: Mat3, r: u32, c: u32) -> f32 {
  return (c == 0u ? m.c0[r] : (c == 1u ? m.c1[r] : m.c2[r]));
}

fn m3_set(inout m: Mat3, r: u32, c: u32, v: f32) {
  if (c == 0u) {
    m.c0[r] = v;
  } else if (c == 1u) {
    m.c1[r] = v;
  } else {
    m.c2[r] = v;
  }
}

fn cholesky3(a: Mat3) -> Mat3 {
  var L = Mat3(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  for (var i: u32 = 0u; i < 3u; i++) {
    for (var j: u32 = 0u; j <= i; j++) {
      var sum = m3_get(a, i, j);
      for (var k: u32 = 0u; k < j; k++) {
        sum -= m3_get(L, i, k) * m3_get(L, j, k);
      }
      if (i == j) {
        let d = max(sum, 1e-12);
        m3_set(L, i, j, sqrt(d));
      } else {
        m3_set(L, i, j, sum / m3_get(L, j, j));
      }
    }
  }
  return L;
}

fn forward_substitute(L: Mat3, b: vec3<f32>) -> vec3<f32> {
  var y = vec3<f32>(0.0);
  y.x = b.x / m3_get(L, 0u, 0u);
  y.y = (b.y - m3_get(L,1u,0u)*y.x) / m3_get(L,1u,1u);
  y.z = (b.z - m3_get(L,2u,0u)*y.x - m3_get(L,2u,1u)*y.y) / m3_get(L,2u,2u);
  return y;
}

fn back_substitute(L: Mat3, y: vec3<f32>) -> vec3<f32> {
  var x = vec3<f32>(0.0);
  x.z = y.z / m3_get(L,2u,2u);
  x.y = (y.y - m3_get(L,2u,1u)*x.z) / m3_get(L,1u,1u);
  x.x = (y.x - m3_get(L,1u,0u)*x.y - m3_get(L,2u,0u)*x.z) / m3_get(L,0u,0u);
  return x;
}

fn solve_spd3(A: Mat3, b: vec3<f32>) -> vec3<f32> {
  let L = cholesky3(A);
  let y = forward_substitute(L, b);
  return back_substitute(L, y);
}
