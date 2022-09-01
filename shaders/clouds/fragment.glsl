#version 330
//#define SHADERTOY 1
//#iChannel0 "file://../textures/wall01.jpg"

// Aux simple functions
#define TWO_PI 6.283185
#define PI 3.14159
float norm(float x) { return x*.5 + .5; }
float denorm(float x) { return x*2. - 1.; }
float xstep(float b, float e, float v) {
    return step(b, v) - step(e, v);
}
float xsmoothstep(float b, float e, float v) {
    return smoothstep(b, e, v) - step(e, v);
}
float flatten(float v, float layers) {
  return floor(v*layers) * (1./layers);
}
float nsin(float t) {return norm(sin(t * TWO_PI));}
float ncos(float t) {return norm(cos(t * TWO_PI));}
float ntan(float t) {return norm(tan(t * TWO_PI));}
float sat(float t) {return clamp(t, 0., 1.);}
float rsat(float t) {return mod(t+10000.0, 1.);}
float xclamp(float v, float minV, float maxV) {
  return clamp(v, minV, maxV) * xstep(minV, maxV, v);
}
float xclampnorm(float v, float minV, float maxV) {
  return (xclamp(v, minV, maxV) - minV) / (maxV-minV);
}
float normrange(float v, float minV, float maxV) { return sat((v-minV)/(maxV-minV)); }

// -------------------
// BEGIN https://github.com/stegu/webgl-noise
// -------------------

vec3 mod289(vec3 x) {return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 mod289(vec4 x) {return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec3 mod7(vec3 x) {return x - floor(x * (1.0 / 7.0)) * 7.0;}
vec3 permute(vec3 x) {return mod289((34.0 * x + 10.0) * x);}
vec4 permute(vec4 x) {return mod289(((x*34.0)+10.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
vec3 fade(vec3 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

// Cellular
vec2 cellular(vec3 P) {
#define K 0.142857142857 // 1/7
#define Ko 0.428571428571 // 1/2-K/2
#define K2 0.020408163265306 // 1/(7*7)
#define Kz 0.166666666667 // 1/6
#define Kzo 0.416666666667 // 1/2-1/6*2
#define jitter 1.0 // smaller jitter gives more regular pattern

	vec3 Pi = mod289(floor(P));
 	vec3 Pf = fract(P) - 0.5;

	vec3 Pfx = Pf.x + vec3(1.0, 0.0, -1.0);
	vec3 Pfy = Pf.y + vec3(1.0, 0.0, -1.0);
	vec3 Pfz = Pf.z + vec3(1.0, 0.0, -1.0);

	vec3 p = permute(Pi.x + vec3(-1.0, 0.0, 1.0));
	vec3 p1 = permute(p + Pi.y - 1.0);
	vec3 p2 = permute(p + Pi.y);
	vec3 p3 = permute(p + Pi.y + 1.0);

	vec3 p11 = permute(p1 + Pi.z - 1.0);
	vec3 p12 = permute(p1 + Pi.z);
	vec3 p13 = permute(p1 + Pi.z + 1.0);

	vec3 p21 = permute(p2 + Pi.z - 1.0);
	vec3 p22 = permute(p2 + Pi.z);
	vec3 p23 = permute(p2 + Pi.z + 1.0);

	vec3 p31 = permute(p3 + Pi.z - 1.0);
	vec3 p32 = permute(p3 + Pi.z);
	vec3 p33 = permute(p3 + Pi.z + 1.0);

	vec3 ox11 = fract(p11*K) - Ko;
	vec3 oy11 = mod7(floor(p11*K))*K - Ko;
	vec3 oz11 = floor(p11*K2)*Kz - Kzo; // p11 < 289 guaranteed

	vec3 ox12 = fract(p12*K) - Ko;
	vec3 oy12 = mod7(floor(p12*K))*K - Ko;
	vec3 oz12 = floor(p12*K2)*Kz - Kzo;

	vec3 ox13 = fract(p13*K) - Ko;
	vec3 oy13 = mod7(floor(p13*K))*K - Ko;
	vec3 oz13 = floor(p13*K2)*Kz - Kzo;

	vec3 ox21 = fract(p21*K) - Ko;
	vec3 oy21 = mod7(floor(p21*K))*K - Ko;
	vec3 oz21 = floor(p21*K2)*Kz - Kzo;

	vec3 ox22 = fract(p22*K) - Ko;
	vec3 oy22 = mod7(floor(p22*K))*K - Ko;
	vec3 oz22 = floor(p22*K2)*Kz - Kzo;

	vec3 ox23 = fract(p23*K) - Ko;
	vec3 oy23 = mod7(floor(p23*K))*K - Ko;
	vec3 oz23 = floor(p23*K2)*Kz - Kzo;

	vec3 ox31 = fract(p31*K) - Ko;
	vec3 oy31 = mod7(floor(p31*K))*K - Ko;
	vec3 oz31 = floor(p31*K2)*Kz - Kzo;

	vec3 ox32 = fract(p32*K) - Ko;
	vec3 oy32 = mod7(floor(p32*K))*K - Ko;
	vec3 oz32 = floor(p32*K2)*Kz - Kzo;

	vec3 ox33 = fract(p33*K) - Ko;
	vec3 oy33 = mod7(floor(p33*K))*K - Ko;
	vec3 oz33 = floor(p33*K2)*Kz - Kzo;

	vec3 dx11 = Pfx + jitter*ox11;
	vec3 dy11 = Pfy.x + jitter*oy11;
	vec3 dz11 = Pfz.x + jitter*oz11;

	vec3 dx12 = Pfx + jitter*ox12;
	vec3 dy12 = Pfy.x + jitter*oy12;
	vec3 dz12 = Pfz.y + jitter*oz12;

	vec3 dx13 = Pfx + jitter*ox13;
	vec3 dy13 = Pfy.x + jitter*oy13;
	vec3 dz13 = Pfz.z + jitter*oz13;

	vec3 dx21 = Pfx + jitter*ox21;
	vec3 dy21 = Pfy.y + jitter*oy21;
	vec3 dz21 = Pfz.x + jitter*oz21;

	vec3 dx22 = Pfx + jitter*ox22;
	vec3 dy22 = Pfy.y + jitter*oy22;
	vec3 dz22 = Pfz.y + jitter*oz22;

	vec3 dx23 = Pfx + jitter*ox23;
	vec3 dy23 = Pfy.y + jitter*oy23;
	vec3 dz23 = Pfz.z + jitter*oz23;

	vec3 dx31 = Pfx + jitter*ox31;
	vec3 dy31 = Pfy.z + jitter*oy31;
	vec3 dz31 = Pfz.x + jitter*oz31;

	vec3 dx32 = Pfx + jitter*ox32;
	vec3 dy32 = Pfy.z + jitter*oy32;
	vec3 dz32 = Pfz.y + jitter*oz32;

	vec3 dx33 = Pfx + jitter*ox33;
	vec3 dy33 = Pfy.z + jitter*oy33;
	vec3 dz33 = Pfz.z + jitter*oz33;

	vec3 d11 = dx11 * dx11 + dy11 * dy11 + dz11 * dz11;
	vec3 d12 = dx12 * dx12 + dy12 * dy12 + dz12 * dz12;
	vec3 d13 = dx13 * dx13 + dy13 * dy13 + dz13 * dz13;
	vec3 d21 = dx21 * dx21 + dy21 * dy21 + dz21 * dz21;
	vec3 d22 = dx22 * dx22 + dy22 * dy22 + dz22 * dz22;
	vec3 d23 = dx23 * dx23 + dy23 * dy23 + dz23 * dz23;
	vec3 d31 = dx31 * dx31 + dy31 * dy31 + dz31 * dz31;
	vec3 d32 = dx32 * dx32 + dy32 * dy32 + dz32 * dz32;
	vec3 d33 = dx33 * dx33 + dy33 * dy33 + dz33 * dz33;

	// Sort out the two smallest distances (F1, F2)
#if 0
	// Cheat and sort out only F1
	vec3 d1 = min(min(d11,d12), d13);
	vec3 d2 = min(min(d21,d22), d23);
	vec3 d3 = min(min(d31,d32), d33);
	vec3 d = min(min(d1,d2), d3);
	d.x = min(min(d.x,d.y),d.z);
	return vec2(sqrt(d.x)); // F1 duplicated, no F2 computed
#else
	// Do it right and sort out both F1 and F2
	vec3 d1a = min(d11, d12);
	d12 = max(d11, d12);
	d11 = min(d1a, d13); // Smallest now not in d12 or d13
	d13 = max(d1a, d13);
	d12 = min(d12, d13); // 2nd smallest now not in d13
	vec3 d2a = min(d21, d22);
	d22 = max(d21, d22);
	d21 = min(d2a, d23); // Smallest now not in d22 or d23
	d23 = max(d2a, d23);
	d22 = min(d22, d23); // 2nd smallest now not in d23
	vec3 d3a = min(d31, d32);
	d32 = max(d31, d32);
	d31 = min(d3a, d33); // Smallest now not in d32 or d33
	d33 = max(d3a, d33);
	d32 = min(d32, d33); // 2nd smallest now not in d33
	vec3 da = min(d11, d21);
	d21 = max(d11, d21);
	d11 = min(da, d31); // Smallest now in d11
	d31 = max(da, d31); // 2nd smallest now not in d31
	d11.xy = (d11.x < d11.y) ? d11.xy : d11.yx;
	d11.xz = (d11.x < d11.z) ? d11.xz : d11.zx; // d11.x now smallest
	d12 = min(d12, d21); // 2nd smallest now not in d21
	d12 = min(d12, d22); // nor in d22
	d12 = min(d12, d31); // nor in d31
	d12 = min(d12, d32); // nor in d32
	d11.yz = min(d11.yz,d12.xy); // nor in d12.yz
	d11.y = min(d11.y,d12.z); // Only two more to go
	d11.y = min(d11.y,d11.z); // Done! (Phew!)
	return sqrt(d11.xy); // F1, F2
#endif
}

vec2 ncellular(vec2 p, float seed) {
  p *= 15.;
  seed *= 0.3;
  return cellular(vec3(p, seed));
}

// Classic Perlin noise
float perlin(vec3 P)
{
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}

float nperlin(vec2 p, float seed) {
  p *= 20.;
  seed *= 0.5;
  return norm(perlin(vec3(p, seed)));
}

// Simplex Noise
float simplex(vec3 v)
  {
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
}

float nsimplex(vec2 x, float seed) {
  x *= 10.;
  seed *= 0.3;
  return norm(simplex(vec3(x, seed)));
}


// -------------------
// END noise3D
// -------------------

// -------------------
// BEGIN https://www.shadertoy.com/view/4dS3Wd
// -------------------
// Precision-adjusted variations of https://www.shadertoy.com/view/4djSRW
float hash(float p) { p = fract(p * 0.011); p *= p + 7.5; p *= p + p; return fract(p); }
float hash(vec2 p) {vec3 p3 = fract(vec3(p.xyx) * 0.13); p3 += dot(p3, p3.yzx + 3.333); return fract((p3.x + p3.y) * p3.z); }

float morgan(float x) {
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(hash(i), hash(i + 1.0), u);
}

float morgan(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);

	// Four corners in 2D of a tile
	float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    // Simple 2D lerp using smoothstep envelope between the values.
	// return vec3(mix(mix(a, b, smoothstep(0.0, 1.0, f.x)),
	//			mix(c, d, smoothstep(0.0, 1.0, f.x)),
	//			smoothstep(0.0, 1.0, f.y)));

	// Same code, with the clamps in smoothstep and common subexpressions
	// optimized away.
    vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}


float morgan(vec3 x) {
    const vec3 step = vec3(110, 241, 171);

    vec3 i = floor(x);
    vec3 f = fract(x);
 
    // For performance, compute the base input to a 1D hash from the integer part of the argument and the 
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

float nmorgan(float x) {
  x *= 20.;
  return morgan(x);
}
float nmorgan(vec2 x) {
  x *= 20.;
  return morgan(x);
}
float nmorgan(vec2 x, float seed) {
  x *= 20.;
  seed *= 0.5;
  return morgan(vec3(x, seed));
}
// -------------------
// END Morgan Noises
// -------------------


// -------------------
// BEGIN http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
// -------------------
vec3 rgb2hsv(vec3 c)
{
    vec4 P = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = c.g < c.b ? vec4(c.bg, P.wz) : vec4(c.gb, P.xy);
    vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);
    // vec4 p = mix(vec4(c.bg, P.wz), vec4(c.gb, P.xy), step(c.b, c.g));
    // vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 P = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + P.xyz) * 6.0 - P.www);
    return c.z * mix(P.xxx, clamp(p - P.xxx, 0.0, 1.0), c.y);
}
// -------------------
// END HSV Conversion 
// -------------------

vec3 hsv(float h, float s, float v) {return hsv2rgb(vec3(h, s, v));}


// Noise Aux Functions
float random(vec2 uv, float seed) {
    const float a = 12.9898;
    const float b = 78.233;
    const float c = 43758.543123;
    return fract(sin(dot(uv, vec2(a, b)) + seed) * c);
}

#define OCTAVES 6
#define OCTAVES_F 6.0

// Simplex Variations
#define SIMPLEX_VARS_SCALE 1.0
float turbSimplex( vec3 p ) {
  p *= SIMPLEX_VARS_SCALE;
	float w = 100.0;
	float t = -.5;
  vec3 shift = vec3(100.);

	for (float f = 1.0 ; f < OCTAVES_F ; f++ ){
		float power = pow( 2.0, f );
		t += abs( simplex( power * p + shift ) / power );
	}

	return t;
}

float nturbSimplex(vec2 p, float seed) {
  p *= 2.;
  seed *= 0.08;
  return 1.-sat(-1.2*turbSimplex(vec3(p, seed)));
}

float nturb2Simplex(vec2 st2, float seed) {
  seed *= 0.04;
  vec3 st = vec3(st2, seed);
  st *= SIMPLEX_VARS_SCALE * 3.;
  float value = 0.0;
  float amplitude = .9;
  vec3 shift = vec3(200.);
  for (int i = 0; i < OCTAVES; i++) {
      value += amplitude * abs(simplex(st));
      st = st * 2. + shift;
      amplitude *= .5;
  }
  return clamp(value, -1., 1.);
}

float fbmSimplex(vec3 x) {
  x *= SIMPLEX_VARS_SCALE;
	float v = 0.0;
	float a = 1.;
	vec3 shift = vec3(300.);
	for (int i = 0; i < OCTAVES; ++i) {
		v += a * simplex(x);
		x = x * 2.0 + shift;
		a *= 0.5;
	}
	return clamp(v, -1., 1.);
}

float nfbmSimplex(vec2 x, float seed) {
  x *= 5.;
  seed *= 0.2;
  return norm(fbmSimplex(vec3(x, seed)));
}


// Morgan Variations
#define MORGAN_VARS_SCALE 2.5
float turbMorgan( vec3 p ) {
  p *= MORGAN_VARS_SCALE;
	float w = 100.0;
	float t = -.5;
  vec3 shift = vec3(0.);

	for (float f = 1.0 ; f < OCTAVES_F ; f++ ){
		float power = pow( 2.0, f );
		t += abs( denorm(morgan( power * p + shift )) / power );
	}

	return t;
}

float nturbMorgan(vec2 p, float seed) {
  p *= 2.;
  seed *= 0.05;
  return 1.-sat(-2.0*turbMorgan(vec3(p, seed)));
}

float nturb2Morgan(vec2 st2, float seed) {
  seed *= 0.03;
  vec3 st = vec3(st2, seed);
  st *= MORGAN_VARS_SCALE * 6.;
  float value = 0.0;
  float amplitude = 1.0;
  vec3 shift = vec3(0.);
  for (int i = 0; i < OCTAVES; i++) {
      value += amplitude * abs(denorm(morgan(st)));
      st = st * 2. + shift;
      amplitude *= .5;
  }
  return clamp(value, -1., 1.);
}

float fbmMorgan(vec3 x) {
  x *= MORGAN_VARS_SCALE;
	float v = 0.0;
	float a = .9;
	vec3 shift = vec3(0.);
	for (int i = 0; i < OCTAVES; ++i) {
		v += a * (denorm(morgan(x))+0.05);
		x = x * 2.0 + shift;
		a *= 0.5;
	}
	return clamp(v, -1., 1.);
}

float nfbmMorgan(vec2 x, float seed) {
  x *= 5.;
  seed *= 0.2;
  return norm(fbmMorgan(vec3(x, seed)));
}



// Cellular Variations
#define CELLULAR_VARS_SCALE 1.5
float turbCellular( vec3 p ) {
  p *= CELLULAR_VARS_SCALE;
	float w = 100.0;
	float t = -.5;
  vec3 shift = vec3(700.);

	for (float f = 1.0 ; f < OCTAVES_F ; f++ ){
		float power = pow( 2.0, f );
		t += abs( denorm(cellular( power * p + shift ).x) / power );
	}

	return t;
}

float nturbCellular(vec2 p, float seed) {
  p *= 2.;
  seed *= 0.07;
  return sat(0.85+2.*turbCellular(vec3(p, seed)));
}

float nturb2Cellular(vec2 st2, float seed) {
  seed *= 0.02;
  vec3 st = vec3(st2, seed);
  st *= CELLULAR_VARS_SCALE * 5.;
  float value = 0.0;
  float amplitude = 1.0;
  vec3 shift = vec3(800.);
  for (int i = 0; i < OCTAVES; i++) {
      value += amplitude * abs(denorm(cellular(st).x));
      st = st * 2. + shift;
      amplitude *= .5;
  }
  return clamp(value, -1., 1.);
}

float fbmCellular(vec3 x) {
  x *= CELLULAR_VARS_SCALE;
	float v = 0.0;
	float a = 1.;
	vec3 shift = vec3(900.);
	for (int i = 0; i < OCTAVES; ++i) {
		v += a * denorm(cellular(x).x);
		x = x * 2.0 + shift;
		a *= 0.5;
	}
	return clamp(v, -1., 1.);
}

float nfbmCellular(vec2 x, float seed) {
  seed *= 0.1;
  x *= 5.;
  return norm(fbmCellular(vec3(x, seed)));
}


vec2 rotate(vec2 v, float angle) {
  float s = sin(angle);
  float c = cos(angle);
  return mat2(c, -s, s, c) * v;
}

#ifndef SHADERTOY
uniform sampler2D iChannel0;
uniform float iTime;
uniform vec2 iResolution2D;
#define iResolution vec4(iResolution2D, 0., 0.)
out vec4 outColor;
#endif


// Main Code
#define SKY_COLOR1 vec3(0.0, 0.65, 1.0)
#define SKY_COLOR2 vec3(0.0, 0.25, 0.85)
#define WHITE vec3(1.0, 1.0, 1.0)

// How big the clouds will be
#define CROWDED_LEVEL 0.5

// Simplify cloud colors
// #define COLOR_LAYERS 3.0

// How much stronger each layer will be
#define COLOR_MIN 0.1
#define COLOR_MAX 1.0


vec3 drawDrySky(vec2 uv) {
    vec3 c = vec3(0.);
    c = mix(SKY_COLOR2, SKY_COLOR1, length(uv-vec2(0.0,1.0))/1.414);

    float t = iTime/60.0*3.0;
    float v = nturb2Simplex(vec2(uv.x-t, uv.y-t/4.0), t);
    float cutMin = 1.-CROWDED_LEVEL;
    float cutMax = 1.001;
    v = v * xstep(cutMin, cutMax, v);

    #ifdef COLOR_LAYERS
    // simplify clouds
    v = normrange(v, cutMin, cutMax);
    v = step(0.001, v) * mix(COLOR_MIN, COLOR_MAX, flatten(v, COLOR_LAYERS+1.));
    v = mix(cutMin, cutMax, v);
    v *= step(cutMin+0.01, v);
    #endif

    c = mix(c, WHITE, v);
    return c;
}

vec3 blurredSky(vec2 uv, float dist) {
    vec3 c1 = drawDrySky(uv + vec2(0.0 , 1.0 )*dist);
    vec3 c2 = drawDrySky(uv + vec2(1.0 , 0.0 )*dist);
    vec3 c3 = drawDrySky(uv + vec2(0.0 , -1.0)*dist);
    vec3 c4 = drawDrySky(uv + vec2(-1.0, 0.0 )*dist);
    return (c1+c2+c3+c4) / 4.0;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from -0.5 to 0.5)
    float mx = max(iResolution.x, iResolution.y);
    vec2 ct = iResolution.xy / mx / 2.0;
    vec2 uv = fragCoord/mx;
    uv -= ct;

    vec3 c = blurredSky(uv, 0.001);

    // Output to screen
    fragColor = vec4(c, 1.0);
}

#ifndef SHADERTOY
void main()
{
  mainImage(outColor, gl_FragCoord.xy);
}
#endif