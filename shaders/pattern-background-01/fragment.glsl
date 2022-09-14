#version 330
//#define SHADERTOY 1
// GITHUB: https://github.com/HugoLnx/shaders-laboratory/tree/master/shaders/pattern-background-01
// SHADERTOY: ???

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
vec3 togrey(vec3 c) {
  return vec3((c.r+c.g+c.b)/3.);
}
float normrange(float v, float minV, float maxV) { return sat((v-minV)/(maxV-minV)); }
float xnormrange(float v, float minV, float maxV) { return normrange(v, minV, maxV) * xstep(minV, maxV, v); }
vec3 mix3(vec3 cMin, vec3 cMid, vec3 cMax, float t) {
  float t1 = normrange(t, -1., 0.);
  float t2 = normrange(t, 0., 1.);
  vec3 c = mix(cMin, cMid, t1);
  c = mix(c, cMax, t2);
  return c;
}

vec2 rotate(vec2 v, float angle) {
  float s = sin(angle);
  float c = cos(angle);
  return mat2(c, -s, s, c) * v;
}

float stepang(vec2 uv, vec2 center,
float minang, float maxang, float rot) {
  uv -= center;
  float angle = atan(uv.y, uv.x);
  angle = mod(angle + TWO_PI*10., TWO_PI);
  minang = mod(minang + rot + TWO_PI*10., TWO_PI);
  maxang = mod(maxang + rot + TWO_PI*10., TWO_PI);
  float iss1 = step(minang, maxang);
  float niss1 = 1. - iss1;
  float s1 = iss1*xstep(minang, maxang, angle);
  float s2 = niss1*xstep(minang, TWO_PI, angle);
  float s3 = niss1*xstep(0., maxang, angle);
  float s4 = xstep(.0, .0001, abs(minang-maxang));
  return sat(s1+s2+s3+s4);
}

#define RED vec3(1., 0., 0.)
#define GRE vec3(0., 1., 0.)
#define BLU vec3(0., 0., 1.)
#define WRED vec3(1., .8, .8)
#define WGRE vec3(.8, 1., .8)
#define WBLU vec3(.8, .8, 1.)
#define BLU2 vec3(0.35, 0.5, 1.)
#define PUR vec3(1., 0., 1.)
#define YEL vec3(1., 1., 0.)
#define CYA vec3(0., 1., 1.)
#define WHI vec3(1.)
#define BLA vec3(0.)
#define BLANK vec3(0.35, 0., 0.35)

#ifndef SHADERTOY
uniform sampler2D iChannel0;
uniform float iTime;
uniform vec2 iResolution2D;
#define iResolution vec4(iResolution2D, 0., 0.)
out vec4 outColor;
#endif

float rand(vec2 uv, float seed){
  uv *= seed+1937.71;
  return fract(sin(
    dot(uv, vec2(12.9898, 78.233)) + seed
  ) * 43758.5453);
}
float rand(float x, float seed) {
  return rand(vec2(x, x+197.937), seed);
}
float circle(vec2 uv, vec2 center, float radius) {
  uv -= center;
  float dist = length(uv);
  return step(dist, radius);
}

#define PIECOLORCOUNT 3

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  // // Normalized pixel coordinates (from -0.5 to 0.5)
  // float mx = max(iResolution.x, iResolution.y);
  // vec2 ct = iResolution.xy / mx / 2.0;
  // vec2 uv = fragCoord/mx - ct;
  // vec2 uv2 = fragCoord / iResolution.xy - .5;

  // float t = iTime;
  // vec3 c = YEL;

  // // Output to screen
  // fragColor = vec4(c, 1.0);

  float mx = max(iResolution.x, iResolution.y);
  vec2 ct = iResolution.xy / mx / 2.0;
  vec2 uv = fragCoord/mx - ct;
  float t = iTime;

  float radius = .35;
  float gridScale = 40.;
  vec3[PIECOLORCOUNT] piecolors;
  piecolors[0] = vec3(1., .4, .4);
  piecolors[1] = vec3(1., .4, .7);
  piecolors[2] = vec3(1., .7, .4);

  vec2 guv = floor(uv*gridScale);
  uv = fract(uv*gridScale);
  uv -= .5;
  vec2 center = vec2(.0);

  float angsize = flatten(rand(guv, 137.791), 3.);
  angsize = mix(PI/4.*3., PI/4.*7., angsize);
  float rotlayers = 5.;
  float pierot = flatten(rand(guv, 737.197), 4.);
  pierot = mix(PI/2., TWO_PI, pierot);

  int piecolorinx = int(floor(
    rand(guv, 397.917)*float(PIECOLORCOUNT)
  ));

  float radiusvar = flatten(rand(guv, 587.173), 3.);
  radiusvar = mix(.35, 1., radiusvar);

  vec2 centervar = vec2(
    flatten(rand(guv.xy, 345.123), 3.),
    flatten(rand(guv.yx, 354.132), 3.)
  );
  centervar = (centervar-.5)*2.;
  centervar *= .08;
  center += centervar;

  float pie = circle(uv, center, radius*radiusvar);
  pie *= stepang(uv, center, 0., angsize, pierot);

  float dotradius = flatten(rand(guv, 897.217), 3.);
  dotradius = mix(.03, .15, dotradius);
  float hasdot = step(.96, rand(guv, 227.994));
  float dotted = hasdot*circle(uv, center, dotradius);


  float npie = 1.-pie;
  float ndotted = 1.-dotted;

  vec3 color = vec3(0.);
  color += ndotted*pie * piecolors[piecolorinx];


  fragColor = vec4(color, 1.0);
}

#ifndef SHADERTOY
void main()
{
  mainImage(outColor, gl_FragCoord.xy);
}
#endif