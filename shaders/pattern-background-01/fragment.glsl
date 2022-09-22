#version 330
//#define SHADERTOY 1
// GITHUB: https://github.com/HugoLnx/shaders-laboratory/tree/master/shaders/pattern-background-01
// SHADERTOY: ???

//#define SEEDROLL 1

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
float flattenfull(float v, float layers) {
  return floor(v*layers)/(layers-1.);
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

vec3 compose(vec3 ocolor, float intensity, vec3 color) {
  intensity = sat(intensity);
  return mix(ocolor, color, intensity);
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

float rand(vec2 uv, float seed, float seedModifier){
  seed += seedModifier;
  uv *= seed+1937.71;
  return fract(sin(
    dot(uv, vec2(12.9898, 78.233)) + seed
  ) * 43758.5453);
}
float rand(float x, float seed, float seedModifier) {
  return rand(vec2(x, x+197.937), seed, seedModifier);
}
float circle(vec2 uv, vec2 center, float radius) {
  uv -= center;
  float dist = length(uv);
  return step(dist, radius);
}

float doline(vec2 uv, vec2 pt1, vec2 pt2, float linewidth) {
  vec2 v = pt2 - pt1;
  uv -= pt1;
  float vlen = length(v);
  vec2 vhead = v / vlen;
  float proj = dot(vhead, uv);
  vec2 vproj = vhead * proj;
  float toLine = length(uv - vproj);
  return (1.-step(linewidth, toLine)) * xstep(0., vlen, proj);
}

#define PIECOLORCOUNT 3
#define STRIPECOLORCOUNT 3
//#define ZOOMGRID 1
//#define MOBILE_SHADER_EDITOR 1

#ifdef MOBILE_SHADER_EDITOR
#define iResolution resolution
#define iTime time
#endif

vec3 composeLine(vec3 ocolor, vec2 uv, vec2 guv, float t, float appearRate, float lineSeedMod,
float seedModifier, vec3[STRIPECOLORCOUNT] stripecolors, vec2 linePt1, vec2 linePt2) {
  float lineWidth = 0.075;
  float maxTimeOffset = 100.;
  float frameDuration = 2.;
  float tRand = rand(guv, 771.339*lineSeedMod, seedModifier);
  float tOffset = (tRand-.5)*2. * maxTimeOffset;
  t += tOffset;
  float frameInx = floor(t / frameDuration);
  float frameTime = mod(t, frameDuration);
  float frameStart = normrange(frameTime, 0., 1.);
  float nFrameTime = normrange(frameTime, 0., frameDuration);
  float frameEnd = 1.-normrange(frameTime, frameDuration-1., frameDuration);
  float frameMod = frameInx * 372.297;

  float invRand = step(.5, rand(guv, 771.339*lineSeedMod, seedModifier+frameMod));
  vec2 pt1 = invRand == 1. ? linePt1 : linePt2;
  vec2 pt2 = invRand == 1. ? linePt2 : linePt1;
  float line = doline(uv, pt1, pt1+(pt2-pt1)*nFrameTime, lineWidth);

  float dRand = rand(guv/17.73, 117.994*lineSeedMod, seedModifier+frameMod);
  line *= step(1.-appearRate, dRand);
  line *= frameStart * frameEnd * .5;

  float colorRand = rand(guv, 171.274*lineSeedMod, seedModifier+frameMod);
  int colorInx = int(floor(colorRand*float(STRIPECOLORCOUNT)));
  vec3 stripecolor = stripecolors[colorInx];
  vec3 color = compose(ocolor, line, stripecolor);
  return color;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  float mx = max(iResolution.x, iResolution.y);
  vec2 ct = iResolution.xy / mx / 2.0;
  vec2 uv = fragCoord/mx - ct;
#ifndef MOBILE_SHADER_EDITOR
  uv /= 2.5;
#endif
  vec2 ouv = uv;
  float t = iTime;
#ifdef SEEDROLL
  float seedModifier = floor(t*.33);
#else
  float seedModifier = 0.;
#endif

#ifdef MOBILE_SHADER_EDITOR
  float radius = .35;
  float minRadiusVar = .35;
#else
  float radius = .45;
  float minRadiusVar = .5;
#endif
  float maxCentervar = (.5-radius)*.5*0.95;
#ifdef ZOOMGRID
  float gridScale = 15.;
#else
  float gridScale = 40.;
#endif
  vec3[PIECOLORCOUNT] piecolors;
  piecolors[0] = vec3(1., .4, .4);
  piecolors[1] = vec3(1., .4, .7);
  piecolors[2] = vec3(1., .7, .4);

  vec3[STRIPECOLORCOUNT] stripecolors;
  float lo = 0.75;
  stripecolors[0] = vec3(lo, lo, 1.);
  stripecolors[1] = vec3(1., lo, lo);
  stripecolors[2] = vec3(lo, 1., lo);

  uv *= gridScale;
  vec2 guv = floor(uv);
  uv = fract(uv);
  uv -= .5;
  vec2 center = vec2(.0);

  float angsize = flattenfull(rand(guv, 137.791, seedModifier), 3.);
  angsize = mix(PI/4.*3., PI/4.*7., angsize);
  float rotlayers = 5.;
  float pierot = flattenfull(rand(guv, 737.197, seedModifier), 4.);
  pierot = mix(PI/2., TWO_PI, pierot);

  int piecolorinx = int(floor(
    rand(guv, 397.917, seedModifier)*float(PIECOLORCOUNT)
  ));

  float radiusvar = flattenfull(rand(guv, 587.173, seedModifier), 3.);
  radiusvar = mix(minRadiusVar, 1., radiusvar);
  //radiusvar = 1.;

  vec2 centervar = vec2(
    flattenfull(rand(guv.xy, 345.123, seedModifier), 3.),
    flattenfull(rand(guv.yx, 354.132, seedModifier), 3.)
  );
  centervar = (centervar-.5)*2.;
  centervar *= maxCentervar;
  //center += centervar; 

  float pie = circle(uv, center, radius*radiusvar);
  pie *= stepang(uv, center, 0., angsize, pierot);

  float dotradius = flattenfull(rand(guv, 897.217, seedModifier), 3.);
  dotradius = mix(.03, .15, dotradius);
  float hasdot = step(.96, rand(guv, 227.994, seedModifier));
  float dotted = hasdot*circle(uv, center, dotradius);


  float npie = 1.-pie;
  float ndotted = 1.-dotted;

  vec3 color = vec3(0.);


  vec2 diaguv = ouv;
  diaguv *= gridScale;
  diaguv -= .5;
  vec2 diagGuv = floor(diaguv);
  diaguv = fract(diaguv);
  diaguv -= .5;

  vec2 xuv = ouv;
  xuv *= gridScale;
  xuv.x -= .5;
  vec2 xGuv = floor(xuv);
  xuv = fract(xuv);
  xuv -= .5;

  vec2 yuv = ouv;
  yuv *= gridScale;
  yuv.y -= .5;
  vec2 yGuv = floor(yuv);
  yuv = fract(yuv);
  yuv -= .5;
  float diagAppearRate = .4;
  float axisAppearRate = .3;
  color = composeLine(color, diaguv, diagGuv, t, diagAppearRate, 0.  , seedModifier, stripecolors, vec2(-.5), vec2(.5));
  color = composeLine(color, diaguv, diagGuv, t, diagAppearRate, 132., seedModifier, stripecolors, vec2(-.5, .5), vec2(.5, -.5));
  color = composeLine(color, xuv, xGuv, t, axisAppearRate, 212., seedModifier, stripecolors, vec2(-.5, .0), vec2(.5, .0) );
  color = composeLine(color, yuv, yGuv, t, axisAppearRate, 398., seedModifier, stripecolors, vec2(.0, .5), vec2(.0, -.5) );

  color = compose(color, ndotted*pie, piecolors[piecolorinx]);

  fragColor = vec4(color, 1.0);
}

#ifndef SHADERTOY
void main()
{
  mainImage(outColor, gl_FragCoord.xy);
}
#endif