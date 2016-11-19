static const char* glyphy_common_glsl = 
"/*\n"
" * Copyright 2012 Google, Inc. All Rights Reserved.\n"
" *\n"
" * Licensed under the Apache License, Version 2.0 (the \"License\");\n"
" * you may not use this file except in compliance with the License.\n"
" * You may obtain a copy of the License at\n"
" *\n"
" *     http://www.apache.org/licenses/LICENSE-2.0\n"
" *\n"
" * Unless required by applicable law or agreed to in writing, software\n"
" * distributed under the License is distributed on an \"AS IS\" BASIS,\n"
" * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
" * See the License for the specific language governing permissions and\n"
" * limitations under the License.\n"
" *\n"
" * Google Author(s): Behdad Esfahbod, Maysum Panju\n"
" */\n"
"\n"
"\n"
"#ifndef GLYPHY_INFINITY\n"
"#  define GLYPHY_INFINITY 1e9\n"
"#endif\n"
"#ifndef GLYPHY_EPSILON\n"
"#  define GLYPHY_EPSILON  1e-5\n"
"#endif\n"
"\n"
"#ifndef GLYPHY_RGBA\n"
"#  ifdef GLYPHY_BGRA\n"
"#    define GLYPHY_RGBA(v) glyphy_bgra (v)\n"
"#  else\n"
"#    define GLYPHY_RGBA(v) glyphy_rgba (v)\n"
"#  endif\n"
"#endif\n"
"\n"
"vec4\n"
"glyphy_rgba (const vec4 v)\n"
"{\n"
"  return v.rgba;\n"
"}\n"
"\n"
"vec4\n"
"glyphy_bgra (const vec4 v)\n"
"{\n"
"  return v.bgra;\n"
"}\n"
"\n"
"\n"
"struct glyphy_arc_t {\n"
"  vec2  p0;\n"
"  vec2  p1;\n"
"  float d;\n"
"};\n"
"\n"
"struct glyphy_arc_endpoint_t {\n"
"  /* Second arc endpoint */\n"
"  vec2  p;\n"
"  /* Infinity if this endpoint does not form an arc with the previous\n"
"   * endpoint.  Ie. a \"move_to\".  Test with glyphy_isinf().\n"
"   * Arc depth otherwise.  */\n"
"  float d;\n"
"};\n"
"\n"
"struct glyphy_arc_list_t {\n"
"  /* Number of endpoints in the list.\n"
"   * Will be zero if we're far away inside or outside, in which case side is set.\n"
"   * Will be -1 if this arc-list encodes a single line, in which case line_* are set. */\n"
"  int num_endpoints;\n"
"\n"
"  /* If num_endpoints is zero, this specifies whether we are inside (-1)\n"
"   * or outside (+1).  Otherwise we're unsure (0). */\n"
"  int side;\n"
"  /* Offset to the arc-endpoints from the beginning of the glyph blob */\n"
"  int offset;\n"
"\n"
"  /* A single line is all we care about.  It's right here. */\n"
"  float line_angle;\n"
"  float line_distance; /* From nominal glyph center */\n"
"};\n"
"\n"
"bool\n"
"glyphy_isinf (const float v)\n"
"{\n"
"  return abs (v) >= GLYPHY_INFINITY * .5;\n"
"}\n"
"\n"
"bool\n"
"glyphy_iszero (const float v)\n"
"{\n"
"  return abs (v) <= GLYPHY_EPSILON * 2.;\n"
"}\n"
"\n"
"vec2\n"
"glyphy_ortho (const vec2 v)\n"
"{\n"
"  return vec2 (-v.y, v.x);\n"
"}\n"
"\n"
"int\n"
"glyphy_float_to_byte (const float v)\n"
"{\n"
"  return int (v * (256. - GLYPHY_EPSILON));\n"
"}\n"
"\n"
"ivec4\n"
"glyphy_vec4_to_bytes (const vec4 v)\n"
"{\n"
"  return ivec4 (v * (256. - GLYPHY_EPSILON));\n"
"}\n"
"\n"
"ivec2\n"
"glyphy_float_to_two_nimbles (const float v)\n"
"{\n"
"  int f = glyphy_float_to_byte (v);\n"
"  return ivec2 (f / 16, int(mod (float(f), 16.)));\n"
"}\n"
"\n"
"/* returns tan (2 * atan (d)) */\n"
"float\n"
"glyphy_tan2atan (const float d)\n"
"{\n"
"  return 2. * d / (1. - d * d);\n"
"}\n"
"\n"
"glyphy_arc_endpoint_t\n"
"glyphy_arc_endpoint_decode (const vec4 v, const ivec2 nominal_size)\n"
"{\n"
"  vec2 p = (vec2 (glyphy_float_to_two_nimbles (v.a)) + v.gb) / 16.;\n"
"  float d = v.r;\n"
"  if (d == 0.)\n"
"    d = GLYPHY_INFINITY;\n"
"  else\n"
"#define GLYPHY_MAX_D .5\n"
"    d = float(glyphy_float_to_byte (d) - 128) * GLYPHY_MAX_D / 127.;\n"
"#undef GLYPHY_MAX_D\n"
"  return glyphy_arc_endpoint_t (p * vec2(nominal_size), d);\n"
"}\n"
"\n"
"vec2\n"
"glyphy_arc_center (const glyphy_arc_t a)\n"
"{\n"
"  return mix (a.p0, a.p1, .5) +\n"
"	 glyphy_ortho (a.p1 - a.p0) / (2. * glyphy_tan2atan (a.d));\n"
"}\n"
"\n"
"bool\n"
"glyphy_arc_wedge_contains (const glyphy_arc_t a, const vec2 p)\n"
"{\n"
"  float d2 = glyphy_tan2atan (a.d);\n"
"  return dot (p - a.p0, (a.p1 - a.p0) * mat2(1,  d2, -d2, 1)) >= 0. &&\n"
"	 dot (p - a.p1, (a.p1 - a.p0) * mat2(1, -d2,  d2, 1)) <= 0.;\n"
"}\n"
"\n"
"float\n"
"glyphy_arc_wedge_signed_dist_shallow (const glyphy_arc_t a, const vec2 p)\n"
"{\n"
"  vec2 v = normalize (a.p1 - a.p0);\n"
"  float line_d = dot (p - a.p0, glyphy_ortho (v));\n"
"  if (a.d == 0.)\n"
"    return line_d;\n"
"\n"
"  float d0 = dot ((p - a.p0), v);\n"
"  if (d0 < 0.)\n"
"    return sign (line_d) * distance (p, a.p0);\n"
"  float d1 = dot ((a.p1 - p), v);\n"
"  if (d1 < 0.)\n"
"    return sign (line_d) * distance (p, a.p1);\n"
"  float r = 2. * a.d * (d0 * d1) / (d0 + d1);\n"
"  if (r * line_d > 0.)\n"
"    return sign (line_d) * min (abs (line_d + r), min (distance (p, a.p0), distance (p, a.p1)));\n"
"  return line_d + r;\n"
"}\n"
"\n"
"float\n"
"glyphy_arc_wedge_signed_dist (const glyphy_arc_t a, const vec2 p)\n"
"{\n"
"  if (abs (a.d) <= .03)\n"
"    return glyphy_arc_wedge_signed_dist_shallow (a, p);\n"
"  vec2 c = glyphy_arc_center (a);\n"
"  return sign (a.d) * (distance (a.p0, c) - distance (p, c));\n"
"}\n"
"\n"
"float\n"
"glyphy_arc_extended_dist (const glyphy_arc_t a, const vec2 p)\n"
"{\n"
"  /* Note: this doesn't handle points inside the wedge. */\n"
"  vec2 m = mix (a.p0, a.p1, .5);\n"
"  float d2 = glyphy_tan2atan (a.d);\n"
"  if (dot (p - m, a.p1 - m) < 0.)\n"
"    return dot (p - a.p0, normalize ((a.p1 - a.p0) * mat2(+d2, -1, +1, +d2)));\n"
"  else\n"
"    return dot (p - a.p1, normalize ((a.p1 - a.p0) * mat2(-d2, -1, +1, -d2)));\n"
"}\n"
"\n"
"int\n"
"glyphy_arc_list_offset (const vec2 p, const ivec2 nominal_size)\n"
"{\n"
"  ivec2 cell = ivec2 (clamp (floor (p), vec2 (0.,0.), vec2(nominal_size - 1)));\n"
"  return cell.y * nominal_size.x + cell.x;\n"
"}\n"
"\n"
"glyphy_arc_list_t\n"
"glyphy_arc_list_decode (const vec4 v, const ivec2 nominal_size)\n"
"{\n"
"  glyphy_arc_list_t l;\n"
"  ivec4 iv = glyphy_vec4_to_bytes (v);\n"
"  l.side = 0; /* unsure */\n"
"  if (iv.r == 0) { /* arc-list encoded */\n"
"    l.offset = (iv.g * 256) + iv.b;\n"
"    l.num_endpoints = iv.a;\n"
"    if (l.num_endpoints == 255) {\n"
"      l.num_endpoints = 0;\n"
"      l.side = -1;\n"
"    } else if (l.num_endpoints == 0)\n"
"      l.side = +1;\n"
"  } else { /* single line encoded */\n"
"    l.num_endpoints = -1;\n"
"    l.line_distance = float(((iv.r - 128) * 256 + iv.g) - 0x4000) / float (0x1FFF)\n"
"                    * max (float (nominal_size.x), float (nominal_size.y));\n"
"    l.line_angle = float(-((iv.b * 256 + iv.a) - 0x8000)) / float (0x7FFF) * 3.14159265358979;\n"
"  }\n"
"  return l;\n"
"}\n"
"\n"
;
