static const char* glyphy_sdf_glsl = 
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
"#ifndef GLYPHY_TEXTURE1D_FUNC\n"
"#define GLYPHY_TEXTURE1D_FUNC glyphy_texture1D_func\n"
"#endif\n"
"#ifndef GLYPHY_TEXTURE1D_EXTRA_DECLS\n"
"#define GLYPHY_TEXTURE1D_EXTRA_DECLS\n"
"#endif\n"
"#ifndef GLYPHY_TEXTURE1D_EXTRA_ARGS\n"
"#define GLYPHY_TEXTURE1D_EXTRA_ARGS\n"
"#endif\n"
"\n"
"#ifndef GLYPHY_SDF_TEXTURE1D_FUNC\n"
"#define GLYPHY_SDF_TEXTURE1D_FUNC GLYPHY_TEXTURE1D_FUNC\n"
"#endif\n"
"#ifndef GLYPHY_SDF_TEXTURE1D_EXTRA_DECLS\n"
"#define GLYPHY_SDF_TEXTURE1D_EXTRA_DECLS GLYPHY_TEXTURE1D_EXTRA_DECLS\n"
"#endif\n"
"#ifndef GLYPHY_SDF_TEXTURE1D_EXTRA_ARGS\n"
"#define GLYPHY_SDF_TEXTURE1D_EXTRA_ARGS GLYPHY_TEXTURE1D_EXTRA_ARGS\n"
"#endif\n"
"#ifndef GLYPHY_SDF_TEXTURE1D\n"
"#define GLYPHY_SDF_TEXTURE1D(offset) GLYPHY_RGBA(GLYPHY_SDF_TEXTURE1D_FUNC (offset GLYPHY_TEXTURE1D_EXTRA_ARGS))\n"
"#endif\n"
"\n"
"#ifndef GLYPHY_MAX_NUM_ENDPOINTS\n"
"#define GLYPHY_MAX_NUM_ENDPOINTS 32\n"
"#endif\n"
"\n"
"glyphy_arc_list_t\n"
"glyphy_arc_list (const vec2 p, const ivec2 nominal_size GLYPHY_SDF_TEXTURE1D_EXTRA_DECLS)\n"
"{\n"
"  int cell_offset = glyphy_arc_list_offset (p, nominal_size);\n"
"  vec4 arc_list_data = GLYPHY_SDF_TEXTURE1D (cell_offset);\n"
"  return glyphy_arc_list_decode (arc_list_data, nominal_size);\n"
"}\n"
"\n"
"float\n"
"glyphy_sdf (const vec2 p, const ivec2 nominal_size GLYPHY_SDF_TEXTURE1D_EXTRA_DECLS)\n"
"{\n"
"  glyphy_arc_list_t arc_list = glyphy_arc_list (p, nominal_size  GLYPHY_SDF_TEXTURE1D_EXTRA_ARGS);\n"
"\n"
"  /* Short-circuits */\n"
"  if (arc_list.num_endpoints == 0) {\n"
"    /* far-away cell */\n"
"    return GLYPHY_INFINITY * float(arc_list.side);\n"
"  } if (arc_list.num_endpoints == -1) {\n"
"    /* single-line */\n"
"    float angle = arc_list.line_angle;\n"
"    vec2 n = vec2 (cos (angle), sin (angle));\n"
"    return dot (p - (vec2(nominal_size) * .5), n) - arc_list.line_distance;\n"
"  }\n"
"\n"
"  float side = float(arc_list.side);\n"
"  float min_dist = GLYPHY_INFINITY;\n"
"  glyphy_arc_t closest_arc;\n"
"\n"
"  glyphy_arc_endpoint_t endpoint_prev, endpoint;\n"
"  endpoint_prev = glyphy_arc_endpoint_decode (GLYPHY_SDF_TEXTURE1D (arc_list.offset), nominal_size);\n"
"  for (int i = 1; i < GLYPHY_MAX_NUM_ENDPOINTS; i++)\n"
"  {\n"
"    if (i >= arc_list.num_endpoints) {\n"
"      break;\n"
"    }\n"
"    endpoint = glyphy_arc_endpoint_decode (GLYPHY_SDF_TEXTURE1D (arc_list.offset + i), nominal_size);\n"
"    glyphy_arc_t a = glyphy_arc_t (endpoint_prev.p, endpoint.p, endpoint.d);\n"
"    endpoint_prev = endpoint;\n"
"    if (glyphy_isinf (a.d)) continue;\n"
"\n"
"    if (glyphy_arc_wedge_contains (a, p))\n"
"    {\n"
"      float sdist = glyphy_arc_wedge_signed_dist (a, p);\n"
"      float udist = abs (sdist) * (1. - GLYPHY_EPSILON);\n"
"      if (udist <= min_dist) {\n"
"	min_dist = udist;\n"
"	side = sdist <= 0. ? -1. : +1.;\n"
"      }\n"
"    } else {\n"
"      float udist = min (distance (p, a.p0), distance (p, a.p1));\n"
"      if (udist < min_dist) {\n"
"	min_dist = udist;\n"
"	side = 0.; /* unsure */\n"
"	closest_arc = a;\n"
"      } else if (side == 0. && udist == min_dist) {\n"
"	/* If this new distance is the same as the current minimum,\n"
"	 * compare extended distances.  Take the sign from the arc\n"
"	 * with larger extended distance. */\n"
"	float old_ext_dist = glyphy_arc_extended_dist (closest_arc, p);\n"
"	float new_ext_dist = glyphy_arc_extended_dist (a, p);\n"
"\n"
"	float ext_dist = abs (new_ext_dist) <= abs (old_ext_dist) ?\n"
"			 old_ext_dist : new_ext_dist;\n"
"\n"
"#ifdef GLYPHY_SDF_PSEUDO_DISTANCE\n"
"	/* For emboldening and stuff: */\n"
"	min_dist = abs (ext_dist);\n"
"#endif\n"
"	side = sign (ext_dist);\n"
"      }\n"
"    }\n"
"  }\n"
"\n"
"  if (side == 0.) {\n"
"    // Technically speaking this should not happen, but it does.  So try to fix it.\n"
"    float ext_dist = glyphy_arc_extended_dist (closest_arc, p);\n"
"    side = sign (ext_dist);\n"
"  }\n"
"\n"
"  return min_dist * side;\n"
"}\n"
"\n"
"float\n"
"glyphy_point_dist (const vec2 p, const ivec2 nominal_size GLYPHY_SDF_TEXTURE1D_EXTRA_DECLS)\n"
"{\n"
"  glyphy_arc_list_t arc_list = glyphy_arc_list (p, nominal_size  GLYPHY_SDF_TEXTURE1D_EXTRA_ARGS);\n"
"\n"
"  float side = float(arc_list.side);\n"
"  float min_dist = GLYPHY_INFINITY;\n"
"\n"
"  if (arc_list.num_endpoints == 0)\n"
"    return min_dist;\n"
"\n"
"  glyphy_arc_endpoint_t endpoint;\n"
"  for (int i = 0; i < GLYPHY_MAX_NUM_ENDPOINTS; i++)\n"
"  {\n"
"    if (i >= arc_list.num_endpoints) {\n"
"      break;\n"
"    }\n"
"    endpoint = glyphy_arc_endpoint_decode (GLYPHY_SDF_TEXTURE1D (arc_list.offset + i), nominal_size);\n"
"    if (glyphy_isinf (endpoint.d)) continue;\n"
"    min_dist = min (min_dist, distance (p, endpoint.p));\n"
"  }\n"
"  return min_dist;\n"
"}\n"
"\n"
;
