// Takes an encoding of a flat array of faces (static indices)
// that is assumed  to be the length of the # of triangles in
// the model * 3. Each vertex then has 3 coordinates.
double surface_area(double *vertices, int *faces, int n_faces)
{
  double total = 0.0;
  // Please, please vectorize. (It does!)
  for (int i = 0; i < n_faces; i += 3)
  {
    int v0 = faces[i + 0];
    int v1 = faces[i + 1];
    int v2 = faces[i + 2];

    double v1x = vertices[v0 * 3 + 0];
    double v1y = vertices[v0 * 3 + 1];
    double v1z = vertices[v0 * 3 + 2];
    double v2x = vertices[v1 * 3 + 0];
    double v2y = vertices[v1 * 3 + 1];
    double v2z = vertices[v1 * 3 + 2];
    double v3x = vertices[v2 * 3 + 0];
    double v3y = vertices[v2 * 3 + 1];
    double v3z = vertices[v2 * 3 + 2];


    double dx1 = v2x - v1x;
    double dy1 = v2y - v1y;
    double dz1 = v2z - v1z;
    double a = (dx1 * dx1) + (dy1 * dy1) + (dz1 * dz1);

    double dx2 = v3x - v1x;
    double dy2 = v3y - v1y;
    double dz2 = v3z - v1z;
    double b = (dx2 * dx2) + (dy2 * dy2) + (dz2 * dz2);

    double dx3 = v3x - v2x;
    double dy3 = v3y - v2y;
    double dz3 = v3z - v2z;
    double c = (dx3 * dx3) + (dy3 * dy3) + (dz3 * dz3);

    double A = ((2.0*a*b) + (2.0*b*c) + (2.0*c*a) - (a*a) - (b*b) - (c*c)) / 16.0;
    total += sqrt(A);
  }
  return total;
}

double volume(double *vertices, int *faces, int n_faces)
{
  double total = 0.0;
  // Please, please vectorize. (It does!)
  for (int i = 0; i < n_faces; i += 3)
  {
    int v0 = faces[i + 0];
    int v1 = faces[i + 1];
    int v2 = faces[i + 2];
    double v1x = vertices[v0 * 3 + 0];
    double v1y = vertices[v0 * 3 + 1];
    double v1z = vertices[v0 * 3 + 2];
    double v2x = vertices[v1 * 3 + 0];
    double v2y = vertices[v1 * 3 + 1];
    double v2z = vertices[v1 * 3 + 2];
    double v3x = vertices[v2 * 3 + 0];
    double v3y = vertices[v2 * 3 + 1];
    double v3z = vertices[v2 * 3 + 2];

    double v321 = v3x * v2y * v1z;
    double v231 = v2x * v3y * v1z;
    double v312 = v3x * v1y * v2z;
    double v132 = v1x * v3y * v2z;
    double v213 = v2x * v1y * v3z;
    double v123 = v1x * v2y * v3z;

    double V = (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123);
    total += V;
  }
  return total;
}

void center_of_mass(double *vertices, int *faces, int n_faces, double *center)
{
  center[0] = 0;
  center[1] = 0;
  center[2] = 0;
  double total_vol = 0.0;
  for (int i = 0; i < n_faces; i+= 3)
  {
    // Could factor out tet volume with volume function
    int v0 = faces[i + 0];
    int v1 = faces[i + 1];
    int v2 = faces[i + 2];
    double v1x = vertices[v0 * 3 + 0];
    double v1y = vertices[v0 * 3 + 1];
    double v1z = vertices[v0 * 3 + 2];
    double v2x = vertices[v1 * 3 + 0];
    double v2y = vertices[v1 * 3 + 1];
    double v2z = vertices[v1 * 3 + 2];
    double v3x = vertices[v2 * 3 + 0];
    double v3y = vertices[v2 * 3 + 1];
    double v3z = vertices[v2 * 3 + 2];

    double v321 = v3x * v2y * v1z;
    double v231 = v2x * v3y * v1z;
    double v312 = v3x * v1y * v2z;
    double v132 = v1x * v3y * v2z;
    double v213 = v2x * v1y * v3z;
    double v123 = v1x * v2y * v3z;

    double V = (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123);
    total_vol += V;
    center[0] += V * ((v1x + v2x + v3x) / 4);
    center[1] += V * ((v1y + v2y + v3y) / 4);
    center[2] += V * ((v1z + v2z + v3z) / 4);
  }
  center[0] /= total_vol;
  center[1] /= total_vol;
  center[2] /= total_vol;
}


double edit(double* vertices, double* target_vertices, int* selected_vertices, int num_selected) {
  double total = 0.0;
  for (int i = 0; i < num_selected; i++) {
    int vtx = selected_vertices[i] * 3;
    double dx = vertices[vtx+0] - target_vertices[vtx+0];
    double dy = vertices[vtx+1] - target_vertices[vtx+1];
    double dz = vertices[vtx+2] - target_vertices[vtx+2];
    total += (dx*dx) + (dy*dy) + (dz*dz);
  }
  return total;
}