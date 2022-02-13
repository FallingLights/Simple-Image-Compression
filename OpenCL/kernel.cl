__kernel void labelPixels(__global byte_t *data, __global double *centers, __global int *labels, __global double *dists, int n_px, int n_ch, int n_clus) {

    int idx = get_global_id(0);

    int px, ch, k;
    int min_k, tmp_changes = 0;
    double dist, min_dist, tmp;

    px = idx;
    if (px < n_px) {
        min_dist = DBL_MAX;

        for (k = 0; k < n_clus; k++) {
            dist = 0;

            for (ch = 0; ch < n_ch; ch++) {
                tmp = (double)(data[px * n_ch + ch] - centers[k * n_ch + ch]);
                dist += tmp * tmp;
            }

            if (dist < min_dist) {
                min_dist = dist;
                min_k = k;
            }
        }

        dists[px] = min_dist;

        if (labels[px] != min_k) {
            labels[px] = min_k;
            tmp_changes = 1;
        }
    }
}
