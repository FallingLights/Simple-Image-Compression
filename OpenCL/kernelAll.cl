__kernel void labelPixels(read_only unsigned char *data, 
                          __global double *centers, 
                          __global int *labels, 
                          __global double *dists,
                          __global int *changed,
                          int n_px, 
                          int n_ch, 
                          int n_clus) {

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

__kernel void update_centers(
            __global unsigned char *data, 
            __global double *centers, 
            __global int *labels, 
            __global double *dists,
            __global ulong *global_sum,
            __global uint *global_count,
            __global uchar *assignments,
            __local double *sum,
            __local uint *count,
            int n_px, 
            int n_ch, 
            int n_clus) {
    const int window = 8;
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    int px, ch, k;
    int *counts;
    int min_k, far_px;
    double max_dist;

    sum[lid] = 0;
    count[lid] = 0;
    global_sum[lid] = 0;
    global_count[lid] = 0;

    const int item_offset = gid*window;

    //Calculating sums
    for (int px = item_offset; px < (item_offset + window) && px < n_px; x++) {
		 if (lid == labels[px]) {
            for (ch = 0; ch < n_ch; ch++) {
			    sum[lid * n_ch + ch] += data[px * n_ch + ch];
		    }
            count[lid]++;
	    }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    //Move to global memory
    for (ch = 0; ch < n_ch; ch++) {
        if (sum[lid * n_ch *ch] != 0) atom_add(global_sum + lid * n_ch + ch, (int)(sum[lid * n_ch + ch]))
    }

    if (count[lid] != 0) atomic_add(global_count + lid, count[lid]);

    //Means
    if(gid == 0){
        for (k = 0; k < n_clus; k++) {
            if (global_count[k] == 0) continue;
            for (ch = 0; ch < n_ch; ch++) {
                centers[k * n_ch + ch] = global_sum[k * n_ch + ch] / global_count[k];
            }
        }
    }
}