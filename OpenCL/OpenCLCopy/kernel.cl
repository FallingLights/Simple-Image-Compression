
int findBestCentroidFor(int r,int g, int b,__global unsigned char* centroids,int K)
{
    double min = 1.7976931348623157E+308;
    int min_i = 0;
    double razdalja = 0.0f;
    // int min = 2147483647;
    // int min_i = 0;
    // int razdalja = 0;


    for (int i = 0; i < K; i++)
    {
        int index = i * 3;
        int r2 = (centroids[ index + 0 ] - r) * (centroids[ index + 0 ] - r );
        int g2 = (centroids[ index + 1 ] - g) * (centroids[ index + 1 ] - g );
        int b2 = (centroids[ index + 2 ] - b) * (centroids[ index + 2 ] - b );

        //razdalja = abs(r2) + abs(g2) + abs(b2); //sqrt((double)(r2 + g2 + b2));
        razdalja = sqrt((double)(r2 + g2 + b2));

        if (razdalja < min)
        {
            min = razdalja;
            min_i = i;
        }
    }
    return min_i;
}
__kernel void kMeansAlgorithm(
                        __global unsigned char* image, 
                        __global unsigned char* centroids,
                        __global unsigned int* centroids_Gsums,
                        __global unsigned int* centroids_Gpopularity,
                        __local unsigned int* centroids_Lsums,
                        __local unsigned int* centroids_Lpopularity,
                        int height,int width,
                        int numOfColors, int itter
                        )
{
    int gid_i = get_global_id(0);
    int lid_i = get_local_id(0);
    int local_size = get_local_size(0);
    int groupId_i = get_group_id(0);

    int index = gid_i;      
    int localIndex = lid_i;  

    int K = numOfColors;
    int ITTERATIONS = itter;
    
    int r = image[index * 4 + 0];
    int g = image[index * 4 + 1];
    int b = image[index * 4 + 2];
    
    int bestCentroid = 0;
    int localValue = 0;

    
    // za tocko (gid_i, gid_j) najdi najboljsi centroid
    if(index < ((width * height) - 1))
    {
        for(int i = 0; i < ITTERATIONS; i++)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            //reset lokalne tabele vsota za delovno skupino
            //izvede naj le nit v delovni skupini ki ima indeks manjsi od K * 3
            if (localIndex < K )
            {
                centroids_Lsums[localIndex * 3 + 0] = 0;
                centroids_Lsums[localIndex * 3 + 1] = 0;
                centroids_Lsums[localIndex * 3 + 2] = 0;
            } 
            
            barrier(CLK_LOCAL_MEM_FENCE);
            //za pixel ki pripada tej niti najdi najboljsi centroid v globalnem bufferju centroidov
            bestCentroid = findBestCentroidFor(r, g, b, centroids,K);

            barrier(CLK_LOCAL_MEM_FENCE);
            //reset lokalne tabele ki belezi koliko tock pripada kateremu centroidu
            if (localIndex < K) centroids_Lpopularity[localIndex] = 0;  
            
            barrier(CLK_LOCAL_MEM_FENCE);
            //povecaj pripadnost centroidu z indexom bestCentroid za 1 - v LOKALNI tabeli
            atomic_inc(centroids_Lpopularity + bestCentroid);
            //pristej vrednost vseh treh kanalov tega pixla v LOKALNO tabelo vsot
            atomic_add(centroids_Lsums + bestCentroid * 3 + 0, r);
            atomic_add(centroids_Lsums + bestCentroid * 3 + 1, g);
            atomic_add(centroids_Lsums + bestCentroid * 3 + 2, b);

            barrier(CLK_LOCAL_MEM_FENCE);
            //tukaj se zacne delat na globalni ravni
            //vrednosti v GLOBALNI tabeli vsot in tabeli popularnosti nastavit na 0
            //to naj pocenjo nit z ustreznim indexom
            if(index < K * 3)
            {
                centroids_Gsums[index] = 0;
                if (index < K) centroids_Gpopularity[index] = 0;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            //naj prvih K * 3 niti znotraj delovne skupine vsaka pristeje eno vsoto v globalno tabelo
            //prav tako naj pristeje tudi popularnost
            if (localIndex < K * 3) 
            {
                localValue = centroids_Lsums[localIndex];
                atomic_add(centroids_Gsums + localIndex, localValue);
                
                localValue = centroids_Lpopularity[localIndex];
                if (localIndex < K) atomic_add(centroids_Gpopularity + localIndex,localValue);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            //posodobi centroide
            //ce gre za zadnjo iteracijo ne posodabljaj centroidov
            if(index < K * 3 && i < ITTERATIONS - 1) 
            {
                if (centroids_Gpopularity[index/3] > 0) centroids[index] = centroids_Gsums[index] / centroids_Gpopularity[index/3];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //set image values
        image[index * 4 + 0] = centroids[bestCentroid * 3 + 0];
        image[index * 4 + 1] = centroids[bestCentroid * 3 + 1];
        image[index * 4 + 2] = centroids[bestCentroid * 3 + 2];
    }
}

