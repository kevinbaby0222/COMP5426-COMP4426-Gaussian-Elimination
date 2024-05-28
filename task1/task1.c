#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

typedef struct
{
    int origin_row;
    int target_row;
} SwapInfo;

void print_matrix(double **T, int rows, int cols);

int test(double **t1, double **t2, int rows);

void gepp_origin(double **A, int n);

int main(int argc, char **argv)
{
    double *a0;
    double **a; // 2D matrix for sequential computation[N][N]
    double *A0; 
    double **A; // 2D matrix for MPI computation       [N][N]
    int N, b; // input size
    int indk;
    double c, amax;
    int i, j, k;
    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;

    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (argc != 3)
    {
        if (myid == 0)
            fprintf(stderr, "Usage: mpiexec -n [numprocs] task1.exe [N] [b]\n");
        MPI_Finalize();
        return 0;       
            
    }
    N = atoi(argv[1]);
    b = atoi(argv[2]);
    if (myid == 0)
    printf("The matrix size:  %d * %d, block size : %d \n", N, N, b);


    if (myid == 0)
    {
        /*** Allocate contiguous memory for 2D matrices ***/
        a0 = (double *)malloc(N * N * sizeof(double));
        a = (double **)malloc(N * sizeof(double *));
        A0 = (double *)malloc(N * N * sizeof(double));
        A = (double **)malloc(N * sizeof(double *));
        for (i = 0; i < N; i++)
            a[i] = a0 + i * N;
        for (i = 0; i < N; i++)
            A[i] = A0 + i * N;

        srand(time(0));
        printf("Creating and initializing matrices...\n\n");
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                a[i][j] = (double)rand() / RAND_MAX;
                A[i][j] = a[i][j];
            }
        }
        gettimeofday(&start_time, 0);
        gepp_origin(a, N);
        gettimeofday(&end_time, 0);
        seconds = end_time.tv_sec - start_time.tv_sec;
        microseconds = end_time.tv_usec - start_time.tv_usec;

        elapsed = seconds + 1e-6 * microseconds;
        printf("Sequential calculation time: %f\n\n", elapsed);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Parameters for MPI programming

    double **AK;  // 2D local matrix for every process [N][K]
    double **AW;  // 2D Working matrix                 [N][b] 
    double *AK0, *AW0;
    double LL[b][b]; // 2D LL matrix                   [b][b]
    int Nb, bm, q, r, kb, K; // For load balancing
    int ib, ib_green, jb, jb_0, l, working_block;

    /* Load balancing in Column Block Cyclicing */
    Nb = N / b;         // The total number of column blocks
    bm = b * N;         // Column block size
    q = Nb / numprocs;  // Each process gets at least q column blocks
    r = Nb % numprocs;  // Remaining column blocks

    if (myid < r)       // One more column block for each of the first r processes
        kb = q + 1;
    else
        kb = q;

    K = kb * b;         // Number of columns in submatrix, may be LLfferent for LLfferent processes

    AK0 = (double *)malloc(N * K * sizeof(double));
    AK = (double **)malloc(N * sizeof(double *));
    AW0 = (double *)malloc(N * b * sizeof(double));
    AW = (double **)malloc(N * sizeof(double *));
    if (AK == NULL)
    {
        fprintf(stderr, "Some processors don't have matrix to calculate. Increase N.");
        exit(1);
    }
    for (i = 0; i < N; i++)
    {
        AK[i] = AK0 + i * K;
        for (j = 0; j < K; j++)
            AK[i][j] = 0.0;
    }            

    for (i = 0; i < N; i++)
    {
        AW[i] = AW0 + i * b;
        for (j = 0; j < b; j++)
            AW[i][j] = 0.0;
    }

    MPI_Datatype column_block_global;
    MPI_Type_vector(N, b, N, MPI_DOUBLE, &column_block_global);
    MPI_Type_commit(&column_block_global);

    MPI_Datatype column_block_local;
    MPI_Type_vector(N, b, K, MPI_DOUBLE, &column_block_local);
    MPI_Type_commit(&column_block_local);
    
    SwapInfo swapInfo[b];
    MPI_Datatype swapInfo_type;
    MPI_Type_contiguous(2, MPI_INT, &swapInfo_type);
    MPI_Type_commit(&swapInfo_type);

    int blocklens[b], disps[b];
    for (i = 0; i < b; i++)
    {
        blocklens[i] = i;
        disps[i] = i * b;
    }
    MPI_Datatype lower_triangular;
    MPI_Type_indexed(b, blocklens, disps, MPI_DOUBLE, &lower_triangular);
    MPI_Type_commit(&lower_triangular);

    /* Send Column Block From A to AK. 
    Processor 0 SEND, others RECV. */
    if (myid == 0)
    {
        jb = 0;
        jb_0 = 0;
        /*Column block cyclic partitioning*/
        for (j = 0; j < q; j++)
        {
            // Copy A to AK
            for (i = 0; i < N; i++)
                for(k = jb_0, l = jb; k < jb_0 + b; k++, l++)
                    AK[i][k] = A[i][l];
            jb_0 += b;
            // Send blocks to other processors
            for (i = 1; i < numprocs; i++)
            {
                jb += b;
                MPI_Send(&A[0][jb], 1, column_block_global, i, 1, MPI_COMM_WORLD);
            }
            jb += b;

        }

        // Send remaining blocks, one block to each processe with myid < r
        if (r > 0)
        {
            for (i = 0; i < N; i++)
                    for(k = jb_0, l = jb; k < jb_0 + b; k++, l++)
                        AK[i][k] = A[i][l];  

            for (i = 1; i < r; i++)
            {
                jb += b;
                MPI_Send(&A[0][jb], 1, column_block_global, i, 1, MPI_COMM_WORLD);
            }
        }
    }

    else
    { // All other processes receive a submatrix from process 0

        jb = 0;
        for (i = 0; i < kb; i++)
        {
            MPI_Recv(&AK[0][jb], 1, column_block_local, 0, 1, MPI_COMM_WORLD, &status);
            jb += b;
        }
    }
    /* End of Send Column Block*/
    MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0)
        gettimeofday(&start_time, 0);

    ib = 0;
    ib_green = b;

    for (working_block = 0; working_block < Nb; working_block++)
    {
        int start_col = (working_block / numprocs) * b;
        int end_col = start_col + b;
        int start_row = working_block * b;
        int end_row = start_row + b;

        if (myid == working_block % numprocs)
        {
            /*Partial pivoting*/
            for (i = start_col, k = start_row; i < end_col; i++, k++)
            {
                amax = AK[k][i];
                indk = k;
                for (j = k + 1; j < N; j++)
                {
                    if (fabs(AK[j][i]) > fabs(amax))
                    {
                        amax = AK[j][i];
                        indk = j;
                    }
                }

                if (amax == 0.0)
                {
                    printf("Matrix is singular!\n");
                    exit(1);
                }

                // Swap rows if needed
                if (indk != k)
                {

                    for (j = 0; j < K; ++j)
                    {
                        c = AK[k][j];
                        AK[k][j] = AK[indk][j];
                        AK[indk][j] = c;
                    }
                }
                swapInfo[i - start_col].origin_row = k;
                swapInfo[i - start_col].target_row = indk;

                /*Gaussian elimination*/
                for (j = k + 1; j < N; j++)
                {
                    AK[j][i] /= AK[k][i];
                }
                for (j = k + 1; j < N; j++)
                {
                    c = AK[j][i];
                    for (l = i + 1; l < end_col; l++)
                    {
                        AK[j][l] -= c * AK[k][l];
                    }
                }
            }

            /*Brodcast pivoting information*/
            MPI_Bcast(swapInfo, b, swapInfo_type, working_block % numprocs, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Bcast(swapInfo, b, swapInfo_type, working_block % numprocs, MPI_COMM_WORLD);
            for (i = 0; i < b; i++)
                // Other blocks swap rows
                if (swapInfo[i].origin_row != swapInfo[i].target_row)
                {
                    for (j = 0; j < K; ++j)
                    {
                        c = AK[swapInfo[i].origin_row][j];
                        AK[swapInfo[i].origin_row][j] = AK[swapInfo[i].target_row][j];
                        AK[swapInfo[i].target_row][j] = c;
                    }
                }
        }
        if (myid == working_block % numprocs) {
            for (i = ib; i < ib + b; i++) {
                memcpy(AW[i], &AK[i][start_col], (i - ib) * sizeof(double));
            }
        }
        // Broadcast AW
        MPI_Bcast(&AW[ib][0], 1, lower_triangular, working_block % numprocs, MPI_COMM_WORLD);
        ib += b;       
        // Extract AW to LL
        for (i = 0; i < b; i++)
            for (j = 0; j < b; j++)
                if (j < i)
                    LL[i][j] = AW[start_row + i][j];
                else if (j == i)
                    LL[i][j] = 1;
                else
                    LL[i][j] = 0;

        if (myid <= working_block % numprocs) {
            for (i = 1; i < b; i++)
                for (j = end_col; j < K; j++)
                    for (k = 0; k < i; k++)
                        AK[i + start_row][j] -= LL[i][k] * AK[start_row + k][j];
        } else {
            for (i = 1; i < b; i++)
                for (j = start_col; j < K; j++)
                    for (k = 0; k < i; k++)
                        AK[i + start_row][j] -= LL[i][k] * AK[start_row + k][j];
        }

        /*Green part*/
        MPI_Datatype column_block_green;
        MPI_Type_vector(N - (working_block + 1) * b, b, b, MPI_DOUBLE, &column_block_green);
        MPI_Type_commit(&column_block_green);

        if (myid == working_block % numprocs && working_block != Nb - 1) {
            for (i = ib_green; i < N; i++) {
                memcpy(AW[i], &AK[i][start_col], b * sizeof(double));
            }
        }
        MPI_Bcast(&AW[ib_green][0], 1, column_block_green, working_block % numprocs, MPI_COMM_WORLD);
        ib_green += b;
        if (myid <= working_block % numprocs) 
        {
            for (i = end_row; i < N; i++)
                for (j = end_col; j < K; j++)
                    for (k = 0; k < b; k++)
                        AK[i][j] -= AW[i][k] * AK[start_row + k][j];

        } else 
        {
            for (i = end_row; i < N; i++)
                for (j = start_col; j < K; j++)
                    for (k = 0; k < b; k++)
                        AK[i][j] -= AW[i][k] * AK[start_row + k][j];
        }
    }
    /* Receive blocks */
    if (myid == 0)
    {
        jb = 0;
        jb_0 = 0;
        for (j = 0; j < q; j++)
        {
            for (i = 0; i < N; i++)
                for (k = jb, l = jb_0; k < jb + b; k++, l++)
                    A[i][k] = AK[i][l];
            jb_0 += b;
            for (i = 1; i < numprocs; i++)
            {
                jb += b;
                MPI_Recv(&A[0][jb], 1, column_block_global, i, 1, MPI_COMM_WORLD, &status);
            }
            jb += b;
            
        }
        if (r > 0)
        {// Receive remaining blocks from processes with myid < r
            for (i = 0; i < N; i++)
                for (k = jb, l = jb_0; k < jb + b; k++, l++)
                    A[i][k] = AK[i][l];
            jb_0 += b;
            for (i = 1; i < r; i++)
            {
                jb += b;
                MPI_Recv(&A[0][jb], 1, column_block_global, i, 1, MPI_COMM_WORLD, &status);
            }
        }
    }
    else
    {
        jb = 0;
        for (i = 0; i < kb; i++)
        {
            MPI_Send(&AK[0][jb], 1, column_block_local, 0, 1, MPI_COMM_WORLD);
            jb += b;
        }
    }

    if (myid == 0)
    {
        gettimeofday(&end_time, 0);
        seconds = end_time.tv_sec - start_time.tv_sec;
        microseconds = end_time.tv_usec - start_time.tv_usec;

        elapsed = seconds + 1e-6 * microseconds;
        printf("MPI calculation time: %f\n\n", elapsed);
        int cnt;
        cnt = test(A, a, N);
        if (cnt == 0)
            printf("Done. There are no differences!\n");        
        else
            printf("Results are incorrect! The number of different elements is %d\n", cnt);
    }

    MPI_Finalize();

    return 0;
}

void print_matrix(double **T, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%6.3f ", T[i][j]);
        }
    }
    printf("\n\n");
    return;
}

int test(double **t1, double **t2, int rows)
{
    int i, j;
    int cnt;
    cnt = 0;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            if ((t1[i][j] - t2[i][j]) * (t1[i][j] - t2[i][j]) > 1.0e-16)
            {
                cnt += 1;
            }
        }
    }

    return cnt;
}
void gepp_origin(double **A, int n) {
    int i, j, k, indk;
    double amax, c;

    for (i = 0; i < n - 1; i++) {
        // PP
        amax = A[i][i];
        indk = i;
        for (k = i + 1; k < n; k++) 
        {
            if (fabs(A[k][i]) > fabs(amax)) 
            {
                amax = A[k][i];
                indk = k;
            }
        }

        if (amax == 0) 
        {
            printf("matrix is singular!\n");
            exit(1);
        } else if (indk != i) 
        {
            for (j = 0; j < n; j++) 
            {
                c = A[i][j];
                A[i][j] = A[indk][j];
                A[indk][j] = c;
            }
        }
        // End of PP
        for (k = i + 1; k < n; k++) 
        {
            A[k][i] = A[k][i] / A[i][i];
        }

        for (k = i + 1; k < n; k++) {
            c = A[k][i];
            for (j = i + 1; j < n; j++) {
                A[k][j] -= c * A[i][j];
            }
        }
    }
}

